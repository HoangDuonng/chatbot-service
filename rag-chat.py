import pandas as pd
import google.generativeai as genai
import numpy as np
import chromadb
from flask import Flask, request, jsonify, session
from flask_cors import CORS
import uuid
from datetime import datetime, timedelta
import time
# from agents import Agent, Runner, function_tool
from dotenv import load_dotenv
import os


# Rate limiting (đã comment để code core hoạt động tự do)
# request_timestamps = {}
# RATE_LIMIT_WINDOW = 60  # 60 giây
# MAX_REQUESTS_PER_WINDOW = 10  # Tối đa 10 requests trong 60 giây

# Cache cho embedding và responses
embedding_cache = {}
response_cache = {}
CACHE_EXPIRY = 3600  

load_dotenv()
gemini_api_key = os.getenv("GEMINI_KEY")
genai.configure(api_key=gemini_api_key)
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL")


app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  
CORS(app, supports_credentials=True)  

chroma_client = chromadb.PersistentClient("db")
collection_name = "hotel_saigon"

model = genai.GenerativeModel('gemini-1.5-flash')

# Hàm tạo embedding từ Gemini
def get_embedding(text: str) -> list[float]:
    # Kiểm tra cache trước
    current_time = time.time()
    if text in embedding_cache:
        cached_time, cached_embedding = embedding_cache[text]
        if current_time - cached_time < CACHE_EXPIRY:
            return cached_embedding
    
    try:
        response = genai.embed_content(
            model=GEMINI_EMBEDDING_MODEL,
            content=text,
            task_type="retrieval_query"
        )
        embedding = response["embedding"]
        
        # Lưu vào cache
        embedding_cache[text] = (current_time, embedding)
        return embedding
    except Exception as e:
        print(f"❌ Lỗi khi tạo embedding: {e}")
        return []

def generate_natural_response(query: str, search_results: str) -> str:
    """Sử dụng Gemini để tạo câu trả lời tự nhiên dựa trên kết quả tìm kiếm"""
    
    prompt = f"""
Bạn là một trợ lý tư vấn khách sạn thân thiện. Dựa trên thông tin khách sạn được cung cấp, hãy trả lời câu hỏi của khách hàng một cách tự nhiên và hữu ích.

Câu hỏi của khách hàng: {query}

Thông tin khách sạn tìm được:
{search_results}

Hướng dẫn trả lời:
- Trả lời trực tiếp câu hỏi của khách hàng
- Sử dụng ngôn ngữ tự nhiên, thân thiện
- Nếu có nhiều khách sạn phù hợp, hãy đề cập đến tất cả
- Nếu không tìm thấy thông tin, hãy nói rõ ràng
- Không cần liệt kê tất cả thông tin, chỉ trả lời những gì được hỏi
- Sử dụng tiếng Việt
- Khi nói về đánh giá, hãy giải thích ý nghĩa của điểm số (ví dụ: 8.0/10 là rất tốt)
- Nếu có thông tin về giá, hãy format đẹp (ví dụ: 368,781 VND)

Trả lời:
"""
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"❌ Lỗi khi tạo câu trả lời: {e}")
        # Fallback: Tạo câu trả lời đơn giản từ search results
        if "rate" in str(e).lower() or "quota" in str(e).lower():
            return create_simple_response(query, search_results)
        return search_results  # Fallback về kết quả thô nếu có lỗi khác

def simple_text_search(query: str) -> dict:
    """Tìm kiếm đơn giản bằng text khi embedding API bị lỗi"""
    try:
        collection = chroma_client.get_collection(name=collection_name)
        
        # Lấy tất cả documents
        all_results = collection.get()
        documents = all_results.get("documents", [])
        metadatas = all_results.get("metadatas", [])
        
        # Tìm kiếm đơn giản bằng từ khóa
        query_lower = query.lower()
        matched_results = []
        
        for i, metadata in enumerate(metadatas):
            if isinstance(metadata, dict):
                information = metadata.get("information", "")
                if information and any(keyword in information.lower() for keyword in query_lower.split()):
                    matched_results.append((i, information))
        
        # Sắp xếp theo độ phù hợp (đơn giản)
        matched_results.sort(key=lambda x: sum(1 for keyword in query_lower.split() if keyword in x[1].lower()), reverse=True)
        
        # Format kết quả
        search_result = ""
        for idx, (i, information) in enumerate(matched_results[:3]):  # Chỉ lấy 3 kết quả đầu
            formatted_info = format_hotel_info(information, idx+1)
            search_result += formatted_info + "\n\n"
        
        if not search_result.strip():
            return {"response": "Không tìm thấy thông tin phù hợp.", "search_results": ""}
        
        # Tạo câu trả lời đơn giản
        simple_response = create_simple_response(query, search_result)
        
        return {
            "response": simple_response,
            "search_results": search_result
        }
        
    except Exception as e:
        print(f"❌ Lỗi trong simple text search: {e}")
        return {"response": "Không thể tìm kiếm thông tin.", "search_results": ""}

def create_simple_response(query: str, search_results: str) -> str:
    """Tạo câu trả lời đơn giản khi Gemini API bị rate limit"""
    try:
        # Phân tích query để hiểu ý định
        query_lower = query.lower()
        
        if "5 sao" in query_lower or "năm sao" in query_lower:
            # Tìm khách sạn 5 sao
            hotels = []
            lines = search_results.split('\n')
            for line in lines:
                if "5.0" in line or "5 sao" in line:
                    # Tìm tên khách sạn
                    for prev_line in reversed(lines[:lines.index(line)]):
                        if "Tên khách sạn:" in prev_line:
                            hotel_name = prev_line.split("Tên khách sạn:")[1].strip()
                            hotels.append(hotel_name)
                            break
            
            if hotels:
                return f"Dựa trên tìm kiếm, tôi tìm thấy {len(hotels)} khách sạn 5 sao:\n" + "\n".join([f"• {hotel}" for hotel in hotels[:2]])
            else:
                return "Tôi không tìm thấy khách sạn 5 sao trong kết quả tìm kiếm."
        
        elif "đánh giá" in query_lower:
            # Tìm thông tin đánh giá
            for line in search_results.split('\n'):
                if "Điểm đánh giá:" in line:
                    rating = line.split("Điểm đánh giá:")[1].strip()
                    return f"Khách sạn này có điểm đánh giá: {rating}/10"
            
            return "Tôi không tìm thấy thông tin đánh giá cụ thể."
        
        else:
            # Trả về kết quả tìm kiếm đã format
            return f"Dựa trên tìm kiếm của bạn '{query}', đây là thông tin tôi tìm được:\n\n{search_results}"
            
    except Exception as e:
        print(f"❌ Lỗi khi tạo simple response: {e}")
        return search_results

def format_hotel_info(information: str, index: int) -> str:
    """Format thông tin khách sạn để dễ đọc hơn"""
    try:
        # Tách các thông tin bằng dấu chấm và dấu phẩy
        parts = []
        
        # Tách theo dấu chấm trước
        dot_parts = information.split('.')
        for part in dot_parts:
            part = part.strip()
            if part:
                # Nếu có dấu phẩy, tách tiếp
                if ',' in part:
                    comma_parts = part.split(',')
                    for comma_part in comma_parts:
                        comma_part = comma_part.strip()
                        if comma_part:
                            parts.append(comma_part)
                else:
                    parts.append(part)
        
        # Format từng phần
        formatted_parts = []
        for part in parts:
            if part:
                # Thêm bullet point cho mỗi thông tin
                formatted_parts.append(f"• {part}")
        
        # Ghép lại với format đẹp
        formatted_info = f"{index}). Khách sạn:\n" + "\n".join(formatted_parts)
        return formatted_info
        
    except Exception as e:
        print(f"❌ Lỗi khi format thông tin: {e}")
        return f"{index}). {information}"

def rag(query: str) -> dict:
    print("----Query:", query)

    try:
        collection = chroma_client.get_collection(name=collection_name)
    except Exception as e:
        print(f"❌ Lỗi khi kết nối ChromaDB: {e}")
        return {"response": "Không thể kết nối đến cơ sở dữ liệu.", "search_results": ""}

    # Tạo embedding từ Gemini cho truy vấn
    try:
        query_embedding = get_embedding(query)
        if not query_embedding:
            return {"response": "Không thể tạo embedding cho câu hỏi.", "search_results": ""}
    except Exception as e:
        print(f"❌ Lỗi khi tạo embedding: {e}")
        # Fallback: Tìm kiếm đơn giản bằng text
        return simple_text_search(query)

    # Truy vấn ChromaDB
    try:
        search_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=5
        )
    except Exception as e:
        print(f"❌ Lỗi khi truy vấn ChromaDB: {e}")
        return {"response": "Không thể tìm kiếm trong cơ sở dữ liệu.", "search_results": ""}

    # Xử lý kết quả tìm kiếm
    metadatas = search_results.get("metadatas", [])
    documents = search_results.get("documents", [])
    
    search_result = ""
    
    if not metadatas or not metadatas[0]:
        return {"response": "Không tìm thấy thông tin phù hợp.", "search_results": ""}
    
    # Lấy metadata đầu tiên (vì chỉ có 1 query)
    metadata_list = metadatas[0]
    
    for i, metadata in enumerate(metadata_list):
        if isinstance(metadata, dict):
            information = metadata.get("information", "")
            if information.strip():
                # Format thông tin đẹp hơn
                formatted_info = format_hotel_info(information, i+1)
                search_result += formatted_info + "\n\n"
    
    if not search_result.strip():
        return {"response": "Không tìm thấy thông tin phù hợp.", "search_results": ""}
    
    # Tạo câu trả lời tự nhiên bằng Gemini
    natural_response = generate_natural_response(query, search_result)
    
    return {
        "response": natural_response,
        "search_results": search_result
    }

# Rate limiting function
# def check_rate_limit(session_id):
#     """Kiểm tra rate limit cho session"""
#     current_time = time.time()
#     
#     if session_id not in request_timestamps:
#         request_timestamps[session_id] = []
#     
#     # Xóa các request cũ hơn window
#     request_timestamps[session_id] = [
#         ts for ts in request_timestamps[session_id] 
#         if current_time - ts < RATE_LIMIT_WINDOW
#     ]
#     
#     # Kiểm tra số lượng request
#     if len(request_timestamps[session_id]) >= MAX_REQUESTS_PER_WINDOW:
#         return False
#     
#     # Thêm timestamp hiện tại
#     request_timestamps[session_id].append(current_time)
#     return True

# API Endpoints
@app.route('/chat', methods=['POST'])
def chat():
    """Endpoint để chat với RAG system"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'Thiếu dữ liệu trong request'
            }), 400
        
        # Kiểm tra action trước
        action = data.get('action')
        session_id = data.get('session_id') or str(uuid.uuid4())
        
        # Xử lý action END_SESSION
        if action == 'end_session' or (data.get('message') == 'END_SESSION'):
            return jsonify({
                'success': True,
                'message': 'Session đã được kết thúc',
                'session_id': session_id
            })
        
    # Kiểm tra rate limit (đã comment để code core hoạt động tự do)
    # if not check_rate_limit(session_id):
    #     return jsonify({
    #         'success': False,
    #         'error': 'Quá nhiều request. Vui lòng thử lại sau 1 phút.'
    #     }), 429
        
        # Xử lý chat bình thường
        if 'message' not in data:
            return jsonify({
                'success': False,
                'error': 'Thiếu thông tin message trong request'
            }), 400
        
        user_message = data['message']
        
        if not user_message.strip():
            return jsonify({
                'success': False,
                'error': 'Message không được để trống'
            }), 400
        
        # Gọi hàm RAG để xử lý câu hỏi
        rag_result = rag(user_message)
        
        return jsonify({
            'success': True,
            'response': rag_result['response'],
            'query': user_message,
            'session_id': session_id
        })
        
    except Exception as e:
        print(f"❌ Lỗi trong API: {e}")
        return jsonify({
            'success': False,
            'error': 'Lỗi server nội bộ'
        }), 500



@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint kiểm tra sức khỏe của API"""
    return jsonify({
        'status': 'healthy',
        'message': 'RAG Chat API đang hoạt động',
        'features': [
            'Chat với RAG',
            'Tìm kiếm khách sạn',
            'Tư vấn thông tin'
        ]
    })

@app.route('/', methods=['GET'])
def home():
    """Endpoint trang chủ"""
    return jsonify({
        'message': 'RAG Chat API với Memory System',
        'endpoints': {
            'chat': '/chat (POST)',
            'health': '/health (GET)'
        },
        'usage': {
            'chat': 'POST /chat với body {"message": "câu hỏi"}'
        }
    })

# Terminal chat mode (giữ lại để test)
def terminal_chat():
    """Chế độ chat qua terminal"""
    print("🤖 RAG Chat Bot - Terminal Mode")
    print("Gõ '0' để thoát")
    print("-" * 50)
    
    while True:
        user_input = input("You: ")
        if user_input.strip() == "0":
            print("Tạm biệt!")
            break
        
        # Xử lý RAG
        rag_result = rag(user_input)
        
        print("Bot:", rag_result['response'])

if __name__ == '__main__':
    import sys
    import warnings
    
    # Ẩn warning của Flask development server
    warnings.filterwarnings("ignore", message=".*development server.*")
    
    # Kiểm tra argument để chọn mode
    if len(sys.argv) > 1 and sys.argv[1] == '--terminal':
        # Chạy terminal chat mode
        terminal_chat()
    else:
        # Chạy API server
        print("🚀 Khởi động RAG Chat API với Memory System...")
        print("📡 API đang chạy tại: http://localhost:5000")
        print("💡 Để chạy terminal mode: python rag-chat.py --terminal")
        print("⚠️  Lưu ý: Đây là development server, không dùng cho production")
        print("-" * 60)
        
        # Cấu hình Flask để ẩn warning
        app.run(
            debug=True, 
            host='0.0.0.0', 
            port=5000,
            use_reloader=False  # Tắt auto-reload để giảm warning
        )
