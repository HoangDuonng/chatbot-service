import pandas as pd
import google.generativeai as genai
import numpy as np
import chromadb
from flask import Flask, request, jsonify
from flask_cors import CORS
# from agents import Agent, Runner, function_tool
from dotenv import load_dotenv
import os

# Load .env và lấy API key của Gemini
load_dotenv()
gemini_api_key = os.getenv("GEMINI_KEY")
genai.configure(api_key=gemini_api_key)
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL")

# Khởi tạo Flask app
app = Flask(__name__)
CORS(app)  # Cho phép frontend gọi API

# Khởi tạo ChromaDB - sử dụng đường dẫn tương đối giống gemini-embedding.py
chroma_client = chromadb.PersistentClient("db")
collection_name = "hotel_saigon"

# Khởi tạo Gemini model cho chat
model = genai.GenerativeModel('gemini-1.5-flash')

# Hàm tạo embedding từ Gemini
def get_embedding(text: str) -> list[float]:
    try:
        response = genai.embed_content(
            model=GEMINI_EMBEDDING_MODEL,
            content=text,
            task_type="retrieval_query"
        )
        return response["embedding"]
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

Trả lời:
"""
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"❌ Lỗi khi tạo câu trả lời: {e}")
        return search_results  # Fallback về kết quả thô nếu có lỗi

def rag(query: str) -> str:
    print("----Query:", query)

    try:
        collection = chroma_client.get_collection(name=collection_name)
    except Exception as e:
        print(f"❌ Lỗi khi kết nối ChromaDB: {e}")
        return "Không thể kết nối đến cơ sở dữ liệu."

    # Tạo embedding từ Gemini cho truy vấn
    query_embedding = get_embedding(query)
    if not query_embedding:
        return "Không thể tạo embedding cho câu hỏi."

    # Truy vấn ChromaDB
    try:
        search_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=5
        )
    except Exception as e:
        print(f"❌ Lỗi khi truy vấn ChromaDB: {e}")
        return "Không thể tìm kiếm trong cơ sở dữ liệu."

    # Xử lý kết quả tìm kiếm
    metadatas = search_results.get("metadatas", [])
    documents = search_results.get("documents", [])
    
    search_result = ""
    
    if not metadatas or not metadatas[0]:
        return "Không tìm thấy thông tin phù hợp."
    
    # Lấy metadata đầu tiên (vì chỉ có 1 query)
    metadata_list = metadatas[0]
    
    for i, metadata in enumerate(metadata_list):
        if isinstance(metadata, dict):
            information = metadata.get("information", "")
            if information.strip():
                search_result += f"{i+1}). {information}\n\n"
    
    if not search_result.strip():
        search_result = "Không tìm thấy thông tin phù hợp."
        return search_result
    
    # Tạo câu trả lời tự nhiên bằng Gemini
    natural_response = generate_natural_response(query, search_result)
    return natural_response

# API Endpoints
@app.route('/chat', methods=['POST'])
def chat():
    """Endpoint để chat với RAG system"""
    try:
        data = request.get_json()
        
        if not data or 'message' not in data:
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
        response = rag(user_message)
        
        return jsonify({
            'success': True,
            'response': response,
            'query': user_message
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
        'message': 'RAG Chat API đang hoạt động'
    })

@app.route('/', methods=['GET'])
def home():
    """Endpoint trang chủ"""
    return jsonify({
        'message': 'RAG Chat API',
        'endpoints': {
            'chat': '/chat (POST)',
            'health': '/health (GET)'
        },
        'usage': {
            'chat': 'POST /chat với body {"message": "câu hỏi của bạn"}'
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
        response = rag(user_input)
        print("Bot:", response)

if __name__ == '__main__':
    import sys
    
    # Kiểm tra argument để chọn mode
    if len(sys.argv) > 1 and sys.argv[1] == '--terminal':
        # Chạy terminal chat mode
        terminal_chat()
    else:
        # Chạy API server
        print("🚀 Khởi động RAG Chat API...")
        print("📡 API đang chạy tại: http://localhost:5000")
        print("💡 Để chạy terminal mode: python rag-chat.py --terminal")
        app.run(debug=True, host='0.0.0.0', port=5000)
