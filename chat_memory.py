import json
import uuid
from datetime import datetime
from typing import List, Dict, Optional
import threading
from pymongo import MongoClient

class ChatMemory:
    """Quản lý memory cho chat system"""
    
    def __init__(self, use_mongodb: bool = False):
        self.session_memory = {}  # In-memory storage cho session
        self.lock = threading.Lock()
        self.use_mongodb = use_mongodb
        
        # Khởi tạo MongoDB nếu cần
        if self.use_mongodb:
            self._init_mongodb()
    
    def _init_mongodb(self):
        """Khởi tạo MongoDB connection"""
        try:
            # Kết nối MongoDB (có thể thay đổi connection string)
            self.mongo_client = MongoClient('mongodb://localhost:27017/')
            self.db = self.mongo_client['chatbot_db']
            self.conversations_collection = self.db['conversations']
            self.messages_collection = self.db['messages']
            
            # Tạo indexes
            self.conversations_collection.create_index("session_id")
            self.messages_collection.create_index("conversation_id")
            self.messages_collection.create_index("created_at")
            
            print("✅ Kết nối MongoDB thành công")
        except Exception as e:
            print(f"❌ Lỗi kết nối MongoDB: {e}")
            self.use_mongodb = False
    
    def create_conversation(self, session_id: str, title: str = None) -> str:
        """Tạo conversation mới"""
        conversation_id = str(uuid.uuid4())
        
        if self.use_mongodb:
            # Lưu vào MongoDB
            conversation_doc = {
                'id': conversation_id,
                'session_id': session_id,
                'title': title,
                'created_at': datetime.now(),
                'updated_at': datetime.now()
            }
            self.conversations_collection.insert_one(conversation_doc)
        
        # Khởi tạo session memory
        self.session_memory[session_id] = {
            'conversation_id': conversation_id,
            'messages': [],
            'context': {}
        }
        
        return conversation_id
    
    def add_message(self, session_id: str, role: str, content: str, metadata: Dict = None):
        """Thêm tin nhắn vào conversation"""
        with self.lock:
            # Lấy conversation_id từ session memory
            if session_id not in self.session_memory:
                conversation_id = self.create_conversation(session_id)
            else:
                conversation_id = self.session_memory[session_id]['conversation_id']
            
            message_id = str(uuid.uuid4())
            
            if self.use_mongodb:
                # Lưu vào MongoDB (long-term memory)
                message_doc = {
                    'id': message_id,
                    'conversation_id': conversation_id,
                    'role': role,
                    'content': content,
                    'metadata': metadata or {},
                    'created_at': datetime.now()
                }
                self.messages_collection.insert_one(message_doc)
                
                # Cập nhật updated_at của conversation
                self.conversations_collection.update_one(
                    {'id': conversation_id},
                    {'$set': {'updated_at': datetime.now()}}
                )
            
            # Lưu vào session memory (short-term memory)
            message = {
                'id': message_id,
                'role': role,
                'content': content,
                'metadata': metadata or {},
                'timestamp': datetime.now().isoformat()
            }
            
            self.session_memory[session_id]['messages'].append(message)
            
            # Giới hạn số tin nhắn trong session memory (tránh memory leak)
            # Mỗi session chỉ lưu tối đa 50 tin nhắn gần nhất
            if len(self.session_memory[session_id]['messages']) > 50:
                self.session_memory[session_id]['messages'] = self.session_memory[session_id]['messages'][-50:]
    
    def get_conversation_history(self, session_id: str, limit: int = 10) -> List[Dict]:
        """Lấy lịch sử conversation từ database"""
        if self.use_mongodb:
            # Lấy từ MongoDB
            pipeline = [
                {'$match': {'session_id': session_id}},
                {'$lookup': {
                    'from': 'messages',
                    'localField': 'id',
                    'foreignField': 'conversation_id',
                    'as': 'messages'
                }},
                {'$unwind': '$messages'},
                {'$sort': {'messages.created_at': -1}},
                {'$limit': limit},
                {'$project': {
                    'id': '$messages.id',
                    'role': '$messages.role',
                    'content': '$messages.content',
                    'metadata': '$messages.metadata',
                    'timestamp': '$messages.created_at'
                }}
            ]
            
            messages = list(self.conversations_collection.aggregate(pipeline))
            return list(reversed(messages))  # Trả về theo thứ tự thời gian
        else:
            # Chỉ trả về session memory (short-term)
            if session_id in self.session_memory:
                return self.session_memory[session_id]['messages'][-limit:]
            return []
    
    def get_session_context(self, session_id: str) -> Dict:
        """Lấy context hiện tại của session"""
        if session_id in self.session_memory:
            return {
                'conversation_id': self.session_memory[session_id]['conversation_id'],
                'messages': self.session_memory[session_id]['messages'][-10:],  # 10 tin nhắn gần nhất cho context
                'context': self.session_memory[session_id]['context']
            }
        return {}
    
    def update_session_context(self, session_id: str, context: Dict):
        """Cập nhật context của session"""
        if session_id not in self.session_memory:
            self.session_memory[session_id] = {
                'conversation_id': None,
                'messages': [],
                'context': {}
            }
        
        self.session_memory[session_id]['context'].update(context)
    
    def get_conversation_summary(self, session_id: str) -> Dict:
        """Lấy summary của conversation"""
        if self.use_mongodb:
            # Lấy từ MongoDB
            pipeline = [
                {'$match': {'session_id': session_id}},
                {'$lookup': {
                    'from': 'messages',
                    'localField': 'id',
                    'foreignField': 'conversation_id',
                    'as': 'messages'
                }},
                {'$project': {
                    'conversation_id': '$id',
                    'title': '$title',
                    'created_at': '$created_at',
                    'updated_at': '$updated_at',
                    'message_count': {'$size': '$messages'}
                }}
            ]
            
            result = list(self.conversations_collection.aggregate(pipeline))
            return result[0] if result else {}
        else:
            # Chỉ trả về session memory summary
            if session_id in self.session_memory:
                return {
                    'conversation_id': self.session_memory[session_id]['conversation_id'],
                    'message_count': len(self.session_memory[session_id]['messages'])
                }
            return {}
    
    def clear_session(self, session_id: str):
        """Xóa session memory (không xóa database)"""
        if session_id in self.session_memory:
            del self.session_memory[session_id]
    
    def get_all_conversations(self, session_id: str) -> List[Dict]:
        """Lấy tất cả conversations của một session"""
        if self.use_mongodb:
            # Lấy từ MongoDB
            pipeline = [
                {'$match': {'session_id': session_id}},
                {'$lookup': {
                    'from': 'messages',
                    'localField': 'id',
                    'foreignField': 'conversation_id',
                    'as': 'messages'
                }},
                {'$project': {
                    'id': '$id',
                    'title': '$title',
                    'created_at': '$created_at',
                    'updated_at': '$updated_at',
                    'message_count': {'$size': '$messages'}
                }},
                {'$sort': {'updated_at': -1}}
            ]
            
            return list(self.conversations_collection.aggregate(pipeline))
        else:
            # Chỉ trả về session memory
            if session_id in self.session_memory:
                return [{
                    'id': self.session_memory[session_id]['conversation_id'],
                    'message_count': len(self.session_memory[session_id]['messages'])
                }]
            return []

# Global instance - chỉ dùng session memory (không dùng MongoDB)
chat_memory = ChatMemory(use_mongodb=False) 
