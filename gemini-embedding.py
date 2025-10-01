# # Save data to Chroma DB
import os
import pandas as pd
from flask import Flask, request, jsonify, Blueprint
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import re
import chromadb
import uuid
import google.generativeai as genai
from google.generativeai import embedding

import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

chroma_client = chromadb.PersistentClient("db")

load_dotenv()

# # Flask App Initialization
app = Flask(__name__)


# Gemini Client for embeddings
genai.configure(api_key=os.getenv("GEMINI_KEY"))
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL")

training_bp = Blueprint('training', __name__, url_prefix='/training')

def get_embedding(text: str) -> list[float]:
    try:
        response = genai.embed_content(
            model=GEMINI_EMBEDDING_MODEL,
            content=text,
            task_type="retrieval_document"
        )
        return response["embedding"]
    except Exception as e:
        logging.error(f"❌ Lỗi khi embedding đoạn: {text[:60]}... -> {e}")
        return []

def sanitize_collection_name(name: str) -> str:
    """Sanitize collection name to be MongoDB-compatible."""
    name = re.sub(r'[^a-zA-Z0-9_]', '_', name)  # Replace invalid characters with '_'
    name = name.strip("_")  # Remove leading/trailing underscores
    return name.lower()  # Convert to lowercase for consistency

def sanitize_metadata(record):
    """Convert None values in metadata to empty string or default values, and exclude embeddings."""
    sanitized_record = {
        k: (str(v) if v is not None else "") for k, v in record.items() if k != "embedding"
    }
    return sanitized_record

df = pd.read_excel("data/traveloka_result_20hotels_converted_20250705_data_RAG.xlsx")
# print(df.columns.tolist())

def join_hotel_info(row):
    parts = []
    if pd.notna(row['Tên khách sạn']):
        parts.append(f"Tên khách sạn: {row['Tên khách sạn']}")
    else:
        parts.append("Tên khách sạn không xác định")
    if pd.notna(row['Địa chỉ']):
        parts.append(f"Địa chỉ: {row['Địa chỉ']}")
    else:
        parts.append("Địa chỉ không có")
    if pd.notna(row['Số sao']):
        parts.append(f"Số sao: {row['Số sao']}")
    else:
        parts.append("Số sao không có")
    if pd.notna(row['Điểm đánh giá']):
        parts.append(f"Điểm đánh giá: {row['Điểm đánh giá']}")
    else:
        parts.append("Điểm đánh giá không có")
    if pd.notna(row['Mô tả đánh giá']):
        parts.append(f"Mô tả đánh giá: {row['Mô tả đánh giá']}")
    else:
        parts.append("Mô tả đánh giá không có")
    if pd.notna(row['Giá thấp nhất']):
        parts.append(f"Giá thấp nhất: {row['Giá thấp nhất']}")
    else:
        parts.append("Giá thấp nhất không có")
    if pd.notna(row['Tiện ích nổi bật']):
        parts.append(f"Tiện ích nổi bật: {row['Tiện ích nổi bật']}")
    else:
        parts.append("Tiện ích nổi bật không có")
    return ". ".join(parts)

# Áp dụng hàm để tạo cột 'information' mới
df['information'] = df.apply(join_hotel_info, axis=1)
df = df[df['information'].notna()].reset_index(drop=True)

# Gán embedding theo từng dòng, log theo batch
embeddings = []
batch_size = 50

logging.info(f"🚀 Bắt đầu xử lý embedding cho {len(df)} bản ghi...")

for idx, row in df.iterrows():
    embedding = get_embedding(row["information"])
    embeddings.append(embedding)
    if (idx + 1) % batch_size == 0 or (idx + 1) == len(df):
        logging.info(f"✅ Đã xử lý {idx + 1}/{len(df)} bản ghi.")

print(df['information'].head(2))

# Display the DataFrame to confirm
print(df.head())

# df = df.head(2) 

# Prepare data
df = df[df['information'].notna()]
df["embedding"] = df["information"].apply(get_embedding)

print(df["embedding"].head(2))
# Metadata
metadatas = [{"information": row["information"]} for _, row in df.iterrows()]
ids = [str(uuid.uuid4()) for _ in range(len(df))]

# ChromaDB setup
chroma_client = chromadb.PersistentClient("db")
collection_name = "hotel_saigon"
collection = chroma_client.get_or_create_collection(name=collection_name)

# Insert data
collection.add(
    ids=ids,
    embeddings=df["embedding"].tolist(),
    metadatas=metadatas
)

print(f"Inserted {len(df)} documents into ChromaDB collection '{collection.name}'.")