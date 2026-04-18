
import os
from dotenv import load_dotenv

load_dotenv()

# === Paths ===
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data-train")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", os.path.join(BASE_DIR, "output"))

CRM_FILE = os.environ.get("CRM_FILE", os.path.join(DATA_DIR, "crm_customer_book_revenue_20260418.csv"))
CHAT_FILE = os.environ.get("CHAT_FILE", os.path.join(DATA_DIR, "raw_message_convertation_20260418.xlsx"))

# Resolve relative paths
if not os.path.isabs(CRM_FILE):
    CRM_FILE = os.path.join(BASE_DIR, CRM_FILE)
if not os.path.isabs(CHAT_FILE):
    CHAT_FILE = os.path.join(BASE_DIR, CHAT_FILE)

# === Qdrant ===
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "conversation_memory")
VECTOR_SIZE = int(os.environ.get("VECTOR_SIZE", "1536"))
TOP_K = int(os.environ.get("TOP_K", "5"))

# === Segment Rules (theo paid_value) ===
SEGMENT_RULES = {
    "LOW": (0, 500_000),
    "MID": (500_000, 3_000_000),
    "HIGH": (3_000_000, 10_000_000),
    "VIP": (10_000_000, float("inf")),
}

# === Scoring ===
SCORE_THRESHOLD = int(os.environ.get("SCORE_THRESHOLD", "70"))

# === Cleaning ===
SYSTEM_MESSAGE_PATTERNS = [
    "đã trả lời một quảng cáo",
    "đã gửi một liên kết",
    "đã gửi một ảnh",
    "đã gửi một video",
    "đã gửi một sticker",
    "đã bắt đầu cuộc trò chuyện",
    "đã thay đổi chủ đề",
]

# Tên page (assistant) — dùng để phân biệt role
PAGE_NAMES = ["DƯỠNG", "Dưỡng", "dưỡng"]

# === Chunking ===
MAX_TURNS_PER_CHUNK = 10
MIN_TURNS_PER_CHUNK = 3
