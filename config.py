
import os
from dotenv import load_dotenv

load_dotenv()

# === Paths ===
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data-train")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", os.path.join(BASE_DIR, "output"))

CRM_FILE = os.environ.get("CRM_FILE", os.path.join(DATA_DIR, "crm_customer_book_revenue_20260418.csv"))
CHAT_FILE = os.environ.get("CHAT_FILE", os.path.join(DATA_DIR, "raw_message_convertation_20260418.xlsx"))
CHATPAGE_FILE = os.environ.get("CHATPAGE_FILE", os.path.join(DATA_DIR, "crm_chatpage_customer_20260416.csv"))

# Resolve relative paths
if not os.path.isabs(CRM_FILE):
    CRM_FILE = os.path.join(BASE_DIR, CRM_FILE)
if not os.path.isabs(CHAT_FILE):
    CHAT_FILE = os.path.join(BASE_DIR, CHAT_FILE)
if not os.path.isabs(CHATPAGE_FILE):
    CHATPAGE_FILE = os.path.join(BASE_DIR, CHATPAGE_FILE)

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

# === Outcome Weighting (v3.1) ===
OUTCOME_WEIGHTS = {
    "paid_high": 1.5,    # paid > 5M
    "show_up": 1.3,      # booking_count > 0 + paid > 0
    "booked": 1.2,       # có booking
    "phone_only": 1.0,   # chỉ có SĐT
    "no_conversion": 0.5, # không chuyển đổi
}

# === Top Sales Identification ===
TOP_SALES_PERCENTILE = float(os.environ.get("TOP_SALES_PERCENTILE", "0.3"))

# === Embedding ===
# EMBED_ENABLED: bật/tắt việc gọi LLM để embed (False = dùng random/zero vector)
EMBED_ENABLED = os.environ.get("EMBED_ENABLED", "true").lower() in ("1", "true", "yes")
# EMBED_PROVIDER: "openai" | "local" (sentence-transformers, chạy offline)
EMBED_PROVIDER = os.environ.get("EMBED_PROVIDER", "openai")
# EMBED_LOCAL_MODEL: model name khi dùng provider "local"
EMBED_LOCAL_MODEL = os.environ.get("EMBED_LOCAL_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# === Conversation States (FSM - v3.1) ===
CONVERSATION_STATES = [
    "greeting", "qualifying", "presenting",
    "handling_objection", "closing", "follow_up",
]

STATE_KEYWORDS = {
    "greeting": ["chào", "hello", "hi", "xin chào", "alo"],
    "qualifying": ["quan tâm", "muốn", "cần", "tư vấn", "cho hỏi", "hỏi"],
    "presenting": ["liệu trình", "dịch vụ", "gồm", "bao lâu", "hiệu quả", "kết quả"],
    "handling_objection": ["đắt", "sợ", "đau", "lo", "không biết", "ngại",
                           "tác dụng phụ", "an toàn", "nguy hiểm"],
    "closing": ["đặt lịch", "book", "hẹn", "sđt", "số điện thoại", "zalo",
                "đăng ký", "liên hệ", "slot", "giữ chỗ"],
    "follow_up": ["nhắc", "hôm trước", "lần trước", "còn quan tâm", "quay lại"],
}

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
