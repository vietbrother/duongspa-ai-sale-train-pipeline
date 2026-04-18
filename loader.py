import pandas as pd
import re
from config import CRM_FILE, CHAT_FILE, PAGE_NAMES, SYSTEM_MESSAGE_PATTERNS


def load_crm(file_path: str = None) -> pd.DataFrame:
    """Load CRM data từ CSV/Excel."""
    file_path = file_path or CRM_FILE
    df = pd.read_csv(file_path) if file_path.endswith(".csv") else pd.read_excel(file_path)

    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    if "phone" in df.columns:
        df["phone"] = df["phone"].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    df["total_paid_value"] = pd.to_numeric(df.get("total_paid_value", 0), errors="coerce").fillna(0)
    df["booking_count"] = pd.to_numeric(df.get("booking_count", 0), errors="coerce").fillna(0)
    return df


def load_conversations(file_path: str = None) -> pd.DataFrame:
    """Load và clean raw conversation data từ Pancake export."""
    file_path = file_path or CHAT_FILE
    df = pd.read_excel(file_path)

    # Map cột tiếng Việt → chuẩn
    col_map = {
        "Mã tin nhắn": "message_id",
        "Kiểu tin nhắn": "message_type",
        "Thẻ hội thoại": "tags",
        "Nội dung tin nhắn": "message",
        "Tên người gửi": "sender_name",
        "Mã người gửi": "sender_id",
        "Mã trang (page_id)": "page_id",
        "Mã hội thoại (thread_key)": "thread_key",
        "Mã hội thoại (conversation_id)": "conversation_id",
        "Tạo lúc": "created_at",
        "Số điện thoại": "phone",
        "Đã xoá": "is_deleted",
        "Bị ẩn": "is_hidden",
        "Nhân viên được phân công": "assigned_staff",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # Bỏ row thiếu message / conversation_id
    df = df.dropna(subset=["message", "conversation_id"])
    df["message"] = df["message"].astype(str).str.strip()
    df = df[df["message"].str.len() > 0]

    # Loại bỏ tin nhắn hệ thống
    pattern = "|".join(re.escape(p) for p in SYSTEM_MESSAGE_PATTERNS)
    df = df[~df["message"].str.contains(pattern, case=False, na=False)]

    # Loại bỏ tin nhắn bị xoá / ẩn
    for col in ["is_deleted", "is_hidden"]:
        if col in df.columns:
            df = df[df[col].astype(str).str.upper() != "TRUE"]

    # Xác định role: page name = assistant, còn lại = user
    df["role"] = df["sender_name"].apply(
        lambda x: "assistant" if str(x).strip() in PAGE_NAMES else "user"
    )

    # Mask số điện thoại trong nội dung
    df["message_clean"] = df["message"].apply(
        lambda x: re.sub(r"0\d{9}", "[PHONE]", str(x))
    )

    # Clean phone column
    if "phone" in df.columns:
        df["phone"] = df["phone"].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
        df["phone"] = df["phone"].replace(["nan", "None", ""], pd.NA)

    # Forward-fill phone trong cùng conversation
    df = df.sort_values(["conversation_id", "created_at"])
    df["phone"] = df.groupby("conversation_id")["phone"].transform(
        lambda s: s.ffill().bfill()
    )

    return df
