import pandas as pd
import re
from config import CRM_FILE, CHAT_FILE, CHATPAGE_FILE, PAGE_NAMES, SYSTEM_MESSAGE_PATTERNS


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
        "Tên người gửi (customer_name)": "sender_name",
        "Mã người gửi": "sender_id",
        "Số lượt thích": "likes_count",
        "Đã xoá": "is_deleted",
        "Bị ẩn": "is_hidden",
        "Có số điện thoại": "has_phone",
        "Có đính kèm": "has_attachment",
        "Tên nhà mạng": "network_name",
        "Liên kết với đơn hàng": "order_link",
        "Mã trang (page_id)": "page_id",
        "Mã hội thoại (thread_key)": "thread_key",
        "Mã hội thoại (conversation_id)": "conversation_id",
        "Tạo lúc": "created_at",
        "Link Pancake": "link_pancake",  
        "Ghi chú": "notes",
        "Bị ẩn": "is_hidden",
        "Nhân viên được phân công": "assigned_staff",
        "Số điện thoại": "phone",
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


def load_chatpage(file_path: str = None) -> pd.DataFrame:
    """Load chatpage staff data (name, total_phone) từ CRM export.

    Dùng để xác định top sales staff cho Tone & Style Extraction.
    """
    file_path = file_path or CHATPAGE_FILE
    df = pd.read_csv(file_path) if file_path.endswith(".csv") else pd.read_excel(file_path)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    if "name" in df.columns:
        df["name"] = df["name"].astype(str).str.strip()
    if "total_phone" in df.columns:
        df["total_phone"] = pd.to_numeric(df["total_phone"], errors="coerce").fillna(0).astype(int)

    return df


def identify_top_sales(chatpage_df: pd.DataFrame, percentile: float = 0.3) -> list[str]:
    """Xác định top sales staff dựa trên total_phone (số khách handle).

    Returns:
        List tên nhân viên top sales (top percentile%)
    """
    if chatpage_df.empty or "total_phone" not in chatpage_df.columns:
        return []

    threshold = chatpage_df["total_phone"].quantile(1 - percentile)
    top = chatpage_df[chatpage_df["total_phone"] >= threshold]
    return top["name"].tolist()
