"""Tone & Style Extraction — v3.1

Phân tích phong cách sales của nhân viên giỏi nhất để bot học theo.
Input: conversations của top sales (identified via chatpage data)
Output: style profile dict
"""

import re
import pandas as pd


def extract_style_profile(df: pd.DataFrame, top_sales_names: list[str]) -> dict:
    """Trích xuất style profile từ conversations của top sales.

    Args:
        df: Joined DataFrame (conversations + CRM)
        top_sales_names: List tên nhân viên top sales từ chatpage

    Returns:
        Style profile dict cho prompt builder
    """
    if not top_sales_names:
        return _default_profile()

    # Filter messages từ top sales (assistant messages mà assigned_staff là top sales)
    mask = pd.Series(False, index=df.index)

    # Match by assigned_staff
    if "assigned_staff" in df.columns:
        mask |= df["assigned_staff"].isin(top_sales_names)

    # Match by sender_name (nếu sender_name là page nhưng assigned_staff là top sales)
    top_conv_ids = df[mask]["conversation_id"].unique() if mask.any() else []

    # Lấy assistant messages từ các conversations của top sales
    if len(top_conv_ids) > 0:
        top_msgs = df[
            (df["conversation_id"].isin(top_conv_ids)) &
            (df["role"] == "assistant")
        ]["message"].dropna()
    else:
        # Fallback: dùng tất cả assistant messages
        top_msgs = df[df["role"] == "assistant"]["message"].dropna()

    if top_msgs.empty:
        return _default_profile()

    profile = {
        "avg_msg_length": round(top_msgs.str.len().mean(), 1),
        "median_msg_length": round(top_msgs.str.len().median(), 1),
        "emoji_usage_rate": _emoji_rate(top_msgs),
        "common_emojis": _top_emojis(top_msgs),
        "greeting_patterns": _extract_patterns(top_msgs, _GREETING_PATTERNS),
        "closing_patterns": _extract_patterns(top_msgs, _CLOSING_PATTERNS),
        "cta_phrases": _extract_cta_phrases(top_msgs),
        "pronoun_style": _detect_pronoun_style(top_msgs),
        "objection_handling_samples": _extract_objection_handling(df, top_conv_ids),
        "top_sales_names": top_sales_names,
        "sample_count": len(top_msgs),
    }

    return profile


# === Pattern constants ===
_GREETING_PATTERNS = [
    r"chào\s+(chị|anh|bạn)",
    r"xin\s+chào",
    r"em\s+chào\s+(chị|anh)",
    r"hello",
    r"hi\s+(chị|anh)",
]

_CLOSING_PATTERNS = [
    r"chúc\s+(chị|anh|bạn)",
    r"cảm\s+ơn",
    r"hẹn\s+gặp",
    r"chào\s+nhé",
]

_CTA_PATTERNS = [
    r"để\s+lại\s+s[đd]t",
    r"cho\s+em\s+xin\s+s[đd]t",
    r"đặt\s+lịch",
    r"liên\s+hệ",
    r"inbox",
    r"nhắn\s+tin",
    r"gọi\s+cho\s+em",
    r"zalo",
    r"em\s+giữ\s+chỗ",
    r"em\s+đặt\s+cho",
    r"slot",
]


def _emoji_rate(msgs: pd.Series) -> float:
    """Tỷ lệ messages có emoji."""
    emoji_pattern = re.compile(
        "[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF"
        "\U00002702-\U000027B0\U0001f900-\U0001f9FF"
        "\U0000FE00-\U0000FE0F\U00002600-\U000026FF"
        "\U0000200D\U00002640\U00002642]+",
        flags=re.UNICODE,
    )
    has_emoji = msgs.str.contains(emoji_pattern, na=False)
    return round(has_emoji.mean(), 3)


def _top_emojis(msgs: pd.Series, top_n: int = 5) -> list[str]:
    """Top N emojis được dùng nhiều nhất."""
    emoji_pattern = re.compile(
        "([\U0001F600-\U0001F64F\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF"
        "\U00002702-\U000027B0\U0001f900-\U0001f9FF"
        "\U00002600-\U000026FF])",
        flags=re.UNICODE,
    )
    all_emojis = msgs.str.findall(emoji_pattern).explode().dropna()
    if all_emojis.empty:
        return []
    return all_emojis.value_counts().head(top_n).index.tolist()


def _extract_patterns(msgs: pd.Series, patterns: list[str]) -> list[str]:
    """Extract matched patterns from messages."""
    found = set()
    for pattern in patterns:
        extracted = msgs.str.extract(f"({pattern})", flags=re.IGNORECASE, expand=True)
        # expand=True luôn trả về DataFrame; lấy cột đầu tiên (full match)
        matches = extracted.iloc[:, 0].dropna()
        found.update(matches.str.lower().unique())
    return list(found)[:10]


def _extract_cta_phrases(msgs: pd.Series) -> list[str]:
    """Extract CTA phrases thực tế từ top sales."""
    found = []
    for pattern in _CTA_PATTERNS:
        matches = msgs[msgs.str.contains(pattern, case=False, na=False)]
        for msg in matches.head(3):
            # Lấy câu chứa CTA
            sentences = re.split(r"[.!?\n]", str(msg))
            for s in sentences:
                if re.search(pattern, s, re.IGNORECASE) and len(s.strip()) > 10:
                    found.append(s.strip())
                    break
    # Deduplicate + limit
    return list(dict.fromkeys(found))[:15]


def _detect_pronoun_style(msgs: pd.Series) -> dict:
    """Phát hiện cách xưng hô: em/chị, em/anh, bạn, etc."""
    text = " ".join(msgs.head(500).tolist())
    return {
        "em": len(re.findall(r"\bem\b", text, re.IGNORECASE)),
        "chị": len(re.findall(r"\bchị\b", text, re.IGNORECASE)),
        "anh": len(re.findall(r"\banh\b", text, re.IGNORECASE)),
        "bạn": len(re.findall(r"\bbạn\b", text, re.IGNORECASE)),
    }


def _extract_objection_handling(df: pd.DataFrame, top_conv_ids) -> list[str]:
    """Trích mẫu xử lý phản đối từ top sales."""
    if len(top_conv_ids) == 0:
        return []

    objection_kw = ["đắt", "sợ", "đau", "lo", "ngại", "không biết"]
    pattern = "|".join(objection_kw)

    samples = []
    for cid in top_conv_ids[:50]:
        conv = df[df["conversation_id"] == cid].sort_values("created_at")
        msgs = conv[["role", "message"]].values.tolist()
        for i, (role, msg) in enumerate(msgs):
            if role == "user" and re.search(pattern, str(msg), re.IGNORECASE):
                # Lấy response kế tiếp từ assistant
                if i + 1 < len(msgs) and msgs[i + 1][0] == "assistant":
                    samples.append(f"Khách: {msg}\nSale: {msgs[i+1][1]}")
                    if len(samples) >= 10:
                        return samples
    return samples


def _default_profile() -> dict:
    """Default style profile khi không có data."""
    return {
        "avg_msg_length": 80,
        "median_msg_length": 60,
        "emoji_usage_rate": 0.3,
        "common_emojis": ["😊", "🥰", "💕"],
        "greeting_patterns": ["chào chị", "em chào chị"],
        "closing_patterns": ["chúc chị ngủ ngon"],
        "cta_phrases": ["Chị để lại SĐT để em tư vấn nhé"],
        "pronoun_style": {"em": 100, "chị": 80, "anh": 20, "bạn": 10},
        "objection_handling_samples": [],
        "top_sales_names": [],
        "sample_count": 0,
    }
