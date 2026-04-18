import re


def extract_features(df):
    """Trích xuất features phục vụ scoring từ conversation data."""

    # Số lượt trao đổi trong conversation
    df["num_turns"] = df.groupby("conversation_id")["message"].transform("count")

    # Số lượt user nói (phản ánh mức độ engage)
    df["user_turns"] = df.groupby("conversation_id")["role"].transform(
        lambda x: (x == "user").sum()
    )

    # Số lượt assistant nói
    df["assistant_turns"] = df.groupby("conversation_id")["role"].transform(
        lambda x: (x == "assistant").sum()
    )

    # Tỷ lệ user/assistant (balanced = tốt)
    df["turn_ratio"] = df["user_turns"] / df["assistant_turns"].replace(0, 1)

    # Có SĐT trong conversation không
    df["has_phone"] = df.groupby("conversation_id")["phone"].transform(
        lambda x: x.notna().any()
    )

    # Conversation có chứa keyword chốt sale không
    closing_keywords = ["đặt lịch", "lịch hẹn", "để lại sđt", "số điện thoại",
                        "zalo", "liên hệ", "đăng ký", "book", "hẹn"]
    pattern = "|".join(re.escape(k) for k in closing_keywords)
    df["has_closing_cta"] = df.groupby("conversation_id")["message"].transform(
        lambda x: x.str.contains(pattern, case=False, na=False).any()
    )

    # Độ dài trung bình message (quá ngắn = spam, quá dài = có thể copy-paste)
    df["avg_msg_length"] = df.groupby("conversation_id")["message"].transform(
        lambda x: x.str.len().mean()
    )

    return df
