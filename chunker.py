from config import MAX_TURNS_PER_CHUNK, MIN_TURNS_PER_CHUNK
from state_engine import detect_conversation_states, get_dominant_state, get_final_state


def extract_segments(df):
    """Tách conversations thành segments chất lượng cho RAG + fine-tune.

    Mỗi segment chứa:
    - text: hội thoại dạng role-based (user: ... / assistant: ...)
    - metadata: phone, paid_value, reward, score, segment, tags, service_interest
    - qa_pairs: danh sách cặp Q/A cho Qdrant embedding

    Tách tại điểm:
    - Đủ MAX_TURNS_PER_CHUNK
    - Chuyển topic (gap thời gian lớn)
    """
    segments = []
    grouped = df.groupby("conversation_id")

    for cid, group in grouped:
        group = group.sort_values("created_at")

        # Bỏ conversation quá ngắn
        if len(group) < MIN_TURNS_PER_CHUNK:
            continue

        # Build role-based text
        messages = []
        qa_pairs = []
        current_user_msg = None

        for _, row in group.iterrows():
            role = row.get("role", "user")
            msg = str(row.get("message_clean", row.get("message", ""))).strip()
            if not msg:
                continue

            messages.append({"role": role, "content": msg})

            # Extract Q/A pairs (user question → assistant answer)
            if role == "user":
                current_user_msg = msg
            elif role == "assistant" and current_user_msg:
                qa_pairs.append({
                    "question": current_user_msg,
                    "answer": msg,
                })
                current_user_msg = None

        if len(messages) < MIN_TURNS_PER_CHUNK:
            continue

        # Tách thành chunks nếu quá dài
        chunks = _split_messages(messages, MAX_TURNS_PER_CHUNK)

        for chunk_idx, chunk_msgs in enumerate(chunks):
            # Build text dạng readable
            text_lines = []
            for m in chunk_msgs:
                prefix = "Khách" if m["role"] == "user" else "Sale"
                text_lines.append(f"{prefix}: {m['content']}")
            text = "\n".join(text_lines)

            # Metadata
            first_row = group.iloc[0]

            # Detect states cho chunk messages
            chunk_states = detect_conversation_states(chunk_msgs)

            segment_data = {
                "conversation_id": cid,
                "chunk_index": chunk_idx,
                "text": text,
                "messages": chunk_msgs,  # Structured messages cho fine-tune
                "phone": str(first_row.get("phone", "")),
                "paid_value": float(first_row.get("paid_value", 0)),
                "reward": float(first_row.get("reward", 0)),
                "score": float(first_row.get("score", 0)),
                "weighted_score": float(first_row.get("weighted_score", first_row.get("score", 0))),
                "segment": str(first_row.get("segment", "LOW")),
                "tags": str(first_row.get("tags", "")),
                "service_interest": str(first_row.get("service_interest", "")),
                "num_turns": len(chunk_msgs),
                "has_closing_cta": bool(first_row.get("has_closing_cta", False)),
                "outcome_label": str(first_row.get("outcome_label", "lost")),
                "outcome_weight": float(first_row.get("outcome_weight", 1.0)),
                # State metadata (v3.1)
                "dominant_state": get_dominant_state(chunk_states),
                "final_state": get_final_state(chunk_states),
                "state_sequence": chunk_states,
            }

            # Attach Q/A pairs thuộc chunk này
            chunk_qa = qa_pairs[
                chunk_idx * (MAX_TURNS_PER_CHUNK // 2):
                (chunk_idx + 1) * (MAX_TURNS_PER_CHUNK // 2)
            ]
            if chunk_qa:
                segment_data["qa_pairs"] = chunk_qa

            segments.append(segment_data)

    return segments


def _split_messages(messages, max_turns):
    """Tách danh sách messages thành các chunks <= max_turns."""
    if len(messages) <= max_turns:
        return [messages]

    chunks = []
    for i in range(0, len(messages), max_turns):
        chunk = messages[i:i + max_turns]
        if len(chunk) >= MIN_TURNS_PER_CHUNK:
            chunks.append(chunk)
    return chunks if chunks else [messages]
