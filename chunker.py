def extract_segments(df):
    segments = []
    grouped = df.groupby("conversation_id")
    for cid, group in grouped:
        text = "\n".join(group["message"].tolist())
        segments.append({
            "conversation_id": cid,
            "text": text,
            "phone": group["phone"].iloc[0],
            "paid_value": group["paid_value"].iloc[0],
            "reward": group["reward"].iloc[0],
            "score": group["score"].iloc[0],
            "segment": group["segment"].iloc[0],
        })
    return segments
