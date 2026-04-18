def compute_score(row):
    """Tính tổng score chất lượng conversation cho training.

    Score cao → conversation tốt, nên đưa vào dataset.
    """
    score = row.get("reward", 0)

    # Bonus cho conversation ngắn gọn hiệu quả (6-15 turns)
    num_turns = row.get("num_turns", 0)
    if 6 <= num_turns <= 15:
        score += 20
    elif num_turns <= 10:
        score += 10

    # Bonus cho conversation có CTA
    if row.get("has_closing_cta"):
        score += 15

    # Bonus cho khách đã chi tiền (proven sales conversation)
    if row.get("paid_value", 0) > 0:
        score += 20

    # Bonus avg message length hợp lý (20-200 chars)
    avg_len = row.get("avg_msg_length", 0)
    if 20 <= avg_len <= 200:
        score += 10

    return score
