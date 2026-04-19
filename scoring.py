def compute_score(row):
    """Tính tổng score chất lượng conversation cho training.

    Score cao → conversation tốt, nên đưa vào dataset.
    v3.1: tích hợp outcome_weight từ Ground Truth.
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
    paid = row.get("paid_value", 0)
    if paid > 0:
        score += 20
    if paid > 5_000_000:
        score += 15  # Extra bonus cho high-value

    # Bonus avg message length hợp lý (20-200 chars)
    avg_len = row.get("avg_msg_length", 0)
    if 20 <= avg_len <= 200:
        score += 10

    # Bonus cho conversation assigned to top sales
    if row.get("is_top_sales_conv", False):
        score += 25

    return score
