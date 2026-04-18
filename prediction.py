def predict_revenue(row):
    """Dự đoán xác suất chuyển đổi và doanh thu kỳ vọng.

    Dựa trên tín hiệu hành vi trong conversation:
    - Số lượt trao đổi (engagement)
    - Đã chi tiền chưa (conversion history)
    - Có CTA / SĐT không (intent signals)
    - Segment khách hàng
    """
    prob = 0.1  # Base probability

    # Engagement signals
    num_turns = row.get("num_turns", 0)
    if num_turns > 10:
        prob += 0.25
    elif num_turns > 5:
        prob += 0.15

    # Conversion history
    paid = row.get("paid_value", 0)
    if paid > 0:
        prob += 0.3

    # Booking history
    if row.get("booking_count", 0) > 0:
        prob += 0.15

    # Intent signals
    if row.get("has_closing_cta"):
        prob += 0.1
    if row.get("has_phone"):
        prob += 0.1

    # Engagement quality
    ratio = row.get("turn_ratio", 0)
    if 0.5 <= ratio <= 2.0:
        prob += 0.05

    prob = min(prob, 0.95)

    # Expected revenue: xác suất × giá trị trung bình theo segment
    segment_avg = {
        "LOW": 500_000,
        "MID": 2_000_000,
        "HIGH": 6_000_000,
        "VIP": 15_000_000,
    }
    avg_value = segment_avg.get(row.get("segment", "LOW"), 1_000_000)
    expected = prob * max(paid, avg_value)

    return prob, expected
