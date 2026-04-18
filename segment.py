from config import SEGMENT_RULES


def get_segment(paid_value):
    """Phân loại khách hàng theo tổng giá trị thanh toán."""
    paid_value = float(paid_value) if paid_value else 0
    for name, (low, high) in SEGMENT_RULES.items():
        if low <= paid_value < high:
            return name
    return "LOW"
