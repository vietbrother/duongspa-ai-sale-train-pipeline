from config import SEGMENT_RULES

def get_segment(paid_value):
    for name, (low, high) in SEGMENT_RULES.items():
        if low <= paid_value < high:
            return name
    return "LOW"
