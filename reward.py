import numpy as np


def compute_reward(row):
    """Tính reward score cho mỗi conversation.

    Reward cao = conversation có giá trị cao cho training:
    - Khách để lại SĐT (lead capture)
    - Khách đã booking (conversion)
    - Khách đã chi tiền (revenue)
    - Conversation có CTA (closing quality)
    - Turn ratio hợp lý (engagement quality)
    """
    reward = 0

    # Lead capture: có SĐT
    if row.get("has_phone"):
        reward += 20

    # Conversion: có booking
    if row.get("booking_count", 0) > 0:
        reward += 50

    # Revenue: đã chi tiền
    paid = row.get("paid_value", 0)
    if paid > 0:
        reward += 100
        reward += np.log1p(paid) * 5  # Scale reward theo revenue

    # Closing quality: có CTA
    if row.get("has_closing_cta"):
        reward += 30

    # Engagement quality: tỷ lệ user/assistant hợp lý (0.5-2.0)
    ratio = row.get("turn_ratio", 0)
    if 0.5 <= ratio <= 2.0:
        reward += 15

    # Penalize quá ngắn hoặc quá dài
    num_turns = row.get("num_turns", 0)
    if num_turns < 3:
        reward -= 20  # Quá ngắn, không đủ context
    elif num_turns > 50:
        reward -= 10  # Quá dài, có thể spam

    return max(reward, 0)
