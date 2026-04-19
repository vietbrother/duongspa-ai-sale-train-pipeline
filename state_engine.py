"""Conversation State Detection (FSM) — v3.1

Phát hiện trạng thái hội thoại để:
- Gắn state metadata vào segments (training)
- Ưu tiên ví dụ cùng state khi retrieval (runtime)
- Điều chỉnh strategy theo state (prompt builder)

States: greeting → qualifying → presenting → handling_objection → closing → follow_up
"""

import re
from config import STATE_KEYWORDS, CONVERSATION_STATES


# State transition probabilities (FSM logic)
# Từ state nào có thể chuyển sang state nào
VALID_TRANSITIONS = {
    "greeting": ["qualifying", "presenting", "greeting"],
    "qualifying": ["presenting", "qualifying", "closing"],
    "presenting": ["handling_objection", "closing", "presenting", "qualifying"],
    "handling_objection": ["presenting", "closing", "handling_objection"],
    "closing": ["follow_up", "handling_objection", "closing"],
    "follow_up": ["greeting", "qualifying", "follow_up"],
}

# State strategy mapping (cho prompt builder)
STATE_STRATEGIES = {
    "greeting": {
        "strategy": "Thân thiện, hỏi nhu cầu",
        "cta_example": "Chị quan tâm dịch vụ nào ạ?",
    },
    "qualifying": {
        "strategy": "Thu thập info, xác định nhu cầu",
        "cta_example": "Chị cho em xin SĐT để tư vấn chi tiết nhé?",
    },
    "presenting": {
        "strategy": "Đưa lợi ích + social proof",
        "cta_example": "Chị muốn em đặt lịch trải nghiệm không ạ?",
    },
    "handling_objection": {
        "strategy": "Empathy + bằng chứng",
        "cta_example": "Em hiểu, để em gửi chị feedback khách đã làm nhé",
    },
    "closing": {
        "strategy": "Urgency + scarcity",
        "cta_example": "Slot cuối tuần còn 2 chỗ, em giữ cho chị nhé?",
    },
    "follow_up": {
        "strategy": "Nhẹ nhàng nhắc lại",
        "cta_example": "Chị ơi, chị còn quan tâm dịch vụ hôm trước không ạ?",
    },
}


def detect_state_single(message: str, role: str = "user") -> str:
    """Phát hiện state từ 1 message bằng keyword matching (rule-based, <10ms)."""
    msg_lower = message.lower()

    scores = {}
    for state, keywords in STATE_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in msg_lower)
        if score > 0:
            scores[state] = score

    if not scores:
        # Default: assistant nói mà không match → presenting
        return "presenting" if role == "assistant" else "qualifying"

    return max(scores, key=scores.get)


def detect_conversation_states(messages: list[dict]) -> list[str]:
    """Phát hiện state sequence cho toàn bộ conversation.

    Sử dụng hybrid: keyword matching + FSM transition logic.
    Mỗi message được gán 1 state, với FSM constraint để tránh nhảy state vô lý.

    Args:
        messages: list of {"role": "user"|"assistant", "content": "..."}

    Returns:
        List states tương ứng từng message
    """
    if not messages:
        return []

    states = []
    prev_state = "greeting"  # Hội thoại luôn bắt đầu từ greeting

    for i, msg in enumerate(messages):
        content = msg.get("content", "")
        role = msg.get("role", "user")

        # First message → greeting
        if i == 0:
            states.append("greeting")
            prev_state = "greeting"
            continue

        # Detect state từ content
        detected = detect_state_single(content, role)

        # FSM constraint: chỉ cho phép transition hợp lệ
        valid_next = VALID_TRANSITIONS.get(prev_state, CONVERSATION_STATES)
        if detected in valid_next:
            state = detected
        else:
            # Nếu detected không hợp lệ, giữ nguyên prev hoặc chọn next hợp lệ gần nhất
            state = prev_state

        states.append(state)
        prev_state = state

    return states


def get_dominant_state(states: list[str]) -> str:
    """Lấy state xuất hiện nhiều nhất (dominant) trong 1 segment."""
    if not states:
        return "greeting"
    from collections import Counter
    return Counter(states).most_common(1)[0][0]


def get_final_state(states: list[str]) -> str:
    """Lấy state cuối cùng — phản ánh conversation kết thúc ở đâu."""
    return states[-1] if states else "greeting"


def get_state_strategy(state: str) -> dict:
    """Lấy strategy theo state cho prompt builder."""
    return STATE_STRATEGIES.get(state, STATE_STRATEGIES["qualifying"])
