def get_strategy(segment: str) -> dict:
    """Trả về chiến lược sales theo segment khách hàng."""
    strategies = {
        "LOW": {
            "style": "hard close + urgency",
            "desc": "Khách chưa chi tiền. Tạo urgency, đưa ưu đãi giới hạn, ép lịch trải nghiệm.",
        },
        "MID": {
            "style": "educate + soft close",
            "desc": "Khách đã trải nghiệm. Educate về lợi ích lâu dài, upsell package.",
        },
        "HIGH": {
            "style": "trust + social proof",
            "desc": "Khách chi nhiều. Xây trust, đưa case study, recommend liệu trình cao cấp.",
        },
        "VIP": {
            "style": "consultant style",
            "desc": "Khách VIP. Tư vấn cá nhân, ưu tiên đặc biệt, giữ chân dài hạn.",
        },
    }
    return strategies.get(segment, strategies["LOW"])


def build_prompt(segment: str, context: list[dict], query: str = "",
                 model: str = "openai", state: str = None,
                 collected_info: dict = None, style_profile: dict = None) -> dict:
    """Build prompt theo segment + state + context + tone (v3.1).

    Args:
        segment: Phân khúc khách hàng
        context: List kết quả search từ Qdrant
        query: Câu hỏi user
        model: Provider LLM (openai/gemini/claude/grok)
        state: Conversation state hiện tại (v3.1)
        collected_info: Info đã thu thập — KHÔNG hỏi lại (v3.1)
        style_profile: Tone & style từ top sales (v3.1)
    """
    strategy = get_strategy(segment)

    # State strategy (v3.1)
    from state_engine import get_state_strategy
    state_info = get_state_strategy(state) if state else {"strategy": "", "cta_example": ""}

    # Collected info block
    collected_block = ""
    if collected_info:
        items = [f"- {k}: {v}" for k, v in collected_info.items() if v]
        if items:
            collected_block = "\n\nTHÔNG TIN ĐÃ BIẾT (KHÔNG HỎI LẠI):\n" + "\n".join(items)

    # Tone block from style profile
    tone_block = ""
    if style_profile and style_profile.get("sample_count", 0) > 0:
        tone_block = f"""

PHONG CÁCH THAM KHẢO (từ nhân viên giỏi nhất):
- Độ dài TB: {style_profile.get('avg_msg_length', 80):.0f} ký tự
- Emoji: {'có dùng' if style_profile.get('emoji_usage_rate', 0) > 0.2 else 'ít dùng'} ({', '.join(style_profile.get('common_emojis', [])[:3])})
- CTA mẫu: {'; '.join(style_profile.get('cta_phrases', [])[:3])}"""

    system_prompt = f"""Bạn là nhân viên tư vấn sales tại DuongSpa.

THÔNG TIN KHÁCH:
- Phân khúc: {segment}
- Chiến lược: {strategy['style']}
- Hướng dẫn: {strategy['desc']}
- Trạng thái hội thoại: {state or 'chưa xác định'}
- Chiến lược theo state: {state_info['strategy']}
{collected_block}{tone_block}

QUY TẮC BẮT BUỘC:
1. Luôn hỏi thêm thông tin khách (tên, SĐT, nhu cầu cụ thể)
2. Luôn có CTA cuối mỗi câu trả lời (VD: {state_info['cta_example']})
3. KHÔNG BAO GIỜ kết thúc hội thoại
4. Trả lời ngắn gọn (2-4 câu), tự nhiên, giống người thật
5. Dẫn dắt khách tới hành động cụ thể
6. KHÔNG hỏi lại thông tin đã biết
7. Gọi khách bằng tên nếu đã biết

PHONG CÁCH:
- Thân thiện, nhiệt tình nhưng không quá đà
- Dùng emoji vừa phải 😊
- Xưng em/chị hoặc em/anh"""

    # Build context từ search results
    context_text = ""
    if context:
        context_parts = []
        for i, ctx in enumerate(context):
            text = ctx if isinstance(ctx, str) else ctx.get("text", "")
            context_parts.append(f"[Tham khảo {i+1}]\n{text}")
        context_text = "\n\n".join(context_parts)

    user_content = ""
    if context_text:
        user_content += f"Hội thoại mẫu tham khảo:\n{context_text}\n\n"
    if query:
        user_content += f"Khách hỏi: {query}\n\nHãy trả lời theo chiến lược {strategy['style']}."

    if model in ("openai", "grok"):
        return {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]
        }
    elif model == "gemini":
        return {
            "contents": [
                {"parts": [{"text": system_prompt + "\n\n" + user_content}]}
            ]
        }
    elif model == "claude":
        return {"prompt": f"{system_prompt}\n\nHuman: {user_content}\n\nAssistant:"}
    else:
        return {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]
        }
