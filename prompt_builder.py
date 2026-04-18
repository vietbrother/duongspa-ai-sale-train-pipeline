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


def build_prompt(segment: str, context: list[dict], query: str = "", model: str = "openai") -> dict:
    """Build prompt theo segment + context + model provider.

    Args:
        segment: Phân khúc khách hàng
        context: List kết quả search từ Qdrant
        query: Câu hỏi user
        model: Provider LLM (openai/gemini/claude/grok)
    """
    strategy = get_strategy(segment)

    system_prompt = f"""Bạn là nhân viên sales spa chuyên chốt khách tại DuongSpa.

THÔNG TIN KHÁCH:
- Phân khúc: {segment}
- Chiến lược: {strategy['style']}
- Hướng dẫn: {strategy['desc']}

QUY TẮC BẮT BUỘC:
1. Luôn hỏi thêm thông tin khách (tên, SĐT, nhu cầu cụ thể)
2. Luôn có CTA cuối mỗi câu trả lời (đặt lịch / xin SĐT / mời qua Zalo)
3. KHÔNG BAO GIỜ kết thúc hội thoại
4. Trả lời ngắn gọn, tự nhiên, giống người thật
5. Dẫn dắt khách tới hành động cụ thể

PHONG CÁCH:
- Thân thiện, nhiệt tình nhưng không quá đà
- Dùng emoji vừa phải
- Gọi khách là "chị/anh" """

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
