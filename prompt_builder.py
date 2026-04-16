def get_strategy(segment):
    return {
        "LOW": "hard close + urgency",
        "MID": "educate + soft close",
        "HIGH": "trust + social proof",
        "VIP": "consultant style"
    }.get(segment, "soft close")

def build_prompt(segment, model="openai"):
    base = f"Khách thuộc nhóm: {segment}\nChiến lược: {get_strategy(segment)}"
    return base
