
def strategy(seg):
    return {
        "LOW":"hard close",
        "MID":"educate",
        "HIGH":"trust",
        "VIP":"consult"
    }.get(seg,"soft")

def build(segment, context, model="openai"):
    base=f"Segment:{segment}\nStrategy:{strategy(segment)}\nContext:{context}"
    if model=="openai":
        return {"messages":[{"role":"system","content":base}]}
    if model=="gemini":
        return {"contents":[{"parts":[{"text":base}]}]}
    if model=="claude":
        return base
    return {"prompt":base}
