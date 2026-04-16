
from fastapi import FastAPI
from retrieval import search
from prompt import build

app = FastAPI()

@app.get("/chat")
def chat(q:str, segment:str="MID"):
    ctx = search(q)
    prompt = build(segment, ctx)
    return {"context":ctx,"prompt":prompt}
