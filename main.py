from loader import load_crm, load_conversations
from joiner import join_data
from feature import extract_features
from reward import compute_reward
from segment import get_segment
from scoring import compute_score
from prediction import predict_revenue
from chunker import extract_segments
from embedding import embed
from qdrant_client import create_collection, push

def run():
    crm = load_crm("crm.xlsx")
    conv = load_conversations("chat.xlsx")

    df = join_data(conv, crm)
    df = extract_features(df)

    df["reward"] = df.apply(compute_reward, axis=1)
    df["segment"] = df["paid_value"].apply(get_segment)
    df["score"] = df.apply(compute_score, axis=1)

    df["pred_prob"], df["expected_rev"] = zip(*df.apply(predict_revenue, axis=1))

    df = df[df["score"] > 70]

    segments = extract_segments(df)

    points = []
    for i, seg in enumerate(segments):
        vector = embed(seg["text"])
        points.append({
            "id": i,
            "vector": vector,
            "payload": seg
        })

    create_collection()
    push(points)

if __name__ == "__main__":
    run()
