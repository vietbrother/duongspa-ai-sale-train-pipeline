def compute_score(row):
    score = row["reward"]
    if row["num_turns"] <= 10:
        score += 10
    return score
