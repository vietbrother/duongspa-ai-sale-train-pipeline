def predict_revenue(row):
    prob = 0.2
    if row["num_turns"] > 5:
        prob += 0.2
    if row.get("paid_value", 0) > 0:
        prob += 0.3
    expected = prob * max(row.get("paid_value", 0), 1_000_000)
    return prob, expected
