import numpy as np

def compute_reward(row):
    reward = 0
    if row.get("phone"):
        reward += 20
    if row.get("booking_count", 0) > 0:
        reward += 50
    if row.get("paid_value", 0) > 0:
        reward += 100
        reward += np.log1p(row["paid_value"])
    return reward
