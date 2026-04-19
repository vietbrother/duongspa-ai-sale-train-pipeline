"""Ground Truth Pipeline — v3.1

Match conversation → CRM outcome (booking → show_up → paid)
Label: won / lost / pending
Compute actual reward (không chỉ predicted)
"""

import pandas as pd
from config import OUTCOME_WEIGHTS


def label_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """Gán label outcome cho mỗi conversation dựa trên CRM data.

    Labels:
        won:     paid > 0 (đã thanh toán)
        booked:  booking_count > 0 nhưng chưa paid
        pending: có phone nhưng chưa booking
        lost:    không có tín hiệu conversion nào
    """
    def _label(row):
        paid = row.get("paid_value", 0)
        booked = row.get("booking_count", 0)
        has_phone = row.get("has_phone", False)

        if paid > 0:
            return "won"
        if booked > 0:
            return "booked"
        if has_phone:
            return "pending"
        return "lost"

    df["outcome_label"] = df.apply(_label, axis=1)
    return df


def compute_outcome_weight(row) -> float:
    """Tính outcome weight multiplier theo v3.1 architecture.

    Conversations dẫn đến paid cao → weight cao hơn trong ranking.
    """
    paid = row.get("paid_value", 0)
    booked = row.get("booking_count", 0)
    has_phone = row.get("has_phone", False)

    if paid > 5_000_000:
        return OUTCOME_WEIGHTS["paid_high"]
    if paid > 0:
        return OUTCOME_WEIGHTS["show_up"]
    if booked > 0:
        return OUTCOME_WEIGHTS["booked"]
    if has_phone:
        return OUTCOME_WEIGHTS["phone_only"]
    return OUTCOME_WEIGHTS["no_conversion"]


def apply_outcome_weights(df: pd.DataFrame) -> pd.DataFrame:
    """Apply outcome weighting vào score cho toàn bộ DataFrame."""
    df["outcome_weight"] = df.apply(compute_outcome_weight, axis=1)
    df["weighted_score"] = df["score"] * df["outcome_weight"]
    return df


def get_gold_dataset(df: pd.DataFrame, min_paid: float = 0) -> pd.DataFrame:
    """Filter gold dataset: won conversations + high revenue.

    Gold dataset dùng cho fine-tune với chất lượng cao nhất.
    """
    gold = df[
        (df["outcome_label"] == "won") &
        (df["paid_value"] > min_paid)
    ].copy()
    return gold.sort_values("weighted_score", ascending=False)


def compute_ground_truth_stats(df: pd.DataFrame) -> dict:
    """Thống kê Ground Truth coverage."""
    total = df["conversation_id"].nunique()
    labels = df.groupby("conversation_id")["outcome_label"].first().value_counts().to_dict()

    return {
        "total_conversations": total,
        "outcome_distribution": labels,
        "ground_truth_coverage": round(
            (labels.get("won", 0) + labels.get("booked", 0)) / max(total, 1) * 100, 1
        ),
        "gold_count": labels.get("won", 0),
        "avg_outcome_weight": round(df["outcome_weight"].mean(), 3) if "outcome_weight" in df.columns else 0,
    }
