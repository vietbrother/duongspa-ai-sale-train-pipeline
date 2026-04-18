import pandas as pd


def join_data(conv_df: pd.DataFrame, crm_df: pd.DataFrame) -> pd.DataFrame:
    """Join conversation với CRM data qua phone.

    Enriches mỗi message với thông tin CRM: booking_count, total_paid_value,
    status, service_interest.
    """
    # Đảm bảo phone cùng kiểu
    conv_df["phone"] = conv_df["phone"].astype(str).str.strip()
    crm_df["phone"] = crm_df["phone"].astype(str).str.strip()

    # Chọn cột CRM cần thiết
    crm_cols = ["phone", "booking_count", "total_paid_value", "total_package_value",
                "status", "service_interest"]
    crm_cols = [c for c in crm_cols if c in crm_df.columns]

    df = conv_df.merge(crm_df[crm_cols], on="phone", how="left")

    # Rename cho thống nhất với pipeline cũ
    if "total_paid_value" in df.columns:
        df = df.rename(columns={"total_paid_value": "paid_value"})
    else:
        df["paid_value"] = 0

    df["paid_value"] = pd.to_numeric(df["paid_value"], errors="coerce").fillna(0)
    df["booking_count"] = pd.to_numeric(df.get("booking_count", 0), errors="coerce").fillna(0)

    return df
