def extract_features(df):
    df["num_turns"] = df.groupby("conversation_id")["message"].transform("count")
    return df
