def join_data(conv_df, crm_df):
    return conv_df.merge(crm_df, on="phone", how="left")
