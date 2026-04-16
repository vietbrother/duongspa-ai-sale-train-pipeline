import pandas as pd

def load_crm(file_path):
    return pd.read_excel(file_path)

def load_conversations(file_path):
    return pd.read_excel(file_path)
