
import pandas as pd, numpy as np
from datetime import datetime

def reward(r):
    x=0
    if r.get("phone"): x+=20
    if r.get("paid_value",0)>0: x+=100+np.log1p(r["paid_value"])
    return x

def run():
    crm=pd.read_excel("crm.xlsx")
    chat=pd.read_excel("chat.xlsx")
    df=chat.merge(crm,on="phone",how="left")
    df["reward"]=df.apply(reward,axis=1)
    df["created_at"]=datetime.now().isoformat()
    df.to_csv("processed.csv",index=False)

if __name__=="__main__":
    run()
