
import joblib
from sklearn.linear_model import LogisticRegression
import numpy as np

class Ranker:
    def __init__(self):
        self.model = LogisticRegression()

    def feat(self,x):
        return np.array([
            x.get("score",0),
            x.get("paid_value",0),
            x.get("reward",0)
        ])

    def train(self, pairs):
        X,y=[],[]
        for a,b,label in pairs:
            X.append(self.feat(a)-self.feat(b))
            y.append(label)
        self.model.fit(X,y)
        joblib.dump(self.model,"ranker.pkl")

    def load(self):
        self.model = joblib.load("ranker.pkl")

    def score(self,x):
        return self.model.predict_proba([self.feat(x)])[0][1]
