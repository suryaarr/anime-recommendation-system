
import numpy as np
import pandas as pd
import joblib

class AnimeRecommender:
    def __init__(self):
        self.df = pd.read_csv("../data/cleaned_anime.csv")
        self.sim = np.load("../models/similarity.npy")
        self.mapping = pd.read_csv("../models/title_index.csv", index_col=0, squeeze=True)

    def recommend(self, title, n=10):
        if title not in self.mapping.index:
            return ["Title not found"]
        idx = self.mapping[title]
        scores = list(enumerate(self.sim[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:n+1]
        return self.df["Title"].iloc[[s[0] for s in scores]].tolist()

if __name__ == "__main__":
    r = AnimeRecommender()
    recs = r.recommend("Fullmetal Alchemist: Brotherhood")
    print("Recommendations:", recs)
