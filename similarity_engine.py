
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class SimilarityEngine:
    def compute(self, matrix):
        return cosine_similarity(matrix)

    def save(self, sim):
        np.save("../models/similarity.npy", sim)

    def save_mapping(self, df):
        mapping = pd.Series(df.index, index=df["Title"]).drop_duplicates()
        mapping.to_csv("../models/title_index.csv")

if __name__ == "__main__":
    df = pd.read_csv("../data/cleaned_anime.csv")
    matrix = np.load("../models/tfidf_matrix.npy")
    engine = SimilarityEngine()
    sim = engine.compute(matrix)
    engine.save(sim)
    engine.save_mapping(df)
    print("Similarity engine complete.")
