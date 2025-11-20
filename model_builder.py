
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

class ModelBuilder:
    def build_tfidf(self, df):
        vectorizer = TfidfVectorizer(stop_words='english')
        matrix = vectorizer.fit_transform(df["Genre"])
        return vectorizer, matrix

    def save(self, vectorizer, matrix):
        joblib.dump(vectorizer, "../models/tfidf_vectorizer.joblib")
        import numpy as np
        np.save("../models/tfidf_matrix.npy", matrix.toarray())
        print("TF-IDF model saved.")

if __name__ == "__main__":
    df = pd.read_csv("../data/cleaned_anime.csv")
    mb = ModelBuilder()
    vec, mat = mb.build_tfidf(df)
    mb.save(vec, mat)
