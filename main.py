
import os
import pandas as pd

from preprocessing.preprocess import Preprocessor
from src.model_builder import ModelBuilder
from src.similarity_engine import SimilarityEngine
from src.recommender import AnimeRecommender


def run_preprocessing():
    print("\nSTEP 1: Running Preprocessing...")
    p = Preprocessor()
    df = p.load("data/raw_anime.csv")
    df = p.clean_rating(df)
    df = p.clean_episodes(df)
    df = p.finalize(df)
    df.to_csv("data/cleaned_anime.csv", index=False)
    print("Preprocessing Completed. Saved cleaned_anime.csv")


def run_model_builder():
    print("\nSTEP 2: Building TF-IDF Model...")
    df = pd.read_csv("data/cleaned_anime.csv")
    mb = ModelBuilder()
    vectorizer, matrix = mb.build_tfidf(df)
    mb.save(vectorizer, matrix)
    print(" TF-IDF Model Saved.")


def run_similarity_engine():
    print("\n STEP 3: Computing Similarity Matrix...")
    df = pd.read_csv("data/cleaned_anime.csv")
    matrix = pd.np.load("models/tfidf_matrix.npy") if hasattr(pd, "np") else __import__("numpy").load("models/tfidf_matrix.npy")
    engine = SimilarityEngine()
    sim = engine.compute(matrix)
    engine.save(sim)
    engine.save_mapping(df)
    print("Similarity Matrix Saved.")


def run_recommender_demo():
    print("\n STEP 4: Running Recommendation Demo...")
    r = AnimeRecommender()
    sample_title = "Fullmetal Alchemist: Brotherhood"

    if sample_title in r.mapping.index:
        recs = r.recommend(sample_title, n=5)
        print(f"\n Top Recommendations for **{sample_title}**:")
        for i, anim in enumerate(recs, 1):
            print(f"{i}. {anim}")
    else:
        print("Sample anime not found in dataset.")


if __name__ == "__main__":
    print("\nAnime Recommendation System — FULL PRODUCTION RUN 🔥")

    run_preprocessing()
    run_model_builder()
    run_similarity_engine()
    run_recommender_demo()

    print("\nSystem run completed successfully!")
