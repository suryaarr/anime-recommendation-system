
import streamlit as st
from src.recommender import AnimeRecommender

st.title("Anime Recommendation System")
rec = AnimeRecommender()
title = st.text_input("Enter Anime Title:")

if st.button("Recommend"):
    result = rec.recommend(title)
    st.write(result)
