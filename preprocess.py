
import pandas as pd
import numpy as np

class Preprocessor:
    def __init__(self):
        self.rating_map = {
            "excellent":9.5,"good":8.0,"bad":5.0,"terrible":3.5,"average":7.0,
            "pending":np.nan,"not rated":np.nan,"unknown":np.nan,"5 stars":9.0,"?":np.nan
        }

    def load(self, path):
        return pd.read_csv(path, on_bad_lines='skip')

    def clean_rating(self, df):
        def conv(x):
            try: return float(x)
            except: return self.rating_map.get(str(x).strip().lower(), np.nan)
        df["Final_Rating"] = df["Rating"].apply(conv)
        return df

    def clean_episodes(self, df):
        def to_int(x):
            try: return int(x)
            except: return np.nan
        df["Episodes_Clean"] = df["Episodes"].apply(to_int)
        return df

    def finalize(self, df):
        df["Genre"] = df["Genre"].fillna("Unknown").astype(str)
        df.drop_duplicates(inplace=True)
        return df

if __name__ == "__main__":
    p = Preprocessor()
    df = p.load("../data/raw_anime.csv")
    df = p.clean_rating(df)
    df = p.clean_episodes(df)
    df = p.finalize(df)
    df.to_csv("../data/cleaned_anime.csv", index=False)
    print("Production preprocessing complete.")
