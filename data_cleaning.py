import pandas as pd

df = pd.read_csv("game_reviews.csv")
df = df.dropna()

if 'ID' in df.columns:
    df = df.drop(columns=['ID'])

df.to_csv("game_reviews_clean.csv", index=False)
