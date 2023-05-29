import pandas as pd


df = pd.read_csv("dataset.csv")

counts = df.groupby(["Emitter"]).count().iloc[:, 0]

print(counts[counts > 200])
