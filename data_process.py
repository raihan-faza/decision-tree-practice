import pandas as pd

df = pd.read_csv("drug200.csv")
print(df.info())
print(df.describe())
print(df.shape)
