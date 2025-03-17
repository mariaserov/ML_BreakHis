
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

df = pd.read_csv("../data/metadata.csv")

train, test = train_test_split(df, test_size=0.25, random_state=42)

holdout_split = 1/3

train, val = train_test_split(train, test_size=holdout_split, random_state=42)

print("Train: ", train.shape)
print("Test: ", test.shape)
print("Holdout: ", val.shape)

train.to_csv("../data/train.csv", index=False)
test.to_csv("../data/test.csv", index=False)
val.to_csv("../data/holdout.csv", index=False)