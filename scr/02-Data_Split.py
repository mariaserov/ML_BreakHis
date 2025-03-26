
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

df = pd.read_csv("../data/metadata.csv")

train, hold_out = train_test_split(df, test_size=0.25, random_state=42)

train, test = train_test_split(train, test_size=0.25, random_state=42)

print("Train: ", train.shape)
print("Test: ", test.shape)
print("Holdout: ", hold_out.shape)

train.to_csv("../data/train.csv", index=False)
test.to_csv("../data/test.csv", index=False)
hold_out.to_csv("../data/holdout.csv", index=False)