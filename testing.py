import pandas as pd

df = pd.read_csv("FetchedEmails.csv")

# Print out the first few label values
print(df["label"].head())

# Count how many are 0s and 1s
print(df["label"].value_counts())