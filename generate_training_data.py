import pandas as pd

# Load the fetched emails
df = pd.read_csv("FetchedEmails.csv")

# Default to Not Phishing (0)
df["label"] = 0

# Define suspicious keywords
phishing_keywords = ["urgent", "free", "click here", "winner", "prize"]

# Mark rows with suspicious words as Phishing (1)
df["label"] = df["body"].str.lower().apply(
    lambda text: 1 if any(kw in text for kw in phishing_keywords) else 0
)

# Save back to the same file
df.to_csv("FetchedEmails.csv", index=False)
print("✅ 'label' column added successfully.")
