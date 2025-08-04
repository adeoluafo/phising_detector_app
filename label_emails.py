import pandas as pd

# Load the fetched emails
df = pd.read_csv("FetchedEmails.csv")

# Add a 'label' column: default to 0 (Not Phishing)
df["label"] = 0

# Mark as phishing (1) if suspicious keywords are found
phishing_keywords = ["urgent", "free", "click here", "verify", "winner", "login", "account", "password", "refund", "prize"]

df["label"] = df["body"].apply(
    lambda text: 1 if isinstance(text, str) and any(kw in text.lower() for kw in phishing_keywords) else 0
)


# Save it back
df.to_csv("FetchedEmails.csv", index=False)
print("✅ Label column added to FetchedEmails.csv")
