import pandas as pd

df = pd.read_csv("FetchedEmails.csv")

# Check if 'label' column exists
if 'label' in df.columns:
    print("✅ 'label' column found!")
    print("Label counts:\n", df['label'].value_counts())
else:
    print("❌ 'label' column NOT found.")
