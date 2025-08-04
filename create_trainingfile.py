import pandas as pd

# Load the existing fetched emails (must include the 'label' column!)
df = pd.read_csv("FetchedEmails.csv")

# Check if 'label' column exists
if 'label' not in df.columns:
    print("❌ 'label' column NOT found in FetchedEmails.csv. Please run the labeling script first.")
else:
    # Save it as your training dataset
    df.to_csv("TrainingEmails.csv", index=False)
    print("✅ TrainingEmails.csv has been created.")
