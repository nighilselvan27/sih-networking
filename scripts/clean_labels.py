import pandas as pd
from pathlib import Path

INPUT = Path("data/scenario1/ctu13_forward.csv")
OUTPUT = Path("data/scenario1/ctu13_forward_clean.csv")

print("Loading directional dataset...")

df = pd.read_csv(INPUT)

def classify_label(label):
    label = str(label).lower()

    if "botnet" in label:
        return "BOTNET"

    if "normal" in label:
        return "BENIGN"

    if "background" in label:
        return "BENIGN"

    if label.startswith("flow=to-") or label.startswith("flow=from-"):
        return "OTHER"

    return "OTHER"


df["ThreatClass"] = df["Label"].apply(classify_label)

print("\nOriginal labels:")
print(df["Label"].value_counts().head(20))

print("\nNew classes:")
print(df["ThreatClass"].value_counts())

# Keep only classes we understand for the first model
df = df[df["ThreatClass"].isin(["BENIGN", "BOTNET"])].copy()

print("\nFinal dataset:")
print(df["ThreatClass"].value_counts())

df.to_csv(OUTPUT, index=False)

print(f"\nSaved: {OUTPUT}")
print(f"Rows: {len(df):,}")