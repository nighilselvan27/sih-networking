import pandas as pd
from pathlib import Path

INPUT = Path("data/scenario1/capture20110810.binetflow")
OUTPUT = Path("data/scenario1/ctu13_forward.csv")

print("Loading CTU-13...")

df = pd.read_csv(INPUT)

print(f"Original records: {len(df):,}")

# Keep only flows observed in the forward direction
forward = df[df["Dir"].astype(str).str.strip() == "->"].copy()

print(f"Forward directional records: {len(forward):,}")

# Remove unnecessary whitespace
forward["Proto"] = forward["Proto"].astype(str).str.strip()
forward["SrcAddr"] = forward["SrcAddr"].astype(str).str.strip()
forward["DstAddr"] = forward["DstAddr"].astype(str).str.strip()
forward["Label"] = forward["Label"].astype(str).str.strip()

# Save
forward.to_csv(OUTPUT, index=False)

print(f"\nSaved to: {OUTPUT}")
print("\nProtocol distribution:")
print(forward["Proto"].value_counts())

print("\nLabel distribution:")
print(forward["Label"].value_counts().head(20))