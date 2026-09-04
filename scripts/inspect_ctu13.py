import pandas as pd
from pathlib import Path

FILE = Path("data/scenario1/capture20110810.binetflow")

print("Loading CTU-13...")
df = pd.read_csv(FILE)

print("\n========== SHAPE ==========")
print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\n========== COLUMNS ==========")
for column in df.columns:
    print(column)

print("\n========== FIRST 5 ROWS ==========")
print(df.head().to_string())

print("\n========== DATA TYPES ==========")
print(df.dtypes)

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== PROTOCOLS ==========")
print(df["Proto"].value_counts())

print("\n========== DIRECTIONS ==========")
print(df["Dir"].value_counts())

print("\n========== LABELS ==========")
print(df["Label"].value_counts())

print("\n========== LABEL PREFIXES ==========")
print(
    df["Label"]
    .astype(str)
    .str.split("-")
    .str[0]
    .value_counts()
)