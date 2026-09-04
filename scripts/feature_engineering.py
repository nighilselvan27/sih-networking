import pandas as pd
import numpy as np
from pathlib import Path

INPUT = Path("data/scenario1/ctu13_forward_clean.csv")
OUTPUT = Path("data/scenario1/ctu13_features.csv")

print("Loading dataset...")

df = pd.read_csv(INPUT)

print(f"Rows: {len(df):,}")

# --------------------------------------------------
# Basic cleanup
# --------------------------------------------------

df["Dur"] = pd.to_numeric(df["Dur"], errors="coerce").fillna(0)
df["TotPkts"] = pd.to_numeric(df["TotPkts"], errors="coerce").fillna(0)
df["TotBytes"] = pd.to_numeric(df["TotBytes"], errors="coerce").fillna(0)
df["SrcBytes"] = pd.to_numeric(df["SrcBytes"], errors="coerce").fillna(0)

df["Sport"] = pd.to_numeric(df["Sport"], errors="coerce").fillna(0)
df["Dport"] = pd.to_numeric(df["Dport"], errors="coerce").fillna(0)

# --------------------------------------------------
# Derived flow features
# --------------------------------------------------

df["DstBytes"] = (
    df["TotBytes"] - df["SrcBytes"]
).clip(lower=0)

df["PacketsPerSecond"] = (
    df["TotPkts"] / df["Dur"].replace(0, np.nan)
).replace([np.inf, -np.inf], np.nan).fillna(0)

df["BytesPerSecond"] = (
    df["TotBytes"] / df["Dur"].replace(0, np.nan)
).replace([np.inf, -np.inf], np.nan).fillna(0)

df["AvgPacketSize"] = (
    df["TotBytes"] / df["TotPkts"].replace(0, np.nan)
).replace([np.inf, -np.inf], np.nan).fillna(0)

df["SrcByteRatio"] = (
    df["SrcBytes"] / df["TotBytes"].replace(0, np.nan)
).replace([np.inf, -np.inf], np.nan).fillna(0)

df["DstByteRatio"] = (
    df["DstBytes"] / df["TotBytes"].replace(0, np.nan)
).replace([np.inf, -np.inf], np.nan).fillna(0)

# --------------------------------------------------
# Protocol encoding
# --------------------------------------------------

protocol_map = {
    "tcp": 1,
    "udp": 2,
    "icmp": 3,
    "igmp": 4,
    "rtcp": 5,
    "rtp": 6,
}

df["ProtocolCode"] = (
    df["Proto"]
    .astype(str)
    .str.lower()
    .map(protocol_map)
    .fillna(0)
)

# --------------------------------------------------
# TCP state encoding
# --------------------------------------------------

df["StateCode"] = (
    df["State"]
    .astype(str)
    .astype("category")
    .cat.codes
)

# --------------------------------------------------
# Target
# --------------------------------------------------

df["Target"] = (df["ThreatClass"] == "BOTNET").astype(int)

# --------------------------------------------------
# Select ML features
# IMPORTANT:
# IP addresses are intentionally NOT included.
# --------------------------------------------------

features = [
    "Dur",
    "ProtocolCode",
    "Sport",
    "Dport",
    "StateCode",
    "TotPkts",
    "TotBytes",
    "SrcBytes",
    "DstBytes",
    "PacketsPerSecond",
    "BytesPerSecond",
    "AvgPacketSize",
    "SrcByteRatio",
    "DstByteRatio",
]

result = df[features + ["Target", "ThreatClass", "StartTime"]].copy()

# Sort chronologically
result["StartTime"] = pd.to_datetime(
    result["StartTime"],
    errors="coerce"
)

result = result.sort_values("StartTime")

result.to_csv(OUTPUT, index=False)

print("\n========== FINAL DATASET ==========")
print(f"Rows: {len(result):,}")
print(f"Features: {len(features)}")

print("\n========== TARGET ==========")
print(result["ThreatClass"].value_counts())

print("\n========== FEATURES ==========")
for feature in features:
    print(feature)

print(f"\nSaved to: {OUTPUT}")