import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

INPUT = Path("data/scenario1/ctu13_forward_clean.csv")
OUTPUT = Path("data/scenario1/ctu13_behavioral_features.csv")

# ============================================================
# LOAD DATASET
# ============================================================

print("Loading dataset...")

if not INPUT.exists():
    raise FileNotFoundError(
        f"\nInput file not found:\n{INPUT}\n\n"
        "Make sure ctu13_forward_clean.csv exists in data/scenario1/"
    )

df = pd.read_csv(INPUT)

print(f"Original rows: {len(df):,}")

# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "StartTime",
    "Dur",
    "Proto",
    "SrcAddr",
    "Sport",
    "Dir",
    "DstAddr",
    "Dport",
    "State",
    "TotPkts",
    "TotBytes",
    "SrcBytes",
    "ThreatClass",
]

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"\nMissing required columns:\n{missing}\n\n"
        f"Available columns:\n{list(df.columns)}"
    )

print("Required columns verified.")

# ============================================================
# TIME CONVERSION
# ============================================================

print("Processing timestamps...")

df["StartTime"] = pd.to_datetime(
    df["StartTime"],
    errors="coerce"
)

# Remove invalid timestamps
df = df.dropna(
    subset=["StartTime"]
).copy()

# Sort chronologically
df = df.sort_values(
    "StartTime"
).reset_index(drop=True)

# ============================================================
# NUMERIC CLEANING
# ============================================================

print("Cleaning numeric fields...")

numeric_columns = [
    "Dur",
    "Sport",
    "Dport",
    "TotPkts",
    "TotBytes",
    "SrcBytes",
]

for column in numeric_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

    df[column] = df[column].fillna(0)

# ============================================================
# BASIC FLOW FEATURES
# ============================================================

print("Creating basic flow features...")

# ------------------------------------------------------------
# Packets per second
# ------------------------------------------------------------

df["PacketsPerSecond"] = (
    df["TotPkts"] /
    df["Dur"].replace(0, np.nan)
)

df["PacketsPerSecond"] = (
    df["PacketsPerSecond"]
    .replace([np.inf, -np.inf], 0)
    .fillna(0)
)

# ------------------------------------------------------------
# Bytes per second
# ------------------------------------------------------------

df["BytesPerSecond"] = (
    df["TotBytes"] /
    df["Dur"].replace(0, np.nan)
)

df["BytesPerSecond"] = (
    df["BytesPerSecond"]
    .replace([np.inf, -np.inf], 0)
    .fillna(0)
)

# ------------------------------------------------------------
# Average packet size
# ------------------------------------------------------------

df["AvgPacketSize"] = (
    df["TotBytes"] /
    df["TotPkts"].replace(0, np.nan)
)

df["AvgPacketSize"] = (
    df["AvgPacketSize"]
    .replace([np.inf, -np.inf], 0)
    .fillna(0)
)

# ------------------------------------------------------------
# Source byte ratio
# ------------------------------------------------------------

df["SrcByteRatio"] = (
    df["SrcBytes"] /
    (df["TotBytes"] + 1)
)

# ------------------------------------------------------------
# Destination bytes
#
# CTU-13 provides total bytes and source bytes.
# Approximate destination bytes as:
#
# destination bytes = total bytes - source bytes
# ------------------------------------------------------------

df["DstBytes"] = (
    df["TotBytes"] -
    df["SrcBytes"]
)

df["DstBytes"] = (
    df["DstBytes"]
    .clip(lower=0)
)

# ------------------------------------------------------------
# Destination byte ratio
# ------------------------------------------------------------

df["DstByteRatio"] = (
    df["DstBytes"] /
    (df["TotBytes"] + 1)
)

# ============================================================
# TARGET
# ============================================================

print("Creating target...")

df["Target"] = (
    df["ThreatClass"]
    .astype(str)
    .str.upper()
    .eq("BOTNET")
    .astype(int)
)

print("\nTarget distribution:")

print(
    df["Target"]
    .value_counts()
    .sort_index()
)

# ============================================================
# TEMPORAL WINDOWS
# ============================================================

print("\nCreating temporal windows...")

# Convert timestamp to Unix seconds
timestamp_seconds = (
    df["StartTime"]
    .astype("int64") // 1_000_000_000
)

# 10-second window
df["TimeWindow10s"] = (
    timestamp_seconds // 10
)

# 30-second window
df["TimeWindow30s"] = (
    timestamp_seconds // 30
)

# ============================================================
# SOURCE BEHAVIOR
# ============================================================

print("Calculating source behavioral features...")

source_window = df.groupby(
    [
        "SrcAddr",
        "TimeWindow30s"
    ],
    sort=False
)

# ------------------------------------------------------------
# Number of flows generated by source
# ------------------------------------------------------------

df["SourceFlowCount30s"] = (
    source_window["SrcAddr"]
    .transform("count")
)

# ------------------------------------------------------------
# Unique destination IPs
# ------------------------------------------------------------

df["UniqueDstIPs30s"] = (
    source_window["DstAddr"]
    .transform("nunique")
)

# ------------------------------------------------------------
# Unique destination ports
# ------------------------------------------------------------

df["UniqueDstPorts30s"] = (
    source_window["Dport"]
    .transform("nunique")
)

# ------------------------------------------------------------
# Unique source ports
# ------------------------------------------------------------

df["UniqueSrcPorts30s"] = (
    source_window["Sport"]
    .transform("nunique")
)

# ------------------------------------------------------------
# Total bytes generated by source
# ------------------------------------------------------------

df["SourceTotalBytes30s"] = (
    source_window["TotBytes"]
    .transform("sum")
)

# ------------------------------------------------------------
# Total packets generated by source
# ------------------------------------------------------------

df["SourceTotalPackets30s"] = (
    source_window["TotPkts"]
    .transform("sum")
)

# ============================================================
# DESTINATION BEHAVIOR
# ============================================================

print("Calculating destination behavioral features...")

destination_window = df.groupby(
    [
        "DstAddr",
        "TimeWindow30s"
    ],
    sort=False
)

# ------------------------------------------------------------
# Number of flows received by destination
# ------------------------------------------------------------

df["DestinationFlowCount30s"] = (
    destination_window["DstAddr"]
    .transform("count")
)

# ------------------------------------------------------------
# Unique source IPs contacting destination
# ------------------------------------------------------------

df["UniqueSrcIPs30s"] = (
    destination_window["SrcAddr"]
    .transform("nunique")
)

# ------------------------------------------------------------
# Total destination traffic
# ------------------------------------------------------------

df["DestinationTotalBytes30s"] = (
    destination_window["TotBytes"]
    .transform("sum")
)

# ============================================================
# DESTINATION REPETITION
# ============================================================

print("Calculating destination repetition...")

df["DestinationRepeatCount"] = (
    df["SourceFlowCount30s"] -
    df["UniqueDstIPs30s"] +
    1
)

df["DestinationRepeatCount"] = (
    df["DestinationRepeatCount"]
    .clip(lower=0)
)

# ============================================================
# INTER-ARRIVAL TIME
# ============================================================

print("Calculating inter-arrival time...")

df["InterArrivalTime"] = (
    df.groupby("SrcAddr")["StartTime"]
    .diff()
    .dt.total_seconds()
)

df["InterArrivalTime"] = (
    df["InterArrivalTime"]
    .fillna(0)
)

# Remove extreme values
df["InterArrivalTime"] = (
    df["InterArrivalTime"]
    .clip(
        lower=0,
        upper=3600
    )
)

# ============================================================
# TRAFFIC INTENSITY
# ============================================================

print("Calculating traffic intensity...")

# ------------------------------------------------------------
# Flows per second
# ------------------------------------------------------------

df["FlowsPerSecond30s"] = (
    df["SourceFlowCount30s"] /
    30.0
)

# ------------------------------------------------------------
# Packets per second
# ------------------------------------------------------------

df["PacketsPerSecond30s"] = (
    df["SourceTotalPackets30s"] /
    30.0
)

# ------------------------------------------------------------
# Bytes per second
# ------------------------------------------------------------

df["BytesPerSecond30s"] = (
    df["SourceTotalBytes30s"] /
    30.0
)

# ============================================================
# SOURCE OUTBOUND RATIO
# ============================================================

print("Calculating source outbound ratio...")

df["SourceOutboundRatio"] = (
    df["SrcBytes"] /
    (df["SrcBytes"] + df["DstBytes"] + 1)
)

# ============================================================
# CLEAN INF / NAN VALUES
# ============================================================

print("Cleaning final values...")

numeric_features = [
    "Dur",
    "Sport",
    "Dport",
    "TotPkts",
    "TotBytes",
    "SrcBytes",
    "DstBytes",
    "PacketsPerSecond",
    "BytesPerSecond",
    "AvgPacketSize",
    "SrcByteRatio",
    "DstByteRatio",
    "SourceFlowCount30s",
    "UniqueDstIPs30s",
    "UniqueDstPorts30s",
    "UniqueSrcPorts30s",
    "SourceTotalBytes30s",
    "SourceTotalPackets30s",
    "DestinationFlowCount30s",
    "UniqueSrcIPs30s",
    "DestinationTotalBytes30s",
    "DestinationRepeatCount",
    "InterArrivalTime",
    "FlowsPerSecond30s",
    "PacketsPerSecond30s",
    "BytesPerSecond30s",
    "SourceOutboundRatio",
]

for column in numeric_features:

    df[column] = (
        pd.to_numeric(
            df[column],
            errors="coerce"
        )
        .replace(
            [np.inf, -np.inf],
            0
        )
        .fillna(0)
    )

# ============================================================
# BEHAVIORAL FEATURE LIST
# ============================================================

behavioral_features = [

    "SourceFlowCount30s",

    "UniqueDstIPs30s",

    "UniqueDstPorts30s",

    "UniqueSrcPorts30s",

    "SourceTotalBytes30s",

    "SourceTotalPackets30s",

    "DestinationFlowCount30s",

    "UniqueSrcIPs30s",

    "DestinationTotalBytes30s",

    "DestinationRepeatCount",

    "InterArrivalTime",

    "SourceOutboundRatio",

    "FlowsPerSecond30s",

    "PacketsPerSecond30s",

    "BytesPerSecond30s",
]

# ============================================================
# FINAL DATASET
# ============================================================

print("\n========== FINAL DATASET ==========")

print(
    f"Rows: {len(df):,}"
)

print(
    f"Columns: {len(df.columns)}"
)

print("\n========== BEHAVIORAL FEATURES ==========")

for feature in behavioral_features:
    print(feature)

# ============================================================
# SHOW SAMPLE
# ============================================================

print("\n========== SAMPLE ==========")

print(
    df[
        [
            "StartTime",
            "SrcAddr",
            "DstAddr",
            "ThreatClass",
            "Target",
            "SourceFlowCount30s",
            "UniqueDstIPs30s",
            "UniqueDstPorts30s",
            "DestinationRepeatCount",
            "InterArrivalTime",
            "FlowsPerSecond30s",
        ]
    ].head(10).to_string(index=False)
)

# ============================================================
# SAVE
# ============================================================

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT,
    index=False
)

print("\n========================================")
print("Behavioral feature engineering complete.")
print("========================================")

print(
    f"\nSaved to:\n{OUTPUT}"
)

print(
    f"\nFinal rows: {len(df):,}"
)

print(
    f"Final columns: {len(df.columns)}"
)