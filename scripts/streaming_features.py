import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict, deque, Counter


# ============================================================
# CONFIG
# ============================================================

INPUT = Path(
    "data/scenario1/ctu13_forward_clean.csv"
)

OUTPUT = Path(
    "data/scenario1/ctu13_streaming_features.csv"
)

WINDOW_SECONDS = 30


# ============================================================
# LOAD
# ============================================================

print("Loading dataset...")

if not INPUT.exists():
    raise FileNotFoundError(
        f"Input file not found:\n{INPUT}"
    )

df = pd.read_csv(INPUT)

print(f"Original rows: {len(df):,}")


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [
    "StartTime",
    "Dur",
    "Proto",
    "SrcAddr",
    "Sport",
    "DstAddr",
    "Dport",
    "State",
    "TotPkts",
    "TotBytes",
    "SrcBytes",
    "ThreatClass",
]

missing = [
    c for c in required_columns
    if c not in df.columns
]

if missing:
    raise ValueError(
        f"Missing columns: {missing}"
    )

print("Required columns verified.")


# ============================================================
# TIMESTAMP
# ============================================================

print("Processing timestamps...")

df["StartTime"] = pd.to_datetime(
    df["StartTime"],
    errors="coerce"
)

df = df.dropna(
    subset=["StartTime"]
).copy()

# Streaming simulation requires chronological order
df = df.sort_values(
    "StartTime"
).reset_index(drop=True)


# ============================================================
# NUMERIC CLEANING
# ============================================================

print("Cleaning numeric columns...")

numeric_columns = [
    "Dur",
    "Sport",
    "Dport",
    "TotPkts",
    "TotBytes",
    "SrcBytes",
]

for col in numeric_columns:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    ).fillna(0)


# ============================================================
# BASIC FLOW FEATURES
# ============================================================

print("Creating basic flow features...")


# Packets / second
df["PacketsPerSecond"] = (
    df["TotPkts"]
    /
    df["Dur"].replace(0, np.nan)
)

df["PacketsPerSecond"] = (
    df["PacketsPerSecond"]
    .replace(
        [np.inf, -np.inf],
        0
    )
    .fillna(0)
)


# Bytes / second
df["BytesPerSecond"] = (
    df["TotBytes"]
    /
    df["Dur"].replace(0, np.nan)
)

df["BytesPerSecond"] = (
    df["BytesPerSecond"]
    .replace(
        [np.inf, -np.inf],
        0
    )
    .fillna(0)
)


# Average packet size
df["AvgPacketSize"] = (
    df["TotBytes"]
    /
    df["TotPkts"].replace(0, np.nan)
)

df["AvgPacketSize"] = (
    df["AvgPacketSize"]
    .replace(
        [np.inf, -np.inf],
        0
    )
    .fillna(0)
)


# Destination bytes
df["DstBytes"] = (
    df["TotBytes"] -
    df["SrcBytes"]
).clip(lower=0)


# Byte ratios
df["SrcByteRatio"] = (
    df["SrcBytes"]
    /
    (df["TotBytes"] + 1)
)

df["DstByteRatio"] = (
    df["DstBytes"]
    /
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
# STREAMING STATE
#
# IMPORTANT:
#
# We maintain counters instead of scanning the entire
# 30-second window for every row.
#
# This makes the implementation dramatically faster.
# ============================================================


# ------------------------------------------------------------
# SOURCE STATE
# ------------------------------------------------------------

source_windows = defaultdict(deque)

source_dst_ip_counts = defaultdict(Counter)

source_dst_port_counts = defaultdict(Counter)

source_src_port_counts = defaultdict(Counter)

source_bytes = defaultdict(float)

source_packets = defaultdict(float)


# ------------------------------------------------------------
# DESTINATION STATE
# ------------------------------------------------------------

destination_windows = defaultdict(deque)

destination_src_ip_counts = defaultdict(Counter)

destination_bytes = defaultdict(float)


# ------------------------------------------------------------
# LAST SEEN
# ------------------------------------------------------------

last_source_time = {}

last_pair_time = {}


# ============================================================
# OUTPUT ARRAYS
# ============================================================

source_flow_count_30s = []

unique_dst_ips_30s = []

unique_dst_ports_30s = []

unique_src_ports_30s = []

source_total_bytes_30s = []

source_total_packets_30s = []

destination_flow_count_30s = []

unique_src_ips_30s = []

destination_total_bytes_30s = []

destination_repeat_count = []

inter_arrival_time = []

pair_inter_arrival_time = []

flows_per_second_30s = []

packets_per_second_30s = []

bytes_per_second_30s = []

source_outbound_ratio = []


# ============================================================
# STREAM PROCESSING
# ============================================================

print(
    "\nProcessing flows sequentially..."
)

total_rows = len(df)


for index, row in df.iterrows():

    # ========================================================
    # CURRENT FLOW
    # ========================================================

    current_time = row["StartTime"].timestamp()

    src = str(row["SrcAddr"])

    dst = str(row["DstAddr"])

    sport = int(row["Sport"])

    dport = int(row["Dport"])

    total_bytes = float(row["TotBytes"])

    total_packets = float(row["TotPkts"])

    cutoff = (
        current_time -
        WINDOW_SECONDS
    )


    # ========================================================
    # SOURCE WINDOW
    # ========================================================

    src_queue = source_windows[src]

    src_dst_counter = (
        source_dst_ip_counts[src]
    )

    src_dport_counter = (
        source_dst_port_counts[src]
    )

    src_sport_counter = (
        source_src_port_counts[src]
    )


    # Remove expired flows
    while (
        src_queue
        and src_queue[0][0] <= cutoff
    ):

        (
            old_time,
            old_dst,
            old_dport,
            old_sport,
            old_bytes,
            old_packets
        ) = src_queue.popleft()


        # Update counters
        src_dst_counter[old_dst] -= 1

        if src_dst_counter[old_dst] <= 0:
            del src_dst_counter[old_dst]


        src_dport_counter[old_dport] -= 1

        if src_dport_counter[old_dport] <= 0:
            del src_dport_counter[old_dport]


        src_sport_counter[old_sport] -= 1

        if src_sport_counter[old_sport] <= 0:
            del src_sport_counter[old_sport]


        source_bytes[src] -= old_bytes

        source_packets[src] -= old_packets


    # ========================================================
    # SOURCE FEATURES
    #
    # These represent PREVIOUSLY observed traffic only.
    # ========================================================

    source_count = len(src_queue)

    source_flow_count_30s.append(
        source_count
    )

    unique_dst_ips_30s.append(
        len(src_dst_counter)
    )

    unique_dst_ports_30s.append(
        len(src_dport_counter)
    )

    unique_src_ports_30s.append(
        len(src_sport_counter)
    )

    source_total_bytes_30s.append(
        source_bytes[src]
    )

    source_total_packets_30s.append(
        source_packets[src]
    )


    # ========================================================
    # DESTINATION WINDOW
    # ========================================================

    dst_queue = destination_windows[dst]

    dst_src_counter = (
        destination_src_ip_counts[dst]
    )


    # Remove expired destination flows
    while (
        dst_queue
        and dst_queue[0][0] <= cutoff
    ):

        (
            old_time,
            old_src,
            old_sport,
            old_dport,
            old_bytes,
            old_packets
        ) = dst_queue.popleft()


        dst_src_counter[old_src] -= 1

        if dst_src_counter[old_src] <= 0:

            del dst_src_counter[old_src]


        destination_bytes[dst] -= (
            old_bytes
        )


    # ========================================================
    # DESTINATION FEATURES
    # ========================================================

    destination_flow_count_30s.append(
        len(dst_queue)
    )

    unique_src_ips_30s.append(
        len(dst_src_counter)
    )

    destination_total_bytes_30s.append(
        destination_bytes[dst]
    )


    # ========================================================
    # DESTINATION REPETITION
    # ========================================================

    destination_repeat_count.append(
        src_dst_counter.get(dst, 0)
    )


    # ========================================================
    # SOURCE INTER-ARRIVAL
    # ========================================================

    if src in last_source_time:

        delta = (
            current_time -
            last_source_time[src]
        )

        delta = max(
            0.0,
            min(delta, 3600.0)
        )

    else:

        delta = 0.0


    inter_arrival_time.append(
        delta
    )


    # ========================================================
    # SOURCE -> DESTINATION INTER-ARRIVAL
    # ========================================================

    pair = (
        src,
        dst
    )


    if pair in last_pair_time:

        pair_delta = (
            current_time -
            last_pair_time[pair]
        )

        pair_delta = max(
            0.0,
            min(pair_delta, 3600.0)
        )

    else:

        pair_delta = 0.0


    pair_inter_arrival_time.append(
        pair_delta
    )


    # ========================================================
    # RATE FEATURES
    # ========================================================

    flows_per_second_30s.append(
        source_count /
        WINDOW_SECONDS
    )

    packets_per_second_30s.append(
        source_packets_30s_value := (
            source_packets[src]
            /
            WINDOW_SECONDS
        )
    )

    bytes_per_second_30s.append(
        source_bytes[src]
        /
        WINDOW_SECONDS
    )


    # ========================================================
    # OUTBOUND RATIO
    # ========================================================

    current_src_bytes = float(
        row["SrcBytes"]
    )

    current_dst_bytes = float(
        row["DstBytes"]
    )

    outbound_ratio = (
        current_src_bytes
        /
        (
            current_src_bytes
            +
            current_dst_bytes
            +
            1
        )
    )

    source_outbound_ratio.append(
        outbound_ratio
    )


    # ========================================================
    # ADD CURRENT FLOW TO SOURCE STATE
    # ========================================================

    src_queue.append(
        (
            current_time,
            dst,
            dport,
            sport,
            total_bytes,
            total_packets
        )
    )


    src_dst_counter[dst] += 1

    src_dport_counter[dport] += 1

    src_sport_counter[sport] += 1

    source_bytes[src] += (
        total_bytes
    )

    source_packets[src] += (
        total_packets
    )


    # ========================================================
    # ADD CURRENT FLOW TO DESTINATION STATE
    # ========================================================

    dst_queue.append(
        (
            current_time,
            src,
            sport,
            dport,
            total_bytes,
            total_packets
        )
    )


    dst_src_counter[src] += 1

    destination_bytes[dst] += (
        total_bytes
    )


    # ========================================================
    # UPDATE LAST-SEEN TIMES
    # ========================================================

    last_source_time[src] = (
        current_time
    )

    last_pair_time[pair] = (
        current_time
    )


    # ========================================================
    # PROGRESS
    # ========================================================

    if (
        (index + 1) % 50000 == 0
    ):

        percentage = (
            (index + 1)
            /
            total_rows
            *
            100
        )

        print(
            f"Processed "
            f"{index + 1:,} / "
            f"{total_rows:,} "
            f"({percentage:.1f}%)"
        )


# ============================================================
# ADD FEATURES
# ============================================================

print(
    "\nAdding behavioral features..."
)


df["SourceFlowCount30s"] = (
    source_flow_count_30s
)

df["UniqueDstIPs30s"] = (
    unique_dst_ips_30s
)

df["UniqueDstPorts30s"] = (
    unique_dst_ports_30s
)

df["UniqueSrcPorts30s"] = (
    unique_src_ports_30s
)

df["SourceTotalBytes30s"] = (
    source_total_bytes_30s
)

df["SourceTotalPackets30s"] = (
    source_total_packets_30s
)

df["DestinationFlowCount30s"] = (
    destination_flow_count_30s
)

df["UniqueSrcIPs30s"] = (
    unique_src_ips_30s
)

df["DestinationTotalBytes30s"] = (
    destination_total_bytes_30s
)

df["DestinationRepeatCount"] = (
    destination_repeat_count
)

df["InterArrivalTime"] = (
    inter_arrival_time
)

df["PairInterArrivalTime"] = (
    pair_inter_arrival_time
)

df["FlowsPerSecond30s"] = (
    flows_per_second_30s
)

df["PacketsPerSecond30s"] = (
    packets_per_second_30s
)

df["BytesPerSecond30s"] = (
    bytes_per_second_30s
)

df["SourceOutboundRatio"] = (
    source_outbound_ratio
)


# ============================================================
# CLEAN NUMERIC FEATURES
# ============================================================

print(
    "Cleaning final values..."
)


behavioral_features = [

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
    "PairInterArrivalTime",

    "FlowsPerSecond30s",
    "PacketsPerSecond30s",
    "BytesPerSecond30s",

    "SourceOutboundRatio",
]


for column in behavioral_features:

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
# SUMMARY
# ============================================================

print(
    "\n=========================================="
)

print(
    "STREAMING FEATURE DATASET"
)

print(
    "=========================================="
)

print(
    f"Rows: {len(df):,}"
)

print(
    f"Columns: {len(df.columns)}"
)


print(
    "\n========== BEHAVIORAL FEATURES =========="
)


for feature in behavioral_features:

    print(
        feature
    )


# ============================================================
# SAMPLE
# ============================================================

print(
    "\n========== SAMPLE =========="
)


sample_columns = [

    "StartTime",

    "SrcAddr",
    "DstAddr",

    "ThreatClass",
    "Target",

    "SourceFlowCount30s",

    "UniqueDstIPs30s",

    "UniqueDstPorts30s",

    "UniqueSrcPorts30s",

    "DestinationRepeatCount",

    "InterArrivalTime",

    "PairInterArrivalTime",

    "FlowsPerSecond30s",

    "PacketsPerSecond30s",

    "BytesPerSecond30s",

    "SourceOutboundRatio",
]


print(
    df[
        sample_columns
    ]
    .head(10)
    .to_string(index=False)
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


# ============================================================
# COMPLETE
# ============================================================

print(
    "\n=========================================="
)

print(
    "STREAMING FEATURE ENGINEERING COMPLETE"
)

print(
    "=========================================="
)

print(
    f"\nSaved to:"
)

print(
    OUTPUT
)

print(
    f"\nFinal rows: {len(df):,}"
)

print(
    f"Final columns: {len(df.columns)}"
)