import pandas as pd
import numpy as np

from pathlib import Path
from collections import defaultdict, deque, Counter


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path("data")

SCENARIOS = range(1, 14)

WINDOW_SECONDS = 30


# ============================================================
# REQUIRED COLUMNS
# ============================================================

REQUIRED_COLUMNS = [
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


# ============================================================
# NUMERIC COLUMNS
# ============================================================

NUMERIC_COLUMNS = [
    "Dur",
    "Sport",
    "Dport",
    "TotPkts",
    "TotBytes",
    "SrcBytes",
]


# ============================================================
# BEHAVIORAL FEATURES
# ============================================================

BEHAVIORAL_FEATURES = [

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


# ============================================================
# FINAL MODEL FEATURES
# ============================================================

MODEL_FEATURES = [

    "Dur",
    "Sport",
    "Dport",

    "TotPkts",
    "TotBytes",
    "SrcBytes",

    "PacketsPerSecond",
    "BytesPerSecond",
    "AvgPacketSize",

    "DstBytes",

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


# ============================================================
# LABEL CONVERSION
# ============================================================

def create_target(df):

    df["ThreatClass"] = (
        df["ThreatClass"]
        .fillna("UNKNOWN")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df["Target"] = (
        df["ThreatClass"]
        .eq("BOTNET")
        .astype(int)
    )

    return df


# ============================================================
# BASIC FLOW FEATURES
# ============================================================

def create_basic_features(df):

    print("Creating basic flow features...")

    # Packets / second
    df["PacketsPerSecond"] = (
        df["TotPkts"]
        /
        df["Dur"].replace(0, np.nan)
    )

    df["PacketsPerSecond"] = (
        df["PacketsPerSecond"]
        .replace([np.inf, -np.inf], 0)
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
        .replace([np.inf, -np.inf], 0)
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
        .replace([np.inf, -np.inf], 0)
        .fillna(0)
    )

    # Destination bytes
    df["DstBytes"] = (
        df["TotBytes"] - df["SrcBytes"]
    ).clip(lower=0)

    # Source byte ratio
    df["SrcByteRatio"] = (
        df["SrcBytes"]
        /
        (df["TotBytes"] + 1)
    )

    # Destination byte ratio
    df["DstByteRatio"] = (
        df["DstBytes"]
        /
        (df["TotBytes"] + 1)
    )

    return df


# ============================================================
# CLEAN INPUT
# ============================================================

def clean_input(df):

    print("Cleaning input columns...")

    # --------------------------------------------------------
    # TIMESTAMP
    # --------------------------------------------------------

    df["StartTime"] = pd.to_datetime(
        df["StartTime"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["StartTime"]
    ).copy()

    # --------------------------------------------------------
    # STRING COLUMNS
    # --------------------------------------------------------

    string_columns = [
        "Proto",
        "SrcAddr",
        "DstAddr",
        "State",
        "ThreatClass",
    ]

    for col in string_columns:

        if col in df.columns:

            df[col] = (
                df[col]
                .fillna("UNKNOWN")
                .astype(str)
                .str.strip()
            )

    # --------------------------------------------------------
    # NUMERIC COLUMNS
    # --------------------------------------------------------

    for col in NUMERIC_COLUMNS:

        if col in df.columns:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            ).fillna(0)

    # --------------------------------------------------------
    # PREVENT NEGATIVE VALUES
    # --------------------------------------------------------

    for col in [
        "Dur",
        "Sport",
        "Dport",
        "TotPkts",
        "TotBytes",
        "SrcBytes",
    ]:

        df[col] = df[col].clip(
            lower=0
        )

    # --------------------------------------------------------
    # CHRONOLOGICAL ORDER
    # --------------------------------------------------------

    df = (
        df
        .sort_values("StartTime")
        .reset_index(drop=True)
    )

    return df


# ============================================================
# STREAMING BEHAVIORAL FEATURES
# ============================================================

def create_streaming_features(df):

    print("Initializing streaming behavioral engine...")
    print("Processing flows sequentially...")

    # ========================================================
    # SOURCE STATE
    # ========================================================

    source_windows = defaultdict(deque)

    source_dst_ip_counts = defaultdict(Counter)

    source_dst_port_counts = defaultdict(Counter)

    source_src_port_counts = defaultdict(Counter)

    source_bytes = defaultdict(float)

    source_packets = defaultdict(float)

    # ========================================================
    # DESTINATION STATE
    # ========================================================

    destination_windows = defaultdict(deque)

    destination_src_ip_counts = defaultdict(Counter)

    destination_bytes = defaultdict(float)

    # ========================================================
    # LAST SEEN
    # ========================================================

    last_source_time = {}

    last_pair_time = {}

    # ========================================================
    # OUTPUT ARRAYS
    # ========================================================

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

    # ========================================================
    # PROCESS
    # ========================================================

    total_rows = len(df)

    for index, row in df.iterrows():

        # ----------------------------------------------------
        # CURRENT FLOW
        # ----------------------------------------------------

        current_time = row["StartTime"].timestamp()

        src = str(row["SrcAddr"])

        dst = str(row["DstAddr"])

        # IMPORTANT:
        # Convert ports safely to integers.
        # Missing values become 0.

        try:
            sport = int(float(row["Sport"]))
        except (ValueError, TypeError):
            sport = 0

        try:
            dport = int(float(row["Dport"]))
        except (ValueError, TypeError):
            dport = 0

        try:
            total_bytes = float(row["TotBytes"])
        except (ValueError, TypeError):
            total_bytes = 0.0

        try:
            total_packets = float(row["TotPkts"])
        except (ValueError, TypeError):
            total_packets = 0.0

        try:
            src_bytes_current = float(row["SrcBytes"])
        except (ValueError, TypeError):
            src_bytes_current = 0.0

        total_bytes = max(0.0, total_bytes)

        total_packets = max(0.0, total_packets)

        src_bytes_current = max(
            0.0,
            src_bytes_current
        )

        cutoff = (
            current_time
            -
            WINDOW_SECONDS
        )

        # ----------------------------------------------------
        # SOURCE WINDOW
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # REMOVE EXPIRED SOURCE FLOWS
        # ----------------------------------------------------

        while (
            src_queue
            and
            src_queue[0][0] <= cutoff
        ):

            (
                old_time,
                old_dst,
                old_dport,
                old_sport,
                old_bytes,
                old_packets,
            ) = src_queue.popleft()

            # Destination IP
            src_dst_counter[old_dst] -= 1

            if src_dst_counter[old_dst] <= 0:
                del src_dst_counter[old_dst]

            # Destination port
            src_dport_counter[old_dport] -= 1

            if src_dport_counter[old_dport] <= 0:
                del src_dport_counter[old_dport]

            # Source port
            src_sport_counter[old_sport] -= 1

            if src_sport_counter[old_sport] <= 0:
                del src_sport_counter[old_sport]

            # Bytes
            source_bytes[src] -= old_bytes

            # Packets
            source_packets[src] -= old_packets

        # ----------------------------------------------------
        # SOURCE FEATURES
        # ----------------------------------------------------

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
            max(
                0.0,
                source_bytes[src]
            )
        )

        source_total_packets_30s.append(
            max(
                0.0,
                source_packets[src]
            )
        )

        # ----------------------------------------------------
        # DESTINATION WINDOW
        # ----------------------------------------------------

        dst_queue = (
            destination_windows[dst]
        )

        dst_src_counter = (
            destination_src_ip_counts[dst]
        )

        # ----------------------------------------------------
        # REMOVE EXPIRED DESTINATION FLOWS
        # ----------------------------------------------------

        while (
            dst_queue
            and
            dst_queue[0][0] <= cutoff
        ):

            (
                old_time,
                old_src,
                old_sport,
                old_dport,
                old_bytes,
                old_packets,
            ) = dst_queue.popleft()

            dst_src_counter[old_src] -= 1

            if dst_src_counter[old_src] <= 0:
                del dst_src_counter[old_src]

            destination_bytes[dst] -= (
                old_bytes
            )

        # ----------------------------------------------------
        # DESTINATION FEATURES
        # ----------------------------------------------------

        destination_flow_count_30s.append(
            len(dst_queue)
        )

        unique_src_ips_30s.append(
            len(dst_src_counter)
        )

        destination_total_bytes_30s.append(
            max(
                0.0,
                destination_bytes[dst]
            )
        )

        # ----------------------------------------------------
        # DESTINATION REPETITION
        # ----------------------------------------------------

        destination_repeat_count.append(
            src_dst_counter.get(
                dst,
                0
            )
        )

        # ----------------------------------------------------
        # SOURCE INTER-ARRIVAL
        # ----------------------------------------------------

        if src in last_source_time:

            delta = (
                current_time
                -
                last_source_time[src]
            )

            delta = max(
                0.0,
                min(
                    delta,
                    3600.0
                )
            )

        else:

            delta = 0.0

        inter_arrival_time.append(
            delta
        )

        # ----------------------------------------------------
        # PAIR INTER-ARRIVAL
        # ----------------------------------------------------

        pair = (
            src,
            dst
        )

        if pair in last_pair_time:

            pair_delta = (
                current_time
                -
                last_pair_time[pair]
            )

            pair_delta = max(
                0.0,
                min(
                    pair_delta,
                    3600.0
                )
            )

        else:

            pair_delta = 0.0

        pair_inter_arrival_time.append(
            pair_delta
        )

        # ----------------------------------------------------
        # RATE FEATURES
        # ----------------------------------------------------

        flows_per_second_30s.append(
            source_count
            /
            WINDOW_SECONDS
        )

        packets_per_second_30s.append(
            source_packets[src]
            /
            WINDOW_SECONDS
        )

        bytes_per_second_30s.append(
            source_bytes[src]
            /
            WINDOW_SECONDS
        )

        # ----------------------------------------------------
        # OUTBOUND RATIO
        # ----------------------------------------------------

        current_dst_bytes = max(
            0.0,
            total_bytes
            -
            src_bytes_current
        )

        outbound_ratio = (
            src_bytes_current
            /
            (
                src_bytes_current
                +
                current_dst_bytes
                +
                1
            )
        )

        source_outbound_ratio.append(
            outbound_ratio
        )

        # ----------------------------------------------------
        # ADD CURRENT FLOW TO SOURCE STATE
        # ----------------------------------------------------

        src_queue.append(
            (
                current_time,
                dst,
                dport,
                sport,
                total_bytes,
                total_packets,
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

        # ----------------------------------------------------
        # ADD CURRENT FLOW TO DESTINATION STATE
        # ----------------------------------------------------

        dst_queue.append(
            (
                current_time,
                src,
                sport,
                dport,
                total_bytes,
                total_packets,
            )
        )

        dst_src_counter[src] += 1

        destination_bytes[dst] += (
            total_bytes
        )

        # ----------------------------------------------------
        # UPDATE LAST SEEN
        # ----------------------------------------------------

        last_source_time[src] = (
            current_time
        )

        last_pair_time[pair] = (
            current_time
        )

        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

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

    # ========================================================
    # ADD FEATURES TO DATAFRAME
    # ========================================================

    print("\nAdding behavioral features...")

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

    return df


# ============================================================
# CLEAN FINAL FEATURES
# ============================================================

def clean_final_features(df):

    print("Cleaning final feature values...")

    numeric_features = [
        col
        for col in MODEL_FEATURES
        if col in df.columns
    ]

    for col in numeric_features:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

        df[col] = (
            df[col]
            .replace(
                [np.inf, -np.inf],
                0
            )
            .fillna(0)
        )

    return df


# ============================================================
# PROCESS ONE SCENARIO
# ============================================================

def process_scenario(scenario):

    print()
    print("=" * 60)
    print(
        f"FEATURE ENGINEERING - SCENARIO {scenario}"
    )
    print("=" * 60)

    input_file = (
        BASE_DIR
        /
        f"scenario{scenario}"
        /
        "ctu13_forward_clean.csv"
    )

    output_file = (
        BASE_DIR
        /
        f"scenario{scenario}"
        /
        "ctu13_features.csv"
    )

    # --------------------------------------------------------
    # CHECK INPUT
    # --------------------------------------------------------

    if not input_file.exists():

        print(
            f"SKIPPING: input not found:"
        )

        print(input_file)

        return None

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    print(
        f"Loading: {input_file}"
    )

    df = pd.read_csv(
        input_file
    )

    print(
        f"Original rows: {len(df):,}"
    )

    # --------------------------------------------------------
    # REQUIRED COLUMNS
    # --------------------------------------------------------

    missing = [
        col
        for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Scenario {scenario} "
            f"missing columns: {missing}"
        )

    print(
        "Required columns verified."
    )

    # --------------------------------------------------------
    # CLEAN
    # --------------------------------------------------------

    df = clean_input(df)

    print(
        f"Rows after cleaning: "
        f"{len(df):,}"
    )

    # --------------------------------------------------------
    # BASIC FEATURES
    # --------------------------------------------------------

    df = create_basic_features(
        df
    )

    # --------------------------------------------------------
    # TARGET
    # --------------------------------------------------------

    print("Creating target...")

    df = create_target(
        df
    )

    print(
        "\nTarget distribution:"
    )

    print(
        df["Target"]
        .value_counts()
        .sort_index()
    )

    # --------------------------------------------------------
    # STREAMING FEATURES
    # --------------------------------------------------------

    df = create_streaming_features(
        df
    )

    # --------------------------------------------------------
    # FINAL CLEANING
    # --------------------------------------------------------

    df = clean_final_features(
        df
    )

    # --------------------------------------------------------
    # ENSURE CATEGORICAL COLUMNS ARE SAFE
    # --------------------------------------------------------

    for col in [
        "Proto",
        "State",
        "ThreatClass",
        "SrcAddr",
        "DstAddr",
    ]:

        if col in df.columns:

            df[col] = (
                df[col]
                .fillna("UNKNOWN")
                .astype(str)
                .str.strip()
            )

    # --------------------------------------------------------
    # OUTPUT DIRECTORY
    # --------------------------------------------------------

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    df.to_csv(
        output_file,
        index=False
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print(
        f"SCENARIO {scenario} COMPLETE"
    )
    print("=" * 60)

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )

    print(
        f"Saved: {output_file}"
    )

    print(
        "\nBehavioral features:"
    )

    for feature in BEHAVIORAL_FEATURES:

        print(
            f"  {feature}"
        )

    return len(df)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("CTU-13 MULTI-SCENARIO FEATURE ENGINEERING")
    print("=" * 70)

    results = {}

    # ========================================================
    # PROCESS SCENARIOS 1-13
    # ========================================================

    for scenario in SCENARIOS:

        try:

            rows = process_scenario(
                scenario
            )

            if rows is not None:

                results[
                    scenario
                ] = rows

        except Exception as error:

            print()
            print(
                "ERROR IN SCENARIO "
                f"{scenario}"
            )

            print(
                f"{type(error).__name__}: "
                f"{error}"
            )

            raise

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("MULTI-SCENARIO FEATURE ENGINEERING COMPLETE")
    print("=" * 70)

    print()

    total_rows = 0

    for scenario, rows in results.items():

        print(
            f"Scenario {scenario}: "
            f"{rows:,} rows"
        )

        total_rows += rows

    print()
    print(
        f"Total processed rows: "
        f"{total_rows:,}"
    )

    print()
    print(
        "Output files:"
    )

    for scenario in results:

        print(
            f"  data/scenario{scenario}/"
            f"ctu13_features.csv"
        )

    print()
    print("Done.")