"""
CTU-13 CALIBRATED SYNTHETIC ATTACK GENERATOR

Generates synthetic:
    - SYN_FLOOD
    - UDP_FLOOD
    - PORT_SCAN
    - BENIGN

The generator calibrates feature distributions from CTU-13
training scenarios 1-10.

Important:
    Synthetic traffic is NOT used for model training.
    It is used as an independent stress/validation benchmark.
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ======================================================================
# CONFIGURATION
# ======================================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = DATA_DIR / "synthetic"

SCENARIO_START = 1
SCENARIO_END = 10

ROWS_PER_CLASS = 20_000
RANDOM_SEED = 42

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(RANDOM_SEED)


# ======================================================================
# MODEL FEATURES
# ======================================================================

FEATURES = [
    "Dur",
    "Sport",
    "Dport",
    "sTos",
    "dTos",
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


# ======================================================================
# HELPERS
# ======================================================================

def banner(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def section(title):
    print()
    print("-" * 70)
    print(title)
    print("-" * 70)


def safe_numeric(df, columns):
    """
    Convert requested columns to numeric safely.
    """
    out = df.copy()

    for col in columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    return out


def clip_to_reference(values, reference):
    """
    Clip synthetic values to robust CTU-13 percentile boundaries.

    This prevents generated values from becoming physically absurd
    compared with the original CTU-13 feature space.
    """
    reference = np.asarray(reference, dtype=float)

    reference = reference[np.isfinite(reference)]

    if len(reference) == 0:
        return np.asarray(values, dtype=float)

    low = np.percentile(reference, 0.5)
    high = np.percentile(reference, 99.5)

    return np.clip(values, low, high)


def sample_empirical(reference, n):
    """
    Bootstrap from actual CTU-13 observations.

    This is preferable to inventing arbitrary distributions because
    the generated traffic remains inside the observed feature space.
    """
    reference = np.asarray(reference, dtype=float)

    reference = reference[np.isfinite(reference)]

    if len(reference) == 0:
        return np.zeros(n)

    indexes = rng.integers(0, len(reference), size=n)

    values = reference[indexes].astype(float, copy=True)

    return values


def perturb(values, scale=0.05):
    """
    Apply small independent noise.

    copy() is intentional.
    It prevents the 'output array is read-only' error.
    """
    values = np.asarray(values, dtype=float).copy()

    noise = rng.normal(
        loc=0.0,
        scale=np.maximum(np.abs(values) * scale, 1e-9),
        size=len(values),
    )

    values = values + noise

    return values


def force_positive(values, minimum=0.0):
    values = np.asarray(values, dtype=float).copy()
    values[~np.isfinite(values)] = minimum
    return np.maximum(values, minimum)


def integer_feature(values, minimum=None, maximum=None):
    values = np.rint(values).astype(np.int64)

    if minimum is not None:
        values = np.maximum(values, minimum)

    if maximum is not None:
        values = np.minimum(values, maximum)

    return values


# ======================================================================
# LOAD CTU-13
# ======================================================================

banner("CTU-13 CALIBRATED SYNTHETIC ATTACK GENERATOR")

print()
print("Configuration:")
print(f"  Calibration scenarios : {SCENARIO_START} - {SCENARIO_END}")
print(f"  Rows per class        : {ROWS_PER_CLASS:,}")
print(f"  Random seed           : {RANDOM_SEED}")
print(f"  Output directory      : {OUTPUT_DIR}")


section("LOADING CTU-13 CALIBRATION DATA")

frames = []

for scenario in range(SCENARIO_START, SCENARIO_END + 1):

    path = DATA_DIR / f"scenario{scenario}" / "ctu13_features.csv"

    print(f"Loading Scenario {scenario}: {path}")

    if not path.exists():
        raise FileNotFoundError(
            f"Missing calibration file:\n{path}"
        )

    df = pd.read_csv(path)

    print(f"  Rows    : {len(df):,}")
    print(f"  Columns : {len(df.columns)}")

    missing = [c for c in FEATURES if c not in df.columns]

    if missing:
        raise ValueError(
            f"Scenario {scenario} is missing features:\n{missing}"
        )

    frames.append(df)


calibration = pd.concat(frames, ignore_index=True)

print()
print(f"TOTAL CALIBRATION ROWS: {len(calibration):,}")


# ======================================================================
# CLEAN DATA
# ======================================================================

section("CLEANING CALIBRATION DATA")

calibration = safe_numeric(calibration, FEATURES)

before = len(calibration)

calibration = calibration.replace(
    [np.inf, -np.inf],
    np.nan
)

calibration = calibration.dropna(
    subset=FEATURES
).copy()

after = len(calibration)

print(f"Rows before cleaning : {before:,}")
print(f"Rows after cleaning  : {after:,}")
print(f"Rows removed         : {before - after:,}")


# ======================================================================
# IDENTIFY ATTACK / BENIGN DATA
# ======================================================================

section("IDENTIFYING CTU-13 TRAFFIC CLASSES")

if "Target" not in calibration.columns:
    raise ValueError(
        "Target column is required in CTU-13 feature files."
    )

calibration["Target"] = pd.to_numeric(
    calibration["Target"],
    errors="coerce"
)

calibration = calibration.dropna(
    subset=["Target"]
).copy()

calibration["Target"] = calibration["Target"].astype(int)

benign_reference = calibration[
    calibration["Target"] == 0
].copy()

attack_reference = calibration[
    calibration["Target"] == 1
].copy()

print(f"Benign reference rows : {len(benign_reference):,}")
print(f"Attack reference rows : {len(attack_reference):,}")

if len(benign_reference) == 0:
    raise ValueError("No benign CTU-13 rows found.")

if len(attack_reference) == 0:
    raise ValueError("No attack CTU-13 rows found.")


# ======================================================================
# BUILD EMPIRICAL DISTRIBUTIONS
# ======================================================================

section("BUILDING CTU-13 FEATURE DISTRIBUTIONS")

distributions = {}

for feature in FEATURES:

    benign_values = benign_reference[feature].to_numpy(
        dtype=float,
        copy=True
    )

    attack_values = attack_reference[feature].to_numpy(
        dtype=float,
        copy=True
    )

    distributions[feature] = {
        "benign": benign_values,
        "attack": attack_values,
    }

print(f"Learned distributions for {len(FEATURES)} features.")


# ======================================================================
# BASE GENERATOR
# ======================================================================

def generate_from_reference(reference_type, n):
    """
    Generate traffic using bootstrap sampling from CTU-13.

    A small amount of noise is added while staying within observed
    CTU-13 ranges.
    """

    result = pd.DataFrame(index=np.arange(n))

    for feature in FEATURES:

        reference = distributions[feature][reference_type]

        values = sample_empirical(reference, n)

        # Small variation around real CTU-13 samples.
        values = perturb(values, scale=0.025)

        # Keep values in robust CTU-13 range.
        values = clip_to_reference(values, reference)

        result[feature] = values

    return result


# ======================================================================
# SYN FLOOD
# ======================================================================

def generate_syn_flood(n):

    data = generate_from_reference("attack", n)

    # SYN flooding characteristics:
    #
    # - short flows
    # - many packets/flows
    # - high packet rate
    # - asymmetric byte ratio
    # - high source activity
    # - usually TCP destination traffic

    data["Dur"] = clip_to_reference(
        sample_empirical(
            distributions["Dur"]["attack"],
            n
        ) * rng.uniform(0.15, 0.60, n),
        distributions["Dur"]["attack"]
    )

    data["TotPkts"] = integer_feature(
        np.maximum(
            data["TotPkts"].to_numpy(),
            rng.integers(2, 80, n)
        ),
        minimum=1
    )

    data["SrcBytes"] = force_positive(
        data["SrcBytes"].to_numpy()
        * rng.uniform(0.8, 2.0, n)
    )

    data["DstBytes"] = force_positive(
        data["DstBytes"].to_numpy()
        * rng.uniform(0.02, 0.25, n)
    )

    data["TotBytes"] = (
        data["SrcBytes"] +
        data["DstBytes"]
    )

    data["SrcByteRatio"] = np.clip(
        data["SrcBytes"] /
        np.maximum(data["TotBytes"], 1.0),
        0.75,
        0.995
    )

    data["DstByteRatio"] = (
        1.0 - data["SrcByteRatio"]
    )

    data["PacketsPerSecond"] = (
        data["TotPkts"] /
        np.maximum(data["Dur"], 0.001)
    )

    data["BytesPerSecond"] = (
        data["TotBytes"] /
        np.maximum(data["Dur"], 0.001)
    )

    data["SourceFlowCount30s"] = integer_feature(
        np.maximum(
            data["SourceFlowCount30s"].to_numpy(),
            rng.integers(20, 250, n)
        ),
        minimum=1
    )

    data["FlowsPerSecond30s"] = (
        data["SourceFlowCount30s"] / 30.0
    )

    data["SourceTotalPackets30s"] = (
        data["SourceFlowCount30s"]
        * rng.uniform(2, 20, n)
    )

    data["SourceTotalBytes30s"] = (
        data["SourceTotalPackets30s"]
        * rng.uniform(40, 1000, n)
    )

    # SYN traffic generally has little destination response traffic.
    data["DestinationFlowCount30s"] = integer_feature(
        rng.integers(1, 40, n),
        minimum=1
    )

    return data


# ======================================================================
# UDP FLOOD
# ======================================================================

def generate_udp_flood(n):

    data = generate_from_reference("attack", n)

    # UDP flood:
    #
    # - high packet rate
    # - high byte rate
    # - repeated destination behavior
    # - many source packets
    # - relatively short flows

    data["Dur"] = clip_to_reference(
        sample_empirical(
            distributions["Dur"]["attack"],
            n
        ) * rng.uniform(0.2, 1.0, n),
        distributions["Dur"]["attack"]
    )

    data["TotPkts"] = integer_feature(
        np.maximum(
            data["TotPkts"].to_numpy(),
            rng.integers(20, 300, n)
        ),
        minimum=1
    )

    data["AvgPacketSize"] = np.clip(
        sample_empirical(
            distributions["AvgPacketSize"]["attack"],
            n
        ),
        50,
        1400
    )

    data["TotBytes"] = (
        data["TotPkts"] *
        data["AvgPacketSize"]
    )

    data["SrcByteRatio"] = np.clip(
        rng.uniform(0.75, 0.99, n),
        0.0,
        1.0
    )

    data["DstByteRatio"] = (
        1.0 - data["SrcByteRatio"]
    )

    data["SrcBytes"] = (
        data["TotBytes"] *
        data["SrcByteRatio"]
    )

    data["DstBytes"] = (
        data["TotBytes"] *
        data["DstByteRatio"]
    )

    data["PacketsPerSecond"] = (
        data["TotPkts"] /
        np.maximum(data["Dur"], 0.001)
    )

    data["BytesPerSecond"] = (
        data["TotBytes"] /
        np.maximum(data["Dur"], 0.001)
    )

    data["SourceFlowCount30s"] = integer_feature(
        rng.integers(30, 350, n),
        minimum=1
    )

    data["UniqueDstIPs30s"] = integer_feature(
        rng.integers(1, 10, n),
        minimum=1
    )

    data["UniqueDstPorts30s"] = integer_feature(
        rng.integers(1, 15, n),
        minimum=1
    )

    data["DestinationRepeatCount"] = integer_feature(
        rng.integers(20, 250, n),
        minimum=0
    )

    data["FlowsPerSecond30s"] = (
        data["SourceFlowCount30s"] / 30.0
    )

    data["PacketsPerSecond30s"] = (
        data["SourceTotalPackets30s"] / 30.0
    )

    data["BytesPerSecond30s"] = (
        data["SourceTotalBytes30s"] / 30.0
    )

    return data


# ======================================================================
# PORT SCAN
# ======================================================================

def generate_port_scan(n):

    data = generate_from_reference("attack", n)

    # Port scanning:
    #
    # - many destination ports
    # - many short flows
    # - low packet count per flow
    # - many unique destinations
    # - many unique source ports
    # - relatively low payload

    data["Dur"] = np.clip(
        rng.uniform(0.001, 3.0, n),
        0.0001,
        None
    )

    data["TotPkts"] = integer_feature(
        rng.integers(1, 8, n),
        minimum=1
    )

    data["AvgPacketSize"] = np.clip(
        rng.uniform(40, 300, n),
        40,
        300
    )

    data["TotBytes"] = (
        data["TotPkts"] *
        data["AvgPacketSize"]
    )

    data["SrcByteRatio"] = np.clip(
        rng.uniform(0.55, 1.0, n),
        0.0,
        1.0
    )

    data["DstByteRatio"] = (
        1.0 - data["SrcByteRatio"]
    )

    data["SrcBytes"] = (
        data["TotBytes"] *
        data["SrcByteRatio"]
    )

    data["DstBytes"] = (
        data["TotBytes"] *
        data["DstByteRatio"]
    )

    data["PacketsPerSecond"] = (
        data["TotPkts"] /
        np.maximum(data["Dur"], 0.001)
    )

    data["BytesPerSecond"] = (
        data["TotBytes"] /
        np.maximum(data["Dur"], 0.001)
    )

    # Strong port-scan signal.
    data["UniqueDstPorts30s"] = integer_feature(
        rng.integers(30, 1000, n),
        minimum=1
    )

    data["UniqueDstIPs30s"] = integer_feature(
        rng.integers(1, 20, n),
        minimum=1
    )

    data["UniqueSrcPorts30s"] = integer_feature(
        rng.integers(20, 500, n),
        minimum=1
    )

    data["UniqueSrcIPs30s"] = integer_feature(
        rng.integers(1, 5, n),
        minimum=1
    )

    data["SourceFlowCount30s"] = integer_feature(
        rng.integers(30, 400, n),
        minimum=1
    )

    data["DestinationFlowCount30s"] = integer_feature(
        rng.integers(20, 350, n),
        minimum=1
    )

    data["DestinationRepeatCount"] = integer_feature(
        rng.integers(0, 30, n),
        minimum=0
    )

    data["FlowsPerSecond30s"] = (
        data["SourceFlowCount30s"] / 30.0
    )

    data["PacketsPerSecond30s"] = (
        data["SourceTotalPackets30s"] / 30.0
    )

    data["BytesPerSecond30s"] = (
        data["SourceTotalBytes30s"] / 30.0
    )

    return data


# ======================================================================
# BENIGN
# ======================================================================

def generate_benign(n):

    # IMPORTANT:
    #
    # Generate a fresh copy from benign CTU-13 observations.
    # This fixes the read-only numpy error.

    data = generate_from_reference("benign", n)

    # Keep benign traffic close to observed CTU-13 behavior.
    # Only mild perturbation is applied.

    for feature in FEATURES:

        reference = distributions[feature]["benign"]

        values = data[feature].to_numpy(
            dtype=float,
            copy=True
        )

        values = clip_to_reference(
            values,
            reference
        )

        data[feature] = values

    return data


# ======================================================================
# GENERATE ALL TRAFFIC
# ======================================================================

section("GENERATING SYN FLOOD")

syn = generate_syn_flood(
    ROWS_PER_CLASS
)

print(f"Generated: {len(syn):,} rows")


section("GENERATING UDP FLOOD")

udp = generate_udp_flood(
    ROWS_PER_CLASS
)

print(f"Generated: {len(udp):,} rows")


section("GENERATING PORT SCAN")

scan = generate_port_scan(
    ROWS_PER_CLASS
)

print(f"Generated: {len(scan):,} rows")


section("GENERATING BENIGN BACKGROUND")

benign = generate_benign(
    ROWS_PER_CLASS
)

print(f"Generated: {len(benign):,} rows")


# ======================================================================
# LABEL DATA
# ======================================================================

syn["Target"] = 1
syn["AttackType"] = "SYN_FLOOD"

udp["Target"] = 1
udp["AttackType"] = "UDP_FLOOD"

scan["Target"] = 1
scan["AttackType"] = "PORT_SCAN"

benign["Target"] = 0
benign["AttackType"] = "BENIGN"


# ======================================================================
# COMBINE
# ======================================================================

section("COMBINING DATASETS")

synthetic = pd.concat(
    [
        syn,
        udp,
        scan,
        benign,
    ],
    ignore_index=True
)


# ======================================================================
# FINAL FEATURE RECONSTRUCTION
# ======================================================================

# Recalculate dependent features so the feature relationships
# remain internally consistent.

synthetic["TotBytes"] = (
    synthetic["SrcBytes"] +
    synthetic["DstBytes"]
)

synthetic["AvgPacketSize"] = (
    synthetic["TotBytes"] /
    np.maximum(synthetic["TotPkts"], 1)
)

synthetic["PacketsPerSecond"] = (
    synthetic["TotPkts"] /
    np.maximum(synthetic["Dur"], 0.001)
)

synthetic["BytesPerSecond"] = (
    synthetic["TotBytes"] /
    np.maximum(synthetic["Dur"], 0.001)
)

synthetic["SrcByteRatio"] = (
    synthetic["SrcBytes"] /
    np.maximum(synthetic["TotBytes"], 1.0)
)

synthetic["DstByteRatio"] = (
    synthetic["DstBytes"] /
    np.maximum(synthetic["TotBytes"], 1.0)
)

synthetic["FlowsPerSecond30s"] = (
    synthetic["SourceFlowCount30s"] / 30.0
)

synthetic["PacketsPerSecond30s"] = (
    synthetic["SourceTotalPackets30s"] / 30.0
)

synthetic["BytesPerSecond30s"] = (
    synthetic["SourceTotalBytes30s"] / 30.0
)

synthetic["SourceOutboundRatio"] = np.clip(
    synthetic["SourceOutboundRatio"],
    0.0,
    1.0
)


# ======================================================================
# SANITIZE
# ======================================================================

synthetic = synthetic.replace(
    [np.inf, -np.inf],
    np.nan
)

synthetic[FEATURES] = synthetic[FEATURES].fillna(0)

synthetic["SyntheticID"] = np.arange(
    1,
    len(synthetic) + 1
)


# Put metadata after model features.
synthetic = synthetic[
    FEATURES +
    [
        "Target",
        "AttackType",
        "SyntheticID",
    ]
]


# ======================================================================
# SAVE INDIVIDUAL DATASETS
# ======================================================================

syn_path = OUTPUT_DIR / "synthetic_syn_flood.csv"
udp_path = OUTPUT_DIR / "synthetic_udp_flood.csv"
scan_path = OUTPUT_DIR / "synthetic_port_scan.csv"
benign_path = OUTPUT_DIR / "synthetic_benign.csv"
all_path = OUTPUT_DIR / "synthetic_attacks.csv"

synthetic[
    synthetic["AttackType"] == "SYN_FLOOD"
].to_csv(
    syn_path,
    index=False
)

synthetic[
    synthetic["AttackType"] == "UDP_FLOOD"
].to_csv(
    udp_path,
    index=False
)

synthetic[
    synthetic["AttackType"] == "PORT_SCAN"
].to_csv(
    scan_path,
    index=False
)

synthetic[
    synthetic["AttackType"] == "BENIGN"
].to_csv(
    benign_path,
    index=False
)

synthetic.to_csv(
    all_path,
    index=False
)


# ======================================================================
# SUMMARY
# ======================================================================

section("SYNTHETIC DATASET SUMMARY")

print()
print(f"Total rows: {len(synthetic):,}")

print()
print("Attack distribution:")
print(
    synthetic["AttackType"]
    .value_counts()
    .to_string()
)

print()
print("Target distribution:")
print(
    synthetic["Target"]
    .value_counts()
    .sort_index()
    .to_string()
)

print()
print(
    f"Feature count: "
    f"{len(FEATURES)} model features + "
    f"Target + AttackType + SyntheticID"
)


# ======================================================================
# FEATURE RANGE CHECK
# ======================================================================

section("FEATURE RANGE CHECK")

for feature in FEATURES:

    values = synthetic[feature].to_numpy(
        dtype=float
    )

    print(
        f"{feature:<30} "
        f"min={np.min(values):.4f} "
        f"max={np.max(values):.4f} "
        f"mean={np.mean(values):.4f}"
    )


# ======================================================================
# SAVE GENERATOR CONFIG
# ======================================================================

config = {
    "generator": "CTU-13 calibrated synthetic traffic generator",
    "calibration_scenarios": [
        SCENARIO_START,
        SCENARIO_END
    ],
    "rows_per_class": ROWS_PER_CLASS,
    "random_seed": RANDOM_SEED,
    "features": FEATURES,
    "classes": [
        "BENIGN",
        "SYN_FLOOD",
        "UDP_FLOOD",
        "PORT_SCAN"
    ],
    "total_rows": int(len(synthetic)),
}

config_path = OUTPUT_DIR / "synthetic_generator_config.json"

with open(
    config_path,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        config,
        f,
        indent=2
    )


# ======================================================================
# COMPLETE
# ======================================================================

section("SYNTHETIC ATTACK GENERATION COMPLETE")

print()
print("Generated files:")

print(f"  {syn_path}")
print(f"  {udp_path}")
print(f"  {scan_path}")
print(f"  {benign_path}")
print(f"  {all_path}")
print(f"  {config_path}")

print()
print("Dataset:")
print(f"  Total rows : {len(synthetic):,}")
print(f"  Benign     : {(synthetic['Target'] == 0).sum():,}")
print(f"  Attacks    : {(synthetic['Target'] == 1).sum():,}")

print()
print("Attack types:")
print(f"  SYN Flood  : {(synthetic['AttackType'] == 'SYN_FLOOD').sum():,}")
print(f"  UDP Flood  : {(synthetic['AttackType'] == 'UDP_FLOOD').sum():,}")
print(f"  Port Scan  : {(synthetic['AttackType'] == 'PORT_SCAN').sum():,}")

print()
print("Next step:")
print("Run:")
print()
print("  python scripts/test_synthetic_attacks.py")

print()
print("=" * 70)
print("DONE")
print("=" * 70)