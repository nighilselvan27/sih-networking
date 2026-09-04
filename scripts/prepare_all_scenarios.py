import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")


def classify_label(label):
    label = str(label).lower().strip()

    if "botnet" in label:
        return "BOTNET"

    if "normal" in label:
        return "BENIGN"

    if "background" in label:
        return "BENIGN"

    return "OTHER"


def process_scenario(scenario_number):
    scenario_dir = DATA_DIR / f"scenario{scenario_number}"

    files = list(scenario_dir.glob("*.binetflow"))

    if not files:
        print(f"\nScenario {scenario_number}: NO .binetflow FILE FOUND")
        return None

    input_file = files[0]

    output_file = scenario_dir / "ctu13_forward_clean.csv"

    print("\n" + "=" * 60)
    print(f"PROCESSING SCENARIO {scenario_number}")
    print("=" * 60)

    print(f"Input: {input_file}")

    df = pd.read_csv(input_file)

    print(f"Original records: {len(df):,}")

    # ---------------------------------------------------------
    # Required columns
    # ---------------------------------------------------------

    required = [
        "StartTime",
        "Dur",
        "Proto",
        "SrcAddr",
        "Sport",
        "Dir",
        "DstAddr",
        "Dport",
        "State",
        "sTos",
        "dTos",
        "TotPkts",
        "TotBytes",
        "SrcBytes",
        "Label"
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        print(f"Missing columns: {missing}")
        return None

    # ---------------------------------------------------------
    # Keep only forward direction
    # ---------------------------------------------------------

    df["Dir"] = df["Dir"].astype(str).str.strip()

    df = df[df["Dir"] == "->"].copy()

    print(f"Forward directional records: {len(df):,}")

    # ---------------------------------------------------------
    # Clean string columns
    # ---------------------------------------------------------

    for col in [
        "Proto",
        "SrcAddr",
        "DstAddr",
        "Sport",
        "Dport",
        "State",
        "Label"
    ]:
        df[col] = df[col].astype(str).str.strip()

    # ---------------------------------------------------------
    # Create threat class
    # ---------------------------------------------------------

    df["ThreatClass"] = df["Label"].apply(classify_label)

    # Keep only BENIGN and BOTNET
    df = df[
        df["ThreatClass"].isin(["BENIGN", "BOTNET"])
    ].copy()

    print("\nClass distribution:")

    print(
        df["ThreatClass"].value_counts()
    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    df.to_csv(output_file, index=False)

    print(f"\nSaved: {output_file}")
    print(f"Final records: {len(df):,}")

    return df


# =============================================================
# MAIN
# =============================================================

print("\n" + "=" * 60)
print("CTU-13 MULTI-SCENARIO PREPROCESSING")
print("=" * 60)

results = {}

for scenario in range(1, 14):

    result = process_scenario(scenario)

    if result is not None:
        results[scenario] = len(result)


print("\n" + "=" * 60)
print("PROCESSING SUMMARY")
print("=" * 60)

for scenario, count in results.items():
    print(
        f"Scenario {scenario:2d}: "
        f"{count:,} records"
    )

print("\nDone.")