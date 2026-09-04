"""
Diagnostic: compare BENIGN feature distributions between
training scenarios (1-10) and scenario 13, to check whether
scenario 13's "normal" traffic has drifted from what the
Isolation Forest learned as normal.

Run this after train_isolation_forest.py so the CSVs are already
cleaned/engineered.
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

FEATURES_TO_CHECK = [
    "TotBytes", "TotPkts", "PacketsPerSecond", "BytesPerSecond",
    "SourceFlowCount30s", "UniqueDstIPs30s", "UniqueDstPorts30s",
    "SourceTotalBytes30s", "SourceOutboundRatio", "InterArrivalTime",
]

def load_benign(scenario):
    path = DATA_DIR / f"scenario{scenario}" / "ctu13_features.csv"
    df = pd.read_csv(path)
    y = pd.to_numeric(df["Target"], errors="coerce").fillna(0).astype(int)
    return df[y == 0]

print("Loading benign traffic from training scenarios (1-10)...")
train_benign_frames = [load_benign(s) for s in range(1, 11)]
train_benign = pd.concat(train_benign_frames, ignore_index=True)

print("Loading benign traffic from scenario 13...")
s13_benign = load_benign(13)

print("\n" + "=" * 80)
print(f"{'Feature':<25}{'Train(1-10) median':>20}{'Scenario13 median':>20}{'Ratio':>15}")
print("=" * 80)

for feat in FEATURES_TO_CHECK:
    train_med = pd.to_numeric(train_benign[feat], errors="coerce").median()
    s13_med = pd.to_numeric(s13_benign[feat], errors="coerce").median()
    ratio = (s13_med / train_med) if train_med != 0 else float("nan")
    print(f"{feat:<25}{train_med:>20.4f}{s13_med:>20.4f}{ratio:>15.2f}x")

print("\nLarge ratios (>3x or <0.33x) suggest scenario 13's benign traffic")
print("looks meaningfully different from the training scenarios' benign")
print("traffic on that feature -- likely contributor to the Isolation")
print("Forest treating scenario 13 benign flows as anomalous.")