"""
CTU-13 CROSS-SCENARIO ISOLATION FOREST (FIXED)

Fixes vs original:
  1. Feature scaling (StandardScaler, fit on benign train only)
  2. Sport/Dport replaced with low-cardinality buckets
     (well-known / registered / ephemeral) instead of raw numbers,
     since raw high-cardinality ports distort distance/split-based
     anomaly scoring in Isolation Forest (this is a different failure
     mode than in tree-based supervised models like XGBoost).
  3. Threshold derived from a held-out slice of BENIGN training data
     (percentile of anomaly scores) instead of relying on
     IsolationForest's internal contamination-based .predict()
     threshold, which is not calibrated to your actual per-scenario
     anomaly rates.
  4. Reports metrics at multiple thresholds so you can choose.

TRAIN: Scenario 1 - 10 (benign only)
TEST:  Scenario 11 - 13
"""

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_SCENARIOS = list(range(1, 11))
TEST_SCENARIOS = list(range(11, 14))

# Raw numeric features (Sport/Dport removed -> replaced by buckets below)
BASE_FEATURES = [
    "Dur", "sTos", "dTos", "TotPkts", "TotBytes", "SrcBytes",
    "PacketsPerSecond", "BytesPerSecond", "AvgPacketSize", "DstBytes",
    "SrcByteRatio", "DstByteRatio", "SourceFlowCount30s", "UniqueDstIPs30s",
    "UniqueDstPorts30s", "UniqueSrcPorts30s", "SourceTotalBytes30s",
    "SourceTotalPackets30s", "DestinationFlowCount30s", "UniqueSrcIPs30s",
    "DestinationTotalBytes30s", "DestinationRepeatCount", "InterArrivalTime",
    "PairInterArrivalTime", "FlowsPerSecond30s", "PacketsPerSecond30s",
    "BytesPerSecond30s", "SourceOutboundRatio",
]

PORT_COLS = ["Sport", "Dport"]
TARGET = "Target"

BENIGN_SAMPLE_PER_SCENARIO = 15000
VALIDATION_FRACTION = 0.15  # slice of benign TRAIN data held out to pick a threshold

N_ESTIMATORS = 300
CONTAMINATION = "auto"
RANDOM_STATE = 42
N_JOBS = -1

MODEL_PATH = MODEL_DIR / "ctu13_isolation_forest_fixed.joblib"
SCALER_PATH = MODEL_DIR / "ctu13_isolation_forest_scaler.joblib"
FEATURE_PATH = MODEL_DIR / "ctu13_isolation_forest_features_fixed.json"


def print_header(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def port_bucket(series):
    """well-known (0-1023) / registered (1024-49151) / ephemeral (49152+)"""
    s = pd.to_numeric(series, errors="coerce").fillna(0)
    return pd.cut(
        s,
        bins=[-1, 1023, 49151, 65535],
        labels=[0, 1, 2],
    ).astype(float)


def load_scenario(scenario):
    path = DATA_DIR / f"scenario{scenario}" / "ctu13_features.csv"
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found:\n{path}")
    print(f"Loading Scenario {scenario}: {path}")
    df = pd.read_csv(path)
    print(f"  Rows: {len(df):,}")
    return df


def engineer_ports(df):
    df = df.copy()
    df["SportBucket"] = port_bucket(df["Sport"])
    df["DportBucket"] = port_bucket(df["Dport"])
    return df


FEATURES = BASE_FEATURES + ["SportBucket", "DportBucket"]


def prepare_features(df):
    df = engineer_ports(df)
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        raise ValueError("Missing required features:\n" + "\n".join(missing))
    X = df[FEATURES].copy()
    for c in FEATURES:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0)
    return X


def get_target(df):
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found.")
    return pd.to_numeric(df[TARGET], errors="coerce").fillna(0).astype(int)


# ============================================================
# LOAD BENIGN TRAINING DATA
# ============================================================

print_header("CTU-13 ISOLATION FOREST (FIXED)")
print(f"\nTRAIN: Scenario {TRAIN_SCENARIOS[0]}-{TRAIN_SCENARIOS[-1]}")
print(f"TEST:  Scenario {TEST_SCENARIOS[0]}-{TEST_SCENARIOS[-1]}")

print_header("LOADING BENIGN TRAINING DATA")

benign_samples = []
for scenario in TRAIN_SCENARIOS:
    df = load_scenario(scenario)
    y = get_target(df)
    benign = df[y == 0]
    sample_count = min(BENIGN_SAMPLE_PER_SCENARIO, len(benign))
    if sample_count < len(benign):
        benign = benign.sample(n=sample_count, random_state=RANDOM_STATE)
    benign_samples.append(benign)
    del df, y, benign

benign_all = pd.concat(benign_samples, ignore_index=True)
print(f"\nTotal benign rows selected: {len(benign_all):,}")

X_benign_all = prepare_features(benign_all)

# Hold out a validation slice of BENIGN data to calibrate the threshold
n_val = int(len(X_benign_all) * VALIDATION_FRACTION)
rng = np.random.RandomState(RANDOM_STATE)
val_idx = rng.choice(len(X_benign_all), size=n_val, replace=False)
train_mask = np.ones(len(X_benign_all), dtype=bool)
train_mask[val_idx] = False

X_train_fit = X_benign_all.iloc[train_mask].reset_index(drop=True)
X_val_benign = X_benign_all.iloc[val_idx].reset_index(drop=True)

print(f"Benign rows for fitting model : {len(X_train_fit):,}")
print(f"Benign rows held out for threshold calibration: {len(X_val_benign):,}")

# ============================================================
# SCALE
# ============================================================

print_header("SCALING FEATURES")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_fit)
X_val_scaled = scaler.transform(X_val_benign)
print("Scaler fit on benign TRAIN rows only.")

# ============================================================
# TRAIN ISOLATION FOREST
# ============================================================

print_header("TRAINING ISOLATION FOREST")
model = IsolationForest(
    n_estimators=N_ESTIMATORS,
    contamination=CONTAMINATION,
    random_state=RANDOM_STATE,
    n_jobs=N_JOBS,
    max_samples="auto",
)
model.fit(X_train_scaled)
print("Training completed.")

joblib.dump(model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)
with open(FEATURE_PATH, "w", encoding="utf-8") as f:
    json.dump(FEATURES, f, indent=2)

# ============================================================
# CALIBRATE THRESHOLD FROM BENIGN VALIDATION SCORES
# ============================================================

print_header("CALIBRATING THRESHOLD FROM BENIGN VALIDATION DATA")

val_scores = -model.decision_function(X_val_scaled)  # higher = more anomalous

# Try a few percentiles of the benign score distribution as candidate cutoffs
candidate_percentiles = [90, 95, 97, 99]
candidate_thresholds = {
    p: np.percentile(val_scores, p) for p in candidate_percentiles
}

for p, t in candidate_thresholds.items():
    print(f"  {p}th percentile of benign scores -> threshold {t:.4f}")

# Default: use 95th percentile (expect ~5% false-positive rate on benign)
CHOSEN_PERCENTILE = 95
threshold = candidate_thresholds[CHOSEN_PERCENTILE]
print(f"\nUsing {CHOSEN_PERCENTILE}th percentile threshold: {threshold:.4f}")

# ============================================================
# LOAD TEST SCENARIOS
# ============================================================

print_header("LOADING TEST SCENARIOS")

test_frames = []
scenario_labels_list = []
for scenario in TEST_SCENARIOS:
    df = load_scenario(scenario)
    scenario_labels_list.extend([scenario] * len(df))
    test_frames.append(df)

test_all = pd.concat(test_frames, ignore_index=True)
scenario_labels = np.array(scenario_labels_list)

y_test = get_target(test_all).to_numpy()
X_test = prepare_features(test_all)
X_test_scaled = scaler.transform(X_test)

print(f"\nTotal test rows: {len(X_test):,}")

# ============================================================
# SCORE + PREDICT AT CALIBRATED THRESHOLD
# ============================================================

print_header("SCORING TEST DATA")

anomaly_scores = -model.decision_function(X_test_scaled)
y_pred = (anomaly_scores >= threshold).astype(int)

# ============================================================
# METRICS AT CALIBRATED THRESHOLD
# ============================================================

print_header(f"RESULTS AT {CHOSEN_PERCENTILE}th PERCENTILE THRESHOLD")

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, anomaly_scores)
pr_auc = average_precision_score(y_test, anomaly_scores)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}  (threshold-independent, sanity check this first)")
print(f"PR-AUC   : {pr_auc:.4f}")

print(classification_report(y_test, y_pred, target_names=["BENIGN", "BOTNET"], digits=4, zero_division=0))

cm = confusion_matrix(y_test, y_pred)
print(cm)

# ============================================================
# METRICS ACROSS MULTIPLE THRESHOLDS (for comparison / tuning)
# ============================================================

print_header("METRICS ACROSS CANDIDATE THRESHOLDS")
for p, t in candidate_thresholds.items():
    yp = (anomaly_scores >= t).astype(int)
    print(
        f"  P{p} (t={t:.3f}): "
        f"Precision={precision_score(y_test, yp, zero_division=0):.4f}  "
        f"Recall={recall_score(y_test, yp, zero_division=0):.4f}  "
        f"F1={f1_score(y_test, yp, zero_division=0):.4f}"
    )

# ============================================================
# PER-SCENARIO (using calibrated threshold + ROC-AUC which is threshold-free)
# ============================================================

print_header("PER-SCENARIO RESULTS")

scenario_results = []
for scenario in TEST_SCENARIOS:
    mask = scenario_labels == scenario
    y_true_s = y_test[mask]
    y_pred_s = y_pred[mask]
    scores_s = anomaly_scores[mask]

    s_acc = accuracy_score(y_true_s, y_pred_s)
    s_prec = precision_score(y_true_s, y_pred_s, zero_division=0)
    s_rec = recall_score(y_true_s, y_pred_s, zero_division=0)
    s_f1 = f1_score(y_true_s, y_pred_s, zero_division=0)
    try:
        s_auc = roc_auc_score(y_true_s, scores_s)
    except Exception:
        s_auc = float("nan")

    print(f"\nScenario {scenario}")
    print("-" * 40)
    print(f"Rows      : {len(y_true_s):,}")
    print(f"Accuracy  : {s_acc:.4f}")
    print(f"Precision : {s_prec:.4f}")
    print(f"Recall    : {s_rec:.4f}")
    print(f"F1 Score  : {s_f1:.4f}")
    print(f"ROC-AUC   : {s_auc:.4f}  <-- check this is >0.5; if not, something is still off for this scenario")

    scenario_results.append({
        "Scenario": scenario, "Rows": len(y_true_s), "Accuracy": s_acc,
        "Precision": s_prec, "Recall": s_rec, "F1": s_f1, "ROC_AUC": s_auc,
    })

results_df = pd.DataFrame(scenario_results)
results_df.to_csv(MODEL_DIR / "isolation_forest_benchmark_fixed.csv", index=False)

print_header("DONE")
print(f"Model saved to:  {MODEL_PATH}")
print(f"Scaler saved to: {SCALER_PATH}")