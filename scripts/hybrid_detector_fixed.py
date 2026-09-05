"""
CTU-13 HYBRID INTRUSION DETECTION (CORRECTED)

Fix vs previous version:
  - Loads the FIXED Isolation Forest model (scaled features,
    SportBucket/DportBucket instead of raw ports) instead of the
    original broken model.
  - Applies the saved StandardScaler to features before scoring
    with Isolation Forest.
  - XGBoost still uses its own original 30 raw features (unchanged,
    unaffected by the IF fix) at the tuned threshold (0.40).

TRAIN: Scenario 1 - 10
TEST:  Scenario 11 - 13
"""

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

TRAIN_SCENARIOS = list(range(1, 11))
TEST_SCENARIOS = list(range(11, 14))

TARGET = "Target"

XGBOOST_PATH = MODEL_DIR / "ctu13_multiscenario_xgboost.json"
XGBOOST_THRESHOLD = 0.40  # from your threshold-tuning run

# FIXED Isolation Forest artifacts
IF_MODEL_PATH = MODEL_DIR / "ctu13_isolation_forest_fixed.joblib"
IF_SCALER_PATH = MODEL_DIR / "ctu13_isolation_forest_scaler.joblib"
IF_FEATURES_PATH = MODEL_DIR / "ctu13_isolation_forest_features_fixed.json"

# XGBoost's original feature list (30 raw features incl. Sport/Dport)
XGB_FEATURES = [
    "Dur", "Sport", "Dport", "sTos", "dTos", "TotPkts", "TotBytes",
    "SrcBytes", "PacketsPerSecond", "BytesPerSecond", "AvgPacketSize",
    "DstBytes", "SrcByteRatio", "DstByteRatio", "SourceFlowCount30s",
    "UniqueDstIPs30s", "UniqueDstPorts30s", "UniqueSrcPorts30s",
    "SourceTotalBytes30s", "SourceTotalPackets30s",
    "DestinationFlowCount30s", "UniqueSrcIPs30s",
    "DestinationTotalBytes30s", "DestinationRepeatCount",
    "InterArrivalTime", "PairInterArrivalTime", "FlowsPerSecond30s",
    "PacketsPerSecond30s", "BytesPerSecond30s", "SourceOutboundRatio",
]

# Hybrid rule: flag as BOTNET if XGBoost says so, OR if Isolation
# Forest anomaly score clears its own calibrated percentile threshold.
# (Loaded below from the fixed IF training run's calibration logic —
# recomputed here the same way for consistency.)
IF_CALIBRATION_PERCENTILE = 95


def print_header(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def load_scenario(scenario):
    path = DATA_DIR / f"scenario{scenario}" / "ctu13_features.csv"
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found:\n{path}")
    print(f"Loading Scenario {scenario}: {path}")
    df = pd.read_csv(path)
    print(f"  Rows: {len(df):,}  Columns: {len(df.columns)}")
    return df


def get_target(df):
    return pd.to_numeric(df[TARGET], errors="coerce").fillna(0).astype(int)


def port_bucket(series):
    s = pd.to_numeric(series, errors="coerce").fillna(0)
    return pd.cut(
        s, bins=[-1, 1023, 49151, 65535], labels=[0, 1, 2]
    ).astype(float)


def prepare_xgb_features(df):
    X = df[XGB_FEATURES].copy()
    for c in XGB_FEATURES:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0)
    return X


def prepare_if_features(df, if_feature_list):
    df = df.copy()
    df["SportBucket"] = port_bucket(df["Sport"])
    df["DportBucket"] = port_bucket(df["Dport"])
    missing = [f for f in if_feature_list if f not in df.columns]
    if missing:
        raise ValueError("Missing IF features:\n" + "\n".join(missing))
    X = df[if_feature_list].copy()
    for c in if_feature_list:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0)
    return X


# ============================================================
# LOAD MODELS
# ============================================================

print_header("LOADING MODELS")

print("Loading XGBoost...")
xgb_model = xgb.XGBClassifier()
xgb_model.load_model(str(XGBOOST_PATH))
print("XGBoost loaded successfully.")

print("Loading FIXED Isolation Forest...")
if_model = joblib.load(IF_MODEL_PATH)
if_scaler = joblib.load(IF_SCALER_PATH)
with open(IF_FEATURES_PATH, "r", encoding="utf-8") as f:
    if_feature_list = json.load(f)
print("Isolation Forest (fixed) loaded successfully.")
print(f"IF features: {if_feature_list}")

# ============================================================
# CALIBRATE IF THRESHOLD FROM BENIGN TRAINING DATA
# (same approach as train_isolation_forest.py, recomputed here
#  so this script is self-contained)
# ============================================================

print_header("CALIBRATING ISOLATION FOREST THRESHOLD")

benign_samples = []
for scenario in TRAIN_SCENARIOS:
    df = load_scenario(scenario)
    y = get_target(df)
    benign = df[y == 0]
    sample_count = min(15000, len(benign))
    if sample_count < len(benign):
        benign = benign.sample(n=sample_count, random_state=42)
    benign_samples.append(benign)
    del df, y, benign

benign_all = pd.concat(benign_samples, ignore_index=True)
X_benign_if = prepare_if_features(benign_all, if_feature_list)
X_benign_if_scaled = if_scaler.transform(X_benign_if)

benign_scores = -if_model.decision_function(X_benign_if_scaled)
if_threshold = np.percentile(benign_scores, IF_CALIBRATION_PERCENTILE)
print(f"IF threshold ({IF_CALIBRATION_PERCENTILE}th percentile of benign scores): {if_threshold:.4f}")

del benign_samples, benign_all, X_benign_if, X_benign_if_scaled, benign_scores

# ============================================================
# LOAD TEST SCENARIOS
# ============================================================

print_header("LOADING UNSEEN TEST SCENARIOS")

test_frames = {}
for scenario in TEST_SCENARIOS:
    test_frames[scenario] = load_scenario(scenario)

total_test_rows = sum(len(df) for df in test_frames.values())
print(f"\nTOTAL TEST ROWS: {total_test_rows:,}")

# ============================================================
# HYBRID PREDICTION PER SCENARIO
# ============================================================

print_header("HYBRID PREDICTION")

all_results = []
overall_y_true = []
overall_xgb_pred = []
overall_if_pred = []
overall_hybrid_pred = []
overall_xgb_proba = []
overall_if_score = []

for scenario in TEST_SCENARIOS:
    print_header(f"PROCESSING SCENARIO {scenario}")
    df = test_frames[scenario]
    y_true = get_target(df).to_numpy()
    print(f"Rows: {len(df):,}")

    # --- XGBoost ---
    X_xgb = prepare_xgb_features(df)
    xgb_proba = xgb_model.predict_proba(X_xgb)[:, 1]
    xgb_pred = (xgb_proba >= XGBOOST_THRESHOLD).astype(int)

    # --- Isolation Forest (fixed) ---
    X_if = prepare_if_features(df, if_feature_list)
    X_if_scaled = if_scaler.transform(X_if)
    if_score = -if_model.decision_function(X_if_scaled)
    if_pred = (if_score >= if_threshold).astype(int)

    # --- Hybrid: flag if EITHER model flags it ---
    hybrid_pred = np.where((xgb_pred == 1) | (if_pred == 1), 1, 0)

    def report(name, y_pred, y_score=None):
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        auc = roc_auc_score(y_true, y_score) if y_score is not None else float("nan")
        pr_auc = average_precision_score(y_true, y_score) if y_score is not None else float("nan")
        print(f"\n{name}")
        print("-" * 40)
        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall   : {rec:.4f}")
        print(f"F1 Score : {f1:.4f}")
        if y_score is not None:
            print(f"ROC-AUC  : {auc:.4f}")
            print(f"PR-AUC   : {pr_auc:.4f}")
        return {"Scenario": scenario, "Model": name, "Accuracy": acc,
                "Precision": prec, "Recall": rec, "F1": f1,
                "ROC_AUC": auc, "PR_AUC": pr_auc}

    print(f"\nScenario {scenario} results:")
    all_results.append(report("XGBoost", xgb_pred, xgb_proba))
    all_results.append(report("IsolationForest (fixed)", if_pred, if_score))
    all_results.append(report("Hybrid", hybrid_pred, None))

    cm = confusion_matrix(y_true, hybrid_pred)
    print("\nHYBRID CONFUSION MATRIX")
    print(cm)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        print(f"\nTrue Negatives : {tn:,}")
        print(f"False Positives: {fp:,}")
        print(f"False Negatives: {fn:,}")
        print(f"True Positives : {tp:,}")

    print("\nPREDICTION DISTRIBUTION")
    print(f"XGBoost BOTNET       : {xgb_pred.sum():,}")
    print(f"Isolation anomalies  : {if_pred.sum():,}")
    print(f"Hybrid BOTNET        : {hybrid_pred.sum():,}")

    overall_y_true.append(y_true)
    overall_xgb_pred.append(xgb_pred)
    overall_if_pred.append(if_pred)
    overall_hybrid_pred.append(hybrid_pred)
    overall_xgb_proba.append(xgb_proba)
    overall_if_score.append(if_score)

    del df

# ============================================================
# OVERALL RESULTS
# ============================================================

print_header("OVERALL HYBRID RESULTS")

y_true_all = np.concatenate(overall_y_true)
xgb_pred_all = np.concatenate(overall_xgb_pred)
if_pred_all = np.concatenate(overall_if_pred)
hybrid_pred_all = np.concatenate(overall_hybrid_pred)
xgb_proba_all = np.concatenate(overall_xgb_proba)
if_score_all = np.concatenate(overall_if_score)


def overall_report(name, y_pred, y_score=None):
    acc = accuracy_score(y_true_all, y_pred)
    prec = precision_score(y_true_all, y_pred, zero_division=0)
    rec = recall_score(y_true_all, y_pred, zero_division=0)
    f1 = f1_score(y_true_all, y_pred, zero_division=0)
    print(f"\n{name}")
    print("-" * 40)
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")
    if y_score is not None:
        auc = roc_auc_score(y_true_all, y_score)
        pr_auc = average_precision_score(y_true_all, y_score)
        print(f"ROC-AUC  : {auc:.4f}")
        print(f"PR-AUC   : {pr_auc:.4f}")
    return acc, prec, rec, f1


xgb_overall = overall_report("XGBoost", xgb_pred_all, xgb_proba_all)
if_overall = overall_report("IsolationForest (fixed)", if_pred_all, if_score_all)
hybrid_overall = overall_report("HYBRID XGBOOST + ISOLATION FOREST (fixed)", hybrid_pred_all, None)

print_header("HYBRID IMPROVEMENT OVER XGBOOST ALONE")
print(f"Recall improvement   : {hybrid_overall[2] - xgb_overall[2]:+.4f}")
print(f"Precision change     : {hybrid_overall[1] - xgb_overall[1]:+.4f}")
print(f"F1 improvement       : {hybrid_overall[3] - xgb_overall[3]:+.4f}")

# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(all_results)
results_df.to_csv(MODEL_DIR / "hybrid_benchmark_fixed_per_scenario.csv", index=False)

print_header("DONE")
print(f"Per-scenario results saved to:\n{MODEL_DIR / 'hybrid_benchmark_fixed_per_scenario.csv'}")