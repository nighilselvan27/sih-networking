"""
CTU-13 GATED HYBRID INTRUSION DETECTOR
Full replacement version.

Design:
- XGBoost is the primary supervised detector.
- Isolation Forest is used only inside an XGBoost uncertainty band.
- Isolation Forest uses its own saved feature schema, scaler and model.
- IF scores are calibrated from benign training traffic.
- Confident XGBoost predictions are never overridden by IF.
- All metrics are calculated independently and consistently.
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
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

TRAIN_SCENARIOS = range(1, 11)
TEST_SCENARIOS = range(11, 14)

TARGET = "Target"
RANDOM_SEED = 42

XGB_MODEL_PATH = MODEL_DIR / "ctu13_multiscenario_xgboost.json"
XGB_THRESHOLD = 0.20

IF_MODEL_PATH = MODEL_DIR / "ctu13_isolation_forest_fixed.joblib"
IF_SCALER_PATH = MODEL_DIR / "ctu13_isolation_forest_scaler.joblib"
IF_FEATURES_PATH = MODEL_DIR / "ctu13_isolation_forest_features_fixed.json"

# IF is only allowed to participate in this XGBoost probability band.
UNCERTAINTY_LOW = 0.05
UNCERTAINTY_HIGH = 0.35

# Conservative IF calibration.
IF_CALIBRATION_PERCENTILE = 99.0

# IF must be strongly anomalous before it can change an uncertain
# XGBoost prediction.
IF_ANOMALY_QUANTILE = 0.99

# Hybrid rule:
#   confident XGB -> keep XGB
#   uncertain XGB + strong IF anomaly -> attack
#   otherwise -> XGB decision
#
# This is intentionally safer than freely blending IF scores.

XGB_FEATURES = [
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


# ============================================================
# HELPERS
# ============================================================

def header(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def get_target(df):
    if TARGET not in df.columns:
        raise ValueError(f"Missing target column: {TARGET}")

    return (
        pd.to_numeric(df[TARGET], errors="coerce")
        .fillna(0)
        .astype(int)
        .to_numpy()
    )


def clean_numeric(df):
    x = df.copy()

    for col in x.columns:
        x[col] = pd.to_numeric(x[col], errors="coerce")

    nan_before = int(x.isna().sum().sum())

    numeric = x.select_dtypes(include=[np.number])
    inf_before = int(np.isinf(numeric.to_numpy()).sum())

    x = x.replace([np.inf, -np.inf], np.nan)

    # IMPORTANT:
    # Use column medians from the current feature frame only.
    # This is only missing-value cleaning, not score normalization.
    for col in x.columns:
        median = x[col].median()
        if pd.isna(median):
            median = 0.0
        x[col] = x[col].fillna(median)

    x = x.fillna(0.0)

    nan_after = int(x.isna().sum().sum())
    numeric_after = x.select_dtypes(include=[np.number])
    inf_after = int(np.isinf(numeric_after.to_numpy()).sum())

    return x, nan_before, inf_before, nan_after, inf_after


def prepare_xgb(df):
    missing = [c for c in XGB_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing XGBoost features:\n"
            + "\n".join(f"  - {c}" for c in missing)
        )

    x, nb, ib, na, ia = clean_numeric(df[XGB_FEATURES])

    print(f"NaN values before cleaning : {nb:,}")
    print(f"Infinite values before cleaning : {ib:,}")
    print(f"NaN values after cleaning  : {na:,}")
    print(f"Infinite values after cleaning : {ia:,}")

    return x


def port_bucket(series):
    s = pd.to_numeric(series, errors="coerce").fillna(0)

    return np.select(
        [s <= 1023, s <= 49151, s <= 65535],
        [0.0, 1.0, 2.0],
        default=0.0,
    )


def prepare_if(df, features):
    work = df.copy()

    if "SportBucket" in features and "SportBucket" not in work.columns:
        if "Sport" not in work.columns:
            raise ValueError("IF requires Sport to create SportBucket.")
        work["SportBucket"] = port_bucket(work["Sport"])

    if "DportBucket" in features and "DportBucket" not in work.columns:
        if "Dport" not in work.columns:
            raise ValueError("IF requires Dport to create DportBucket.")
        work["DportBucket"] = port_bucket(work["Dport"])

    missing = [c for c in features if c not in work.columns]
    if missing:
        raise ValueError(
            "Missing Isolation Forest features:\n"
            + "\n".join(f"  - {c}" for c in missing)
        )

    return clean_numeric(work[features])


def load_scenario(scenario):
    path = DATA_DIR / f"scenario{scenario}" / "ctu13_features.csv"

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found:\n{path}")

    df = pd.read_csv(path)

    print(f"Loading Scenario {scenario}: {path}")
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")

    return df


def safe_auc(y, scores, kind="roc"):
    if len(np.unique(y)) < 2:
        return np.nan

    try:
        if kind == "roc":
            return float(roc_auc_score(y, scores))
        return float(average_precision_score(y, scores))
    except Exception:
        return np.nan


def metrics(y, pred, score):
    return {
        "Accuracy": float(accuracy_score(y, pred)),
        "Precision": float(precision_score(y, pred, zero_division=0)),
        "Recall": float(recall_score(y, pred, zero_division=0)),
        "F1": float(f1_score(y, pred, zero_division=0)),
        "ROC_AUC": safe_auc(y, score, "roc"),
        "PR_AUC": safe_auc(y, score, "pr"),
    }


def print_metrics(name, m):
    print()
    print(name)
    print("-" * 40)
    print(f"Accuracy : {m['Accuracy']:.4f}")
    print(f"Precision: {m['Precision']:.4f}")
    print(f"Recall   : {m['Recall']:.4f}")
    print(f"F1 Score : {m['F1']:.4f}")
    print(
        "ROC-AUC  : N/A"
        if pd.isna(m["ROC_AUC"])
        else f"ROC-AUC  : {m['ROC_AUC']:.4f}"
    )
    print(
        "PR-AUC   : N/A"
        if pd.isna(m["PR_AUC"])
        else f"PR-AUC   : {m['PR_AUC']:.4f}"
    )


# ============================================================
# START
# ============================================================

header("CTU-13 GATED HYBRID INTRUSION DETECTION")

print("Configuration:")
print("  Training scenarios : 1 - 10")
print("  Testing scenarios  : 11 - 13")
print(f"  XGBoost threshold   : {XGB_THRESHOLD:.2f}")
print(f"  Uncertainty range   : {UNCERTAINTY_LOW:.2f} - {UNCERTAINTY_HIGH:.2f}")
print(f"  IF calibration     : {IF_CALIBRATION_PERCENTILE:.0f}th benign percentile")
print(f"  IF anomaly gate    : {IF_ANOMALY_QUANTILE:.0%} calibrated percentile")
print("  Hybrid mode         : GATED")
print(f"  Random seed         : {RANDOM_SEED}")


# ============================================================
# LOAD MODELS
# ============================================================

header("LOADING MODELS")

print("Loading XGBoost...")
print(f"Path: {XGB_MODEL_PATH}")

if not XGB_MODEL_PATH.exists():
    raise FileNotFoundError(f"XGBoost model not found:\n{XGB_MODEL_PATH}")

xgb_model = xgb.XGBClassifier()
xgb_model.load_model(str(XGB_MODEL_PATH))
print("XGBoost loaded successfully.")

print()
print("Loading Isolation Forest...")
print(f"Path: {IF_MODEL_PATH}")

for path, label in [
    (IF_MODEL_PATH, "Isolation Forest model"),
    (IF_SCALER_PATH, "Isolation Forest scaler"),
    (IF_FEATURES_PATH, "Isolation Forest feature list"),
]:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found:\n{path}")

if_model = joblib.load(IF_MODEL_PATH)
if_scaler = joblib.load(IF_SCALER_PATH)

with open(IF_FEATURES_PATH, "r", encoding="utf-8") as f:
    if_features = json.load(f)

print("Isolation Forest loaded successfully.")
print(f"IF feature count: {len(if_features)}")


# ============================================================
# CALIBRATE IF ON TRAINING BENIGN TRAFFIC
# ============================================================

header("CALIBRATING ISOLATION FOREST")

benign_parts = []

for scenario in TRAIN_SCENARIOS:
    df = load_scenario(scenario)
    y = get_target(df)

    benign = df[y == 0].copy()

    # Keep calibration memory reasonable.
    if len(benign) > 15000:
        benign = benign.sample(
            n=15000,
            random_state=RANDOM_SEED,
        )

    if len(benign):
        benign_parts.append(benign)

    print(f"  Benign calibration rows: {len(benign):,}")

if not benign_parts:
    raise RuntimeError("No benign training rows available for IF calibration.")

benign_train = pd.concat(benign_parts, ignore_index=True)

print()
print(f"Total benign calibration rows: {len(benign_train):,}")

X_if_train, *_ = prepare_if(benign_train, if_features)
X_if_train_scaled = if_scaler.transform(X_if_train)

# Larger score = more anomalous.
if_train_score = -if_model.decision_function(X_if_train_scaled)

if_threshold = float(
    np.percentile(if_train_score, IF_CALIBRATION_PERCENTILE)
)

if_gate_threshold = float(
    np.percentile(if_train_score, IF_ANOMALY_QUANTILE * 100)
)

print(f"IF threshold: {if_threshold:.8f}")
print(f"IF strong-anomaly gate: {if_gate_threshold:.8f}")

del benign_parts
del benign_train
del X_if_train
del X_if_train_scaled
del if_train_score


# ============================================================
# LOAD TEST DATA
# ============================================================

header("LOADING UNSEEN TEST SCENARIOS")

test_frames = {
    scenario: load_scenario(scenario)
    for scenario in TEST_SCENARIOS
}

total_rows = sum(len(df) for df in test_frames.values())

print()
print(f"TOTAL TEST ROWS: {total_rows:,}")


# ============================================================
# TEST
# ============================================================

header("GATED HYBRID PREDICTION")

scenario_results = []
prediction_frames = []

for scenario in TEST_SCENARIOS:

    header(f"PROCESSING SCENARIO {scenario}")

    df = test_frames[scenario]
    y_true = get_target(df)

    print(f"Rows: {len(df):,}")

    # --------------------------------------------------------
    # XGBoost
    # --------------------------------------------------------

    print()
    print(f"Preparing {len(XGB_FEATURES)} XGBoost features...")

    X_xgb = prepare_xgb(df)

    print(
        f"Feature matrix: "
        f"{X_xgb.shape[0]:,} x {X_xgb.shape[1]}"
    )

    print()
    print("Running XGBoost...")

    xgb_probability = xgb_model.predict_proba(X_xgb)[:, 1]
    xgb_prediction = (
        xgb_probability >= XGB_THRESHOLD
    ).astype(np.int8)

    # --------------------------------------------------------
    # Isolation Forest
    # --------------------------------------------------------

    print("Running Isolation Forest...")

    (
        X_if,
        if_nan_before,
        if_inf_before,
        if_nan_after,
        if_inf_after,
    ) = prepare_if(df, if_features)

    print(f"IF NaN before cleaning : {if_nan_before:,}")
    print(f"IF Inf before cleaning : {if_inf_before:,}")
    print(f"IF NaN after cleaning  : {if_nan_after:,}")
    print(f"IF Inf after cleaning  : {if_inf_after:,}")

    X_if_scaled = if_scaler.transform(X_if)

    if_score = -if_model.decision_function(X_if_scaled)

    # Strong anomaly only.
    if_prediction = (
        if_score >= if_threshold
    ).astype(np.int8)

    if_strong_anomaly = (
        if_score >= if_gate_threshold
    )

    # --------------------------------------------------------
    # GATED HYBRID
    # --------------------------------------------------------

    uncertain = (
        (xgb_probability >= UNCERTAINTY_LOW)
        & (xgb_probability <= UNCERTAINTY_HIGH)
    )

    # Start exactly from XGBoost.
    hybrid_prediction = xgb_prediction.copy()

    # Only uncertain rows may be changed.
    # Only a very strong IF anomaly can promote an uncertain
    # row to attack.
    promote = uncertain & if_strong_anomaly

    hybrid_prediction[promote] = 1

    # Hybrid score is kept continuous for ranking metrics.
    # Outside uncertainty, XGB remains dominant.
    hybrid_score = xgb_probability.copy()

    # Give promoted uncertain rows a conservative score that
    # remains monotonic with the IF anomaly strength.
    if np.any(promote):
        normalized_if = np.clip(
            (if_score[promote] - if_threshold)
            / max(if_gate_threshold - if_threshold, 1e-9),
            0.0,
            1.0,
        )

        hybrid_score[promote] = np.maximum(
            hybrid_score[promote],
            XGB_THRESHOLD
            + 0.10 * normalized_if,
        )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    xgb_m = metrics(
        y_true,
        xgb_prediction,
        xgb_probability,
    )

    if_m = metrics(
        y_true,
        if_prediction,
        if_score,
    )

    hybrid_m = metrics(
        y_true,
        hybrid_prediction,
        hybrid_score,
    )

    print_metrics("XGBoost", xgb_m)
    print_metrics("Isolation Forest", if_m)
    print_metrics("GATED HYBRID", hybrid_m)

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_true,
        hybrid_prediction,
        labels=[0, 1],
    )

    tn, fp, fn, tp = cm.ravel()

    print()
    print("HYBRID CONFUSION MATRIX")
    print("-" * 40)
    print(cm)
    print()
    print(f"True Negatives : {tn:,}")
    print(f"False Positives: {fp:,}")
    print(f"False Negatives: {fn:,}")
    print(f"True Positives : {tp:,}")

    print()
    print("PREDICTION DISTRIBUTION")
    print(f"XGBoost BOTNET       : {xgb_prediction.sum():,}")
    print(f"Isolation anomalies  : {if_prediction.sum():,}")
    print(f"Strong IF anomalies  : {if_strong_anomaly.sum():,}")
    print(f"XGBoost uncertain    : {uncertain.sum():,}")
    print(f"Hybrid promotions    : {promote.sum():,}")
    print(f"Hybrid BOTNET        : {hybrid_prediction.sum():,}")

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    scenario_results.extend(
        [
            {"Scenario": scenario, "Model": "XGBoost", **xgb_m},
            {
                "Scenario": scenario,
                "Model": "IsolationForest",
                **if_m,
            },
            {"Scenario": scenario, "Model": "Hybrid", **hybrid_m},
        ]
    )

    prediction_frames.append(
        pd.DataFrame(
            {
                "Scenario": scenario,
                "Target": y_true,
                "XGBoostProbability": xgb_probability,
                "XGBoostPrediction": xgb_prediction,
                "IsolationScore": if_score,
                "IsolationPrediction": if_prediction,
                "IFStrongAnomaly": if_strong_anomaly.astype(np.int8),
                "XGBoostUncertain": uncertain.astype(np.int8),
                "HybridPromotion": promote.astype(np.int8),
                "HybridScore": hybrid_score,
                "HybridPrediction": hybrid_prediction,
            }
        )
    )


# ============================================================
# OVERALL
# ============================================================

predictions = pd.concat(
    prediction_frames,
    ignore_index=True,
)

y_all = predictions["Target"].to_numpy()

xgb_prob_all = predictions["XGBoostProbability"].to_numpy()
xgb_pred_all = predictions["XGBoostPrediction"].to_numpy()

if_score_all = predictions["IsolationScore"].to_numpy()
if_pred_all = predictions["IsolationPrediction"].to_numpy()

hybrid_score_all = predictions["HybridScore"].to_numpy()
hybrid_pred_all = predictions["HybridPrediction"].to_numpy()

overall_xgb = metrics(
    y_all,
    xgb_pred_all,
    xgb_prob_all,
)

overall_if = metrics(
    y_all,
    if_pred_all,
    if_score_all,
)

overall_hybrid = metrics(
    y_all,
    hybrid_pred_all,
    hybrid_score_all,
)


header("OVERALL HYBRID RESULTS")

print_metrics("XGBoost", overall_xgb)
print_metrics("Isolation Forest", overall_if)
print_metrics("GATED HYBRID XGBOOST + ISOLATION FOREST", overall_hybrid)


# ============================================================
# OVERALL CONFUSION MATRICES
# ============================================================

header("OVERALL CONFUSION MATRICES")

print("XGBoost:")
print(
    confusion_matrix(
        y_all,
        xgb_pred_all,
        labels=[0, 1],
    )
)

print()
print("Isolation Forest:")
print(
    confusion_matrix(
        y_all,
        if_pred_all,
        labels=[0, 1],
    )
)

print()
print("Gated Hybrid:")
print(
    confusion_matrix(
        y_all,
        hybrid_pred_all,
        labels=[0, 1],
    )
)


# ============================================================
# IMPROVEMENT
# ============================================================

recall_change = (
    overall_hybrid["Recall"]
    - overall_xgb["Recall"]
)

precision_change = (
    overall_hybrid["Precision"]
    - overall_xgb["Precision"]
)

f1_change = (
    overall_hybrid["F1"]
    - overall_xgb["F1"]
)

header("HYBRID IMPROVEMENT")

print(f"Recall improvement   : {recall_change:+.4f}")
print(f"Precision change     : {precision_change:+.4f}")
print(f"F1 improvement       : {f1_change:+.4f}")


# ============================================================
# PER-SCENARIO
# ============================================================

header("PER-SCENARIO HYBRID SUMMARY")

scenario_df = pd.DataFrame(scenario_results)
print(scenario_df.to_string(index=False))


# ============================================================
# SAVE
# ============================================================

header("SAVING RESULTS")

MODEL_DIR.mkdir(parents=True, exist_ok=True)

overall_df = pd.DataFrame(
    [
        {"Model": "XGBoost", **overall_xgb},
        {"Model": "IsolationForest", **overall_if},
        {"Model": "Hybrid", **overall_hybrid},
    ]
)

overall_path = MODEL_DIR / "hybrid_benchmark.csv"
scenario_path = MODEL_DIR / "hybrid_benchmark_per_scenario.csv"
prediction_path = MODEL_DIR / "hybrid_predictions.csv"
config_path = MODEL_DIR / "hybrid_config.json"

overall_df.to_csv(overall_path, index=False)
scenario_df.to_csv(scenario_path, index=False)
predictions.to_csv(prediction_path, index=False)

config = {
    "training_scenarios": "1-10",
    "testing_scenarios": "11-13",
    "xgboost_model": str(XGB_MODEL_PATH),
    "isolation_forest_model": str(IF_MODEL_PATH),
    "isolation_forest_scaler": str(IF_SCALER_PATH),
    "isolation_forest_features": str(IF_FEATURES_PATH),
    "xgboost_threshold": XGB_THRESHOLD,
    "uncertainty_low": UNCERTAINTY_LOW,
    "uncertainty_high": UNCERTAINTY_HIGH,
    "if_calibration_percentile": IF_CALIBRATION_PERCENTILE,
    "if_anomaly_quantile": IF_ANOMALY_QUANTILE,
    "if_threshold": if_threshold,
    "if_strong_anomaly_gate": if_gate_threshold,
    "hybrid_mode": "gated",
    "random_seed": RANDOM_SEED,
    "test_rows": int(len(predictions)),
    "xgboost_metrics": overall_xgb,
    "isolation_metrics": overall_if,
    "hybrid_metrics": overall_hybrid,
    "improvement": {
        "recall": float(recall_change),
        "precision": float(precision_change),
        "f1": float(f1_change),
    },
}

with open(config_path, "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2, allow_nan=False)

print("Overall results saved to:")
print(f"  {overall_path}")

print("Per-scenario results saved to:")
print(f"  {scenario_path}")

print("Detailed predictions saved to:")
print(f"  {prediction_path}")

print("Configuration saved to:")
print(f"  {config_path}")


# ============================================================
# FINAL
# ============================================================

header("GATED HYBRID DETECTOR COMPLETE")

print("Training:")
print("  Scenario 1 - Scenario 10")

print()
print("Testing:")
print("  Scenario 11 - Scenario 13")

print()
print(f"Test rows: {len(predictions):,}")

print()
print("Models:")
print("  XGBoost")
print("  Isolation Forest")
print("  Gated Hybrid XGBoost + Isolation Forest")

print()
print(f"XGBoost threshold   : {XGB_THRESHOLD:.2f}")
print(f"IF threshold        : {if_threshold:.8f}")
print(f"IF strong gate      : {if_gate_threshold:.8f}")
print(
    f"Uncertainty range   : "
    f"{UNCERTAINTY_LOW:.2f} - {UNCERTAINTY_HIGH:.2f}"
)

print()
print(f"XGBoost F1          : {overall_xgb['F1']:.4f}")
print(f"Hybrid F1           : {overall_hybrid['F1']:.4f}")
print(f"XGBoost Recall      : {overall_xgb['Recall']:.4f}")
print(f"Hybrid Recall       : {overall_hybrid['Recall']:.4f}")

print()
print("DONE")
print("=" * 70)
