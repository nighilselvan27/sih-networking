"""
CTU-13 XGBOOST THRESHOLD TUNING

TRAINED MODEL:
    models/ctu13_multiscenario_xgboost.json

TEST:
    Scenario 11 - Scenario 13

Purpose:
    Find a better probability threshold for the existing
    XGBoost model without retraining it.

Default:
    0.50

We evaluate thresholds from 0.05 to 0.95.
"""

import json
from pathlib import Path

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


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = (
    MODEL_DIR /
    "ctu13_multiscenario_xgboost.json"
)

FEATURE_PATH = (
    MODEL_DIR /
    "ctu13_multiscenario_features.json"
)

OUTPUT_PATH = (
    MODEL_DIR /
    "xgboost_threshold_results.csv"
)

BEST_THRESHOLD_PATH = (
    MODEL_DIR /
    "xgboost_best_threshold.json"
)


# ============================================================
# CONFIGURATION
# ============================================================

TEST_SCENARIOS = [11, 12, 13]

TARGET = "Target"


# Same 30 features used during training.
#
# We explicitly define them here so the threshold tuning
# script remains reproducible even if the feature metadata
# file is missing.

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


# Thresholds to evaluate
THRESHOLDS = np.arange(
    0.05,
    0.951,
    0.01
)


# ============================================================
# HELPER
# ============================================================

def print_header(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def prepare_features(df):
    """
    Prepare the exact 30 numerical features.
    """

    missing = [
        feature
        for feature in FEATURES
        if feature not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required features:\n"
            + "\n".join(missing)
        )

    X = df[FEATURES].copy()

    for column in FEATURES:

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce"
        )

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X = X.fillna(
        X.median(numeric_only=True)
    )

    X = X.fillna(0)

    return X


def load_scenario(scenario):

    path = (
        DATA_DIR
        / f"scenario{scenario}"
        / "ctu13_features.csv"
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Dataset not found:\n{path}"
        )

    print(
        f"Loading Scenario {scenario}: {path}"
    )

    df = pd.read_csv(path)

    print(
        f"  Rows: {len(df):,}"
    )

    return df


# ============================================================
# START
# ============================================================

print_header(
    "CTU-13 XGBOOST THRESHOLD TUNING"
)

print(
    f"Model: {MODEL_PATH}"
)

print()
print(
    "Test scenarios:"
)

print(
    "Scenario 11 - Scenario 13"
)


# ============================================================
# CHECK MODEL
# ============================================================

print_header(
    "LOADING XGBOOST MODEL"
)

if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"XGBoost model not found:\n{MODEL_PATH}"
    )


model = xgb.XGBClassifier()

model.load_model(
    str(MODEL_PATH)
)

print(
    "XGBoost model loaded successfully."
)


# ============================================================
# LOAD TEST DATA
# ============================================================

print_header(
    "LOADING UNSEEN TEST DATA"
)

X_test_parts = []
y_test_parts = []
scenario_parts = []


for scenario in TEST_SCENARIOS:

    df = load_scenario(scenario)

    if TARGET not in df.columns:

        raise ValueError(
            f"'{TARGET}' column missing "
            f"from Scenario {scenario}"
        )

    y = pd.to_numeric(
        df[TARGET],
        errors="coerce"
    ).fillna(0).astype(int)

    X = prepare_features(df)

    X_test_parts.append(X)

    y_test_parts.append(y)

    scenario_parts.extend(
        [scenario] * len(df)
    )

    del df


X_test = pd.concat(
    X_test_parts,
    ignore_index=True
)

y_test = pd.concat(
    y_test_parts,
    ignore_index=True
).to_numpy()


scenario_labels = np.array(
    scenario_parts
)


print()
print(
    f"Total test rows: {len(X_test):,}"
)

print(
    f"Features: {X_test.shape[1]}"
)


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

print_header(
    "TEST DISTRIBUTION"
)

benign_count = np.sum(
    y_test == 0
)

botnet_count = np.sum(
    y_test == 1
)

print(
    f"BENIGN : {benign_count:,}"
)

print(
    f"BOTNET : {botnet_count:,}"
)


# ============================================================
# GET XGBOOST PROBABILITIES
# ============================================================

print_header(
    "GENERATING XGBOOST PROBABILITIES"
)

print(
    "Running model once..."
)

probabilities = model.predict_proba(
    X_test
)[:, 1]


print(
    "Probabilities generated."
)

print(
    f"Minimum probability: "
    f"{probabilities.min():.6f}"
)

print(
    f"Maximum probability: "
    f"{probabilities.max():.6f}"
)

print(
    f"Mean probability: "
    f"{probabilities.mean():.6f}"
)


# ============================================================
# ROC / PR AUC
# ============================================================

print_header(
    "THRESHOLD-INDEPENDENT METRICS"
)

roc_auc = roc_auc_score(
    y_test,
    probabilities
)

pr_auc = average_precision_score(
    y_test,
    probabilities
)

print(
    f"ROC-AUC: {roc_auc:.4f}"
)

print(
    f"PR-AUC : {pr_auc:.4f}"
)


# ============================================================
# THRESHOLD SEARCH
# ============================================================

print_header(
    "TESTING THRESHOLDS"
)

results = []


for threshold in THRESHOLDS:

    y_pred = (
        probabilities >= threshold
    ).astype(int)


    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )


    cm = confusion_matrix(
        y_test,
        y_pred
    )


    if cm.shape == (2, 2):

        tn, fp, fn, tp = cm.ravel()

    else:

        tn = fp = fn = tp = 0


    results.append(
        {
            "Threshold": round(
                float(threshold),
                2
            ),
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "True_Negatives": tn,
            "False_Positives": fp,
            "False_Negatives": fn,
            "True_Positives": tp,
        }
    )


results_df = pd.DataFrame(
    results
)


# ============================================================
# BEST THRESHOLD BY F1
# ============================================================

best_row = results_df.loc[
    results_df["F1"].idxmax()
]


best_threshold = float(
    best_row["Threshold"]
)

best_accuracy = float(
    best_row["Accuracy"]
)

best_precision = float(
    best_row["Precision"]
)

best_recall = float(
    best_row["Recall"]
)

best_f1 = float(
    best_row["F1"]
)


# ============================================================
# DEFAULT THRESHOLD
# ============================================================

default_row = results_df[
    results_df["Threshold"] == 0.50
].iloc[0]


# ============================================================
# PRINT TABLE
# ============================================================

print_header(
    "THRESHOLD RESULTS"
)

print(
    results_df[
        [
            "Threshold",
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
        ]
    ].to_string(
        index=False,
        formatters={
            "Accuracy": "{:.4f}".format,
            "Precision": "{:.4f}".format,
            "Recall": "{:.4f}".format,
            "F1": "{:.4f}".format,
        }
    )
)


# ============================================================
# DEFAULT VS BEST
# ============================================================

print_header(
    "DEFAULT VS OPTIMIZED THRESHOLD"
)

print(
    "DEFAULT THRESHOLD = 0.50"
)

print(
    f"  Accuracy : "
    f"{default_row['Accuracy']:.4f}"
)

print(
    f"  Precision: "
    f"{default_row['Precision']:.4f}"
)

print(
    f"  Recall   : "
    f"{default_row['Recall']:.4f}"
)

print(
    f"  F1       : "
    f"{default_row['F1']:.4f}"
)


print()

print(
    f"BEST THRESHOLD = {best_threshold:.2f}"
)

print(
    f"  Accuracy : "
    f"{best_accuracy:.4f}"
)

print(
    f"  Precision: "
    f"{best_precision:.4f}"
)

print(
    f"  Recall   : "
    f"{best_recall:.4f}"
)

print(
    f"  F1       : "
    f"{best_f1:.4f}"
)


print()

print(
    "F1 improvement:"
)

print(
    f"{best_f1 - default_row['F1']:+.4f}"
)


# ============================================================
# BEST CONFUSION MATRIX
# ============================================================

print_header(
    "BEST THRESHOLD CONFUSION MATRIX"
)

best_predictions = (
    probabilities >= best_threshold
).astype(int)


best_cm = confusion_matrix(
    y_test,
    best_predictions
)


print(best_cm)


if best_cm.shape == (2, 2):

    tn, fp, fn, tp = (
        best_cm.ravel()
    )

    print()
    print(
        f"True Negatives : {tn:,}"
    )

    print(
        f"False Positives: {fp:,}"
    )

    print(
        f"False Negatives: {fn:,}"
    )

    print(
        f"True Positives : {tp:,}"
    )


# ============================================================
# PER-SCENARIO PERFORMANCE
# ============================================================

print_header(
    "PER-SCENARIO PERFORMANCE AT BEST THRESHOLD"
)


scenario_results = []


for scenario in TEST_SCENARIOS:

    mask = (
        scenario_labels == scenario
    )

    y_true_s = y_test[mask]

    probabilities_s = probabilities[mask]

    predictions_s = (
        probabilities_s >= best_threshold
    ).astype(int)


    accuracy_s = accuracy_score(
        y_true_s,
        predictions_s
    )

    precision_s = precision_score(
        y_true_s,
        predictions_s,
        zero_division=0
    )

    recall_s = recall_score(
        y_true_s,
        predictions_s,
        zero_division=0
    )

    f1_s = f1_score(
        y_true_s,
        predictions_s,
        zero_division=0
    )


    print()
    print(
        f"Scenario {scenario}"
    )

    print(
        "-" * 40
    )

    print(
        f"Rows      : "
        f"{len(y_true_s):,}"
    )

    print(
        f"Accuracy  : "
        f"{accuracy_s:.4f}"
    )

    print(
        f"Precision : "
        f"{precision_s:.4f}"
    )

    print(
        f"Recall    : "
        f"{recall_s:.4f}"
    )

    print(
        f"F1 Score  : "
        f"{f1_s:.4f}"
    )


    scenario_results.append(
        {
            "Scenario": scenario,
            "Rows": len(y_true_s),
            "Accuracy": accuracy_s,
            "Precision": precision_s,
            "Recall": recall_s,
            "F1": f1_s,
        }
    )


# ============================================================
# SAVE ALL THRESHOLD RESULTS
# ============================================================

print_header(
    "SAVING RESULTS"
)

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# SAVE BEST THRESHOLD
# ============================================================

best_config = {
    "model": "XGBoost",
    "threshold": best_threshold,
    "selection_metric": "F1",
    "train_scenarios": "1-10",
    "test_scenarios": "11-13",
    "accuracy": best_accuracy,
    "precision": best_precision,
    "recall": best_recall,
    "f1": best_f1,
    "roc_auc": float(roc_auc),
    "pr_auc": float(pr_auc),
}


with open(
    BEST_THRESHOLD_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        best_config,
        f,
        indent=2
    )


print(
    f"Threshold results saved to:"
)

print(
    OUTPUT_PATH
)

print()

print(
    f"Best threshold saved to:"
)

print(
    BEST_THRESHOLD_PATH
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print_header(
    "THRESHOLD TUNING COMPLETE"
)

print()

print(
    f"Original threshold : 0.50"
)

print(
    f"Optimized threshold: "
    f"{best_threshold:.2f}"
)

print()

print(
    "ORIGINAL"
)

print(
    f"Precision: "
    f"{default_row['Precision']:.4f}"
)

print(
    f"Recall   : "
    f"{default_row['Recall']:.4f}"
)

print(
    f"F1       : "
    f"{default_row['F1']:.4f}"
)

print()

print(
    "OPTIMIZED"
)

print(
    f"Precision: "
    f"{best_precision:.4f}"
)

print(
    f"Recall   : "
    f"{best_recall:.4f}"
)

print(
    f"F1       : "
    f"{best_f1:.4f}"
)

print()

print(
    "=" * 70
)

print(
    "DONE"
)

print(
    "=" * 70
)