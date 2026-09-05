"""
CTU-13 SYNTHETIC ATTACK DETECTION BENCHMARK

Tests the trained XGBoost model against synthetic flow-level traffic:

    - BENIGN
    - SYN_FLOOD
    - UDP_FLOOD
    - PORT_SCAN

Model:
    models/ctu13_multiscenario_xgboost.json

Input:
    data/synthetic/synthetic_attacks.csv

XGBoost threshold:
    0.40

Outputs:
    models/synthetic_benchmark.csv
    models/synthetic_benchmark_per_attack.csv
    models/synthetic_predictions.csv
    models/synthetic_config.json

IMPORTANT:
This is a synthetic behavioral stress test.
It is NOT a replacement for validation on real packet captures.
"""


import os
import json
import numpy as np
import pandas as pd

from xgboost import XGBClassifier

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


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/ctu13_multiscenario_xgboost.json"

INPUT_PATH = "data/synthetic/synthetic_attacks.csv"

OUTPUT_DIR = "models"

XGBOOST_THRESHOLD = 0.40

RANDOM_SEED = 42


# ============================================================
# EXACT 30 FEATURES USED BY YOUR CTU-13 MODEL
# ============================================================

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


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def calculate_metrics(y_true, y_pred, y_prob):
    """
    Calculate classification metrics.
    """

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    try:
        roc_auc = roc_auc_score(
            y_true,
            y_prob
        )
    except ValueError:
        roc_auc = np.nan

    try:
        pr_auc = average_precision_score(
            y_true,
            y_prob
        )
    except ValueError:
        pr_auc = np.nan

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc,
    }


def print_metrics(metrics):
    """
    Pretty-print metrics.
    """

    print(
        f"Accuracy : {metrics['Accuracy']:.4f}"
    )

    print(
        f"Precision: {metrics['Precision']:.4f}"
    )

    print(
        f"Recall   : {metrics['Recall']:.4f}"
    )

    print(
        f"F1 Score : {metrics['F1']:.4f}"
    )

    if pd.notna(metrics["ROC_AUC"]):
        print(
            f"ROC-AUC  : {metrics['ROC_AUC']:.4f}"
        )
    else:
        print(
            "ROC-AUC  : N/A"
        )

    if pd.notna(metrics["PR_AUC"]):
        print(
            f"PR-AUC   : {metrics['PR_AUC']:.4f}"
        )
    else:
        print(
            "PR-AUC   : N/A"
        )


def print_confusion_matrix(y_true, y_pred):
    """
    Print confusion matrix with named values.
    """

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    print()
    print("CONFUSION MATRIX")
    print("----------------------------------------")

    print(cm)

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

    return tn, fp, fn, tp


# ============================================================
# HEADER
# ============================================================

print()
print("=" * 70)
print("CTU-13 SYNTHETIC ATTACK DETECTION BENCHMARK")
print("=" * 70)

print()
print("Configuration:")
print("  Model                 : XGBoost")
print("  Synthetic dataset     : synthetic_attacks.csv")
print(f"  XGBoost threshold     : {XGBOOST_THRESHOLD}")
print(f"  Random seed           : {RANDOM_SEED}")


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("=" * 70)
print("LOADING XGBOOST MODEL")
print("=" * 70)

print(
    f"Path: {MODEL_PATH}"
)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"XGBoost model not found: {MODEL_PATH}"
    )


model = XGBClassifier()

model.load_model(
    MODEL_PATH
)

print(
    "XGBoost loaded successfully."
)


# ============================================================
# LOAD SYNTHETIC DATASET
# ============================================================

print()
print("=" * 70)
print("LOADING SYNTHETIC DATASET")
print("=" * 70)

print(
    f"Path: {INPUT_PATH}"
)

if not os.path.exists(INPUT_PATH):
    raise FileNotFoundError(
        f"Synthetic dataset not found: {INPUT_PATH}"
    )


df = pd.read_csv(
    INPUT_PATH
)

print()
print(
    f"Rows    : {len(df):,}"
)

print(
    f"Columns : {len(df.columns)}"
)


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

print()
print("=" * 70)
print("VALIDATING DATASET")
print("=" * 70)


required_columns = (
    FEATURES +
    [
        "Target",
        "AttackType",
    ]
)


missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]


if missing_columns:

    print()
    print("ERROR: Missing columns:")

    for column in missing_columns:
        print(
            f"  - {column}"
        )

    raise ValueError(
        "Synthetic dataset does not contain the required columns."
    )


print(
    "All required columns are present."
)


# ============================================================
# ATTACK DISTRIBUTION
# ============================================================

print()
print("=" * 70)
print("SYNTHETIC TRAFFIC DISTRIBUTION")
print("=" * 70)

print()

print(
    df["AttackType"]
    .value_counts()
    .to_string()
)

print()

print(
    "Target distribution:"
)

print(
    df["Target"]
    .value_counts()
    .sort_index()
    .to_string()
)


# ============================================================
# PREPARE FEATURES
# ============================================================

print()
print("=" * 70)
print("PREPARING FEATURES")
print("=" * 70)

print(
    f"Using {len(FEATURES)} model features."
)

X = df[FEATURES].copy()

y = df["Target"].astype(
    int
)


# ============================================================
# NUMERIC CONVERSION
# ============================================================

print()
print("Converting features to numeric...")

for feature in FEATURES:

    X[feature] = pd.to_numeric(
        X[feature],
        errors="coerce"
    )


# ============================================================
# CHECK NaN
# ============================================================

nan_count = X.isna().sum().sum()

print(
    f"NaN values found: {nan_count:,}"
)

if nan_count > 0:

    print(
        "Replacing NaN values with 0."
    )

    X = X.fillna(0)


# ============================================================
# CHECK INFINITE VALUES
# ============================================================

inf_count = np.isinf(
    X.to_numpy()
).sum()

print(
    f"Infinite values found: {inf_count:,}"
)

if inf_count > 0:

    print(
        "Replacing infinite values with 0."
    )

    X = X.replace(
        [np.inf, -np.inf],
        0
    )


print()
print(
    f"Feature matrix: {X.shape[0]:,} x {X.shape[1]}"
)


# ============================================================
# PREDICTION
# ============================================================

print()
print("=" * 70)
print("RUNNING XGBOOST")
print("=" * 70)

print()
print(
    "Generating probability scores..."
)


probabilities = model.predict_proba(
    X
)[:, 1]


print(
    "Probability generation complete."
)


# ============================================================
# THRESHOLD
# ============================================================

print()
print(
    f"Applying threshold: {XGBOOST_THRESHOLD}"
)


predictions = (
    probabilities >= XGBOOST_THRESHOLD
).astype(
    int
)


# ============================================================
# OVERALL RESULTS
# ============================================================

print()
print("=" * 70)
print("OVERALL SYNTHETIC RESULTS")
print("=" * 70)

overall_metrics = calculate_metrics(
    y,
    predictions,
    probabilities
)

print()

print_metrics(
    overall_metrics
)

tn, fp, fn, tp = print_confusion_matrix(
    y,
    predictions
)


# ============================================================
# DETECTION RATES
# ============================================================

print()
print("=" * 70)
print("DETECTION STATISTICS")
print("=" * 70)

attack_count = int(
    (y == 1).sum()
)

benign_count = int(
    (y == 0).sum()
)

detected_attacks = int(
    ((y == 1) &
     (predictions == 1)).sum()
)

missed_attacks = int(
    ((y == 1) &
     (predictions == 0)).sum()
)

false_alarms = int(
    ((y == 0) &
     (predictions == 1)).sum()
)


if attack_count > 0:

    attack_detection_rate = (
        detected_attacks /
        attack_count
    )

else:

    attack_detection_rate = 0


if benign_count > 0:

    false_positive_rate = (
        false_alarms /
        benign_count
    )

else:

    false_positive_rate = 0


print()

print(
    f"Total attacks       : {attack_count:,}"
)

print(
    f"Detected attacks    : {detected_attacks:,}"
)

print(
    f"Missed attacks      : {missed_attacks:,}"
)

print(
    f"Attack detection rate: "
    f"{attack_detection_rate:.4f}"
)

print(
    f"False alarms        : {false_alarms:,}"
)

print(
    f"False-positive rate : "
    f"{false_positive_rate:.4f}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print()
print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print()

print(
    classification_report(
        y,
        predictions,
        labels=[0, 1],
        target_names=[
            "BENIGN",
            "ATTACK",
        ],
        zero_division=0
    )
)


# ============================================================
# PER-ATTACK RESULTS
# ============================================================

print()
print("=" * 70)
print("PER-ATTACK RESULTS")
print("=" * 70)


attack_types = [
    "SYN_FLOOD",
    "UDP_FLOOD",
    "PORT_SCAN",
]


per_attack_results = []


for attack_type in attack_types:

    print()
    print(
        attack_type
    )

    print(
        "-" * 40
    )

    mask = (
        df["AttackType"] ==
        attack_type
    )

    attack_df = df.loc[
        mask
    ]

    attack_y = y.loc[
        mask
    ]

    attack_predictions = predictions[
        mask.to_numpy()
    ]

    attack_probabilities = probabilities[
        mask.to_numpy()
    ]


    metrics = calculate_metrics(
        attack_y,
        attack_predictions,
        attack_probabilities
    )


    print(
        f"Rows     : {len(attack_df):,}"
    )

    print_metrics(
        metrics
    )


    detected = int(
        (
            attack_predictions == 1
        ).sum()
    )

    missed = int(
        (
            attack_predictions == 0
        ).sum()
    )


    detection_rate = (
        detected /
        len(attack_df)
        if len(attack_df) > 0
        else 0
    )


    print()

    print(
        f"Detected : {detected:,}"
    )

    print(
        f"Missed   : {missed:,}"
    )

    print(
        f"Detection Rate: "
        f"{detection_rate:.4f}"
    )


    # Confusion matrix for attack vs benign
    # Since this subset contains only attacks,
    # we record TP/FN directly.

    per_attack_results.append(
        {
            "AttackType": attack_type,
            "Rows": len(attack_df),
            "Detected": detected,
            "Missed": missed,
            "DetectionRate": detection_rate,
            "Accuracy": metrics["Accuracy"],
            "Precision": metrics["Precision"],
            "Recall": metrics["Recall"],
            "F1": metrics["F1"],
            "ROC_AUC": metrics["ROC_AUC"],
            "PR_AUC": metrics["PR_AUC"],
            "MeanProbability": float(
                np.mean(
                    attack_probabilities
                )
            ),
            "MedianProbability": float(
                np.median(
                    attack_probabilities
                )
            ),
        }
    )


# ============================================================
# BENIGN RESULTS
# ============================================================

print()
print("=" * 70)
print("BENIGN TRAFFIC RESULTS")
print("=" * 70)

benign_mask = (
    df["AttackType"] ==
    "BENIGN"
)

benign_predictions = predictions[
    benign_mask.to_numpy()
]

benign_probabilities = probabilities[
    benign_mask.to_numpy()
]


benign_false_alarms = int(
    (
        benign_predictions == 1
    ).sum()
)


benign_correct = int(
    (
        benign_predictions == 0
    ).sum()
)


benign_fpr = (
    benign_false_alarms /
    len(benign_predictions)
    if len(benign_predictions) > 0
    else 0
)


print()

print(
    f"Benign rows       : "
    f"{len(benign_predictions):,}"
)

print(
    f"Correctly benign  : "
    f"{benign_correct:,}"
)

print(
    f"False alarms      : "
    f"{benign_false_alarms:,}"
)

print(
    f"False-positive rate: "
    f"{benign_fpr:.4f}"
)

print(
    f"Mean attack score : "
    f"{np.mean(benign_probabilities):.6f}"
)


# ============================================================
# PREDICTION DISTRIBUTION
# ============================================================

print()
print("=" * 70)
print("PREDICTION DISTRIBUTION")
print("=" * 70)

prediction_distribution = pd.Series(
    predictions
).value_counts(
    sort=False
)


print()

print(
    f"Predicted BENIGN : "
    f"{prediction_distribution.get(0, 0):,}"
)

print(
    f"Predicted ATTACK : "
    f"{prediction_distribution.get(1, 0):,}"
)


# ============================================================
# ATTACK-TYPE DETECTION SUMMARY
# ============================================================

print()
print("=" * 70)
print("ATTACK-TYPE DETECTION SUMMARY")
print("=" * 70)

for result in per_attack_results:

    print()

    print(
        f"{result['AttackType']:<12} "
        f"{result['Detected']:>7,} / "
        f"{result['Rows']:>7,} "
        f"("
        f"{result['DetectionRate'] * 100:.2f}%"
        f")"
    )


# ============================================================
# SAVE DETAILED PREDICTIONS
# ============================================================

print()
print("=" * 70)
print("SAVING RESULTS")
print("=" * 70)


prediction_output = df[
    [
        "SyntheticID",
        "AttackType",
        "Target",
    ]
].copy()


prediction_output[
    "XGBoostProbability"
] = probabilities


prediction_output[
    "XGBoostPrediction"
] = predictions


prediction_output[
    "Correct"
] = (
    prediction_output["Target"] ==
    prediction_output["XGBoostPrediction"]
)


prediction_output[
    "DetectionStatus"
] = np.where(
    prediction_output["Target"] == 1,
    np.where(
        prediction_output["XGBoostPrediction"] == 1,
        "DETECTED",
        "MISSED"
    ),
    np.where(
        prediction_output["XGBoostPrediction"] == 1,
        "FALSE_POSITIVE",
        "BENIGN"
    )
)


predictions_path = os.path.join(
    OUTPUT_DIR,
    "synthetic_predictions.csv"
)


prediction_output.to_csv(
    predictions_path,
    index=False
)


print(
    f"Detailed predictions saved to:"
)

print(
    f"  {predictions_path}"
)


# ============================================================
# SAVE OVERALL BENCHMARK
# ============================================================

overall_output = pd.DataFrame(
    [
        {
            "Model": "XGBoost",
            "Threshold": XGBOOST_THRESHOLD,
            "Rows": len(df),
            "Accuracy": overall_metrics["Accuracy"],
            "Precision": overall_metrics["Precision"],
            "Recall": overall_metrics["Recall"],
            "F1": overall_metrics["F1"],
            "ROC_AUC": overall_metrics["ROC_AUC"],
            "PR_AUC": overall_metrics["PR_AUC"],
            "TrueNegatives": tn,
            "FalsePositives": fp,
            "FalseNegatives": fn,
            "TruePositives": tp,
            "AttackDetectionRate": attack_detection_rate,
            "FalsePositiveRate": false_positive_rate,
        }
    ]
)


overall_path = os.path.join(
    OUTPUT_DIR,
    "synthetic_benchmark.csv"
)


overall_output.to_csv(
    overall_path,
    index=False
)


print()

print(
    "Overall benchmark saved to:"
)

print(
    f"  {overall_path}"
)


# ============================================================
# SAVE PER-ATTACK BENCHMARK
# ============================================================

per_attack_df = pd.DataFrame(
    per_attack_results
)


per_attack_path = os.path.join(
    OUTPUT_DIR,
    "synthetic_benchmark_per_attack.csv"
)


per_attack_df.to_csv(
    per_attack_path,
    index=False
)


print()

print(
    "Per-attack benchmark saved to:"
)

print(
    f"  {per_attack_path}"
)


# ============================================================
# SAVE CONFIGURATION
# ============================================================

config = {
    "model": "XGBoost",
    "model_path": MODEL_PATH,
    "input_path": INPUT_PATH,
    "threshold": XGBOOST_THRESHOLD,
    "feature_count": len(FEATURES),
    "features": FEATURES,
    "synthetic_attack_types": [
        "SYN_FLOOD",
        "UDP_FLOOD",
        "PORT_SCAN",
    ],
    "benign_included": True,
    "total_rows": int(len(df)),
    "random_seed": RANDOM_SEED,
    "evaluation_type": (
        "Synthetic behavioral stress test"
    ),
}


config_path = os.path.join(
    OUTPUT_DIR,
    "synthetic_config.json"
)


with open(
    config_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        config,
        file,
        indent=2
    )


print()

print(
    "Configuration saved to:"
)

print(
    f"  {config_path}"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("SYNTHETIC BENCHMARK COMPLETE")
print("=" * 70)

print()

print(
    "Dataset:"
)

print(
    f"  Total rows       : {len(df):,}"
)

print(
    f"  Benign rows      : {benign_count:,}"
)

print(
    f"  Attack rows      : {attack_count:,}"
)

print()

print(
    "Model:"
)

print(
    "  XGBoost"
)

print(
    f"  Threshold: {XGBOOST_THRESHOLD}"
)

print()

print(
    "Overall performance:"
)

print(
    f"  Accuracy : {overall_metrics['Accuracy']:.4f}"
)

print(
    f"  Precision: {overall_metrics['Precision']:.4f}"
)

print(
    f"  Recall   : {overall_metrics['Recall']:.4f}"
)

print(
    f"  F1       : {overall_metrics['F1']:.4f}"
)

print(
    f"  ROC-AUC  : {overall_metrics['ROC_AUC']:.4f}"
)

print(
    f"  PR-AUC   : {overall_metrics['PR_AUC']:.4f}"
)

print()

print(
    "Detection rate:"
)

print(
    f"  {attack_detection_rate * 100:.2f}%"
)

print()

print(
    "False-positive rate:"
)

print(
    f"  {false_positive_rate * 100:.4f}%"
)

print()

print(
    "Output files:"
)

print(
    f"  {overall_path}"
)

print(
    f"  {per_attack_path}"
)

print(
    f"  {predictions_path}"
)

print(
    f"  {config_path}"
)

print()

print("=" * 70)
print("DONE")
print("=" * 70)