"""
======================================================================
CTU-13 XGBOOST THRESHOLD ANALYSIS
======================================================================

Purpose:
    Evaluate different XGBoost decision thresholds on the synthetic
    CTU-13-calibrated attack dataset.

Input:
    data/synthetic/synthetic_attacks.csv
    models/ctu13_multiscenario_xgboost.json

Output:
    models/threshold_analysis.csv
    models/threshold_recommendation.json

Attacks:
    SYN_FLOOD
    UDP_FLOOD
    PORT_SCAN

Author:
    CTU-13 IDS Project
======================================================================
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)

warnings.filterwarnings("ignore")


# ======================================================================
# CONFIGURATION
# ======================================================================

MODEL_PATH = Path("models/ctu13_multiscenario_xgboost.json")

DATA_PATH = Path(
    "data/synthetic/synthetic_attacks.csv"
)

OUTPUT_DIR = Path("models")

RESULTS_PATH = OUTPUT_DIR / "threshold_analysis.csv"

RECOMMENDATION_PATH = (
    OUTPUT_DIR / "threshold_recommendation.json"
)

RANDOM_SEED = 42

# Thresholds to evaluate
THRESHOLDS = np.round(
    np.arange(0.05, 0.91, 0.05),
    2
)

# Model features used during CTU-13 training
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
# DISPLAY HELPERS
# ======================================================================

def print_header(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def print_section(title):
    print()
    print("-" * 70)
    print(title)
    print("-" * 70)


# ======================================================================
# LOAD MODEL
# ======================================================================

def load_model():

    print_header("LOADING XGBOOST MODEL")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"XGBoost model not found:\n{MODEL_PATH}"
        )

    print(f"Path: {MODEL_PATH}")

    model = XGBClassifier()

    model.load_model(str(MODEL_PATH))

    print("XGBoost loaded successfully.")

    return model


# ======================================================================
# LOAD DATA
# ======================================================================

def load_dataset():

    print_header("LOADING SYNTHETIC DATASET")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Synthetic dataset not found:\n{DATA_PATH}"
        )

    print(f"Path: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    print()
    print(f"Rows    : {len(df):,}")
    print(f"Columns : {len(df.columns)}")

    return df


# ======================================================================
# VALIDATE DATASET
# ======================================================================

def validate_dataset(df):

    print_header("VALIDATING DATASET")

    required_columns = FEATURES + [
        "Target",
        "AttackType",
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:

        print("ERROR: Missing columns:")

        for col in missing:
            print(f"  - {col}")

        raise ValueError(
            "Synthetic dataset does not contain all required columns."
        )

    print("All required columns are present.")

    print_section("ATTACK DISTRIBUTION")

    print(df["AttackType"].value_counts())

    print_section("TARGET DISTRIBUTION")

    print(df["Target"].value_counts().sort_index())


# ======================================================================
# PREPARE FEATURES
# ======================================================================

def prepare_features(df):

    print_header("PREPARING FEATURES")

    print(f"Using {len(FEATURES)} model features.")

    X = df[FEATURES].copy()

    print()
    print("Converting features to numeric...")

    for column in FEATURES:

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce"
        )

    nan_count = int(X.isna().sum().sum())

    inf_count = int(
        np.isinf(
            X.select_dtypes(include=[np.number])
        ).sum().sum()
    )

    print(f"NaN values found      : {nan_count}")
    print(f"Infinite values found : {inf_count}")

    if nan_count > 0:

        print()
        print("Filling NaN values with 0.")

        X = X.fillna(0)

    if inf_count > 0:

        print()
        print("Replacing infinite values.")

        X = X.replace(
            [np.inf, -np.inf],
            0
        )

    print()
    print(
        f"Feature matrix: {X.shape[0]:,} x {X.shape[1]}"
    )

    return X


# ======================================================================
# CALCULATE METRICS
# ======================================================================

def calculate_metrics(
    y_true,
    y_pred,
    scores
):

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

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

    total_benign = tn + fp

    if total_benign > 0:
        fpr = fp / total_benign
    else:
        fpr = 0.0

    total_attack = tp + fn

    if total_attack > 0:
        detection_rate = tp / total_attack
    else:
        detection_rate = 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "detection_rate": detection_rate,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


# ======================================================================
# ATTACK TYPE METRICS
# ======================================================================

def attack_detection_rate(
    df,
    predictions,
    attack_type
):

    mask = (
        df["AttackType"] == attack_type
    )

    total = int(mask.sum())

    if total == 0:
        return 0.0, 0

    detected = int(
        predictions[mask].sum()
    )

    rate = detected / total

    return rate, detected


# ======================================================================
# MAIN THRESHOLD ANALYSIS
# ======================================================================

def run_threshold_analysis(
    model,
    df,
    X
):

    print_header("GENERATING XGBOOST SCORES")

    print("Generating probability scores...")

    probabilities = model.predict_proba(X)[:, 1]

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64
    )

    print("Probability generation complete.")

    y_true = df["Target"].astype(int).to_numpy()

    # ------------------------------------------------------------------
    # Threshold-independent metrics
    # ------------------------------------------------------------------

    try:
        roc_auc = roc_auc_score(
            y_true,
            probabilities
        )
    except Exception:
        roc_auc = None

    try:
        pr_auc = average_precision_score(
            y_true,
            probabilities
        )
    except Exception:
        pr_auc = None

    print()
    print(f"ROC-AUC : {roc_auc:.4f}" if roc_auc is not None else "ROC-AUC : N/A")
    print(f"PR-AUC  : {pr_auc:.4f}" if pr_auc is not None else "PR-AUC  : N/A")

    # ------------------------------------------------------------------
    # Evaluate thresholds
    # ------------------------------------------------------------------

    print_header("THRESHOLD EVALUATION")

    results = []

    for threshold in THRESHOLDS:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        metrics = calculate_metrics(
            y_true,
            predictions,
            probabilities
        )

        syn_rate, syn_detected = (
            attack_detection_rate(
                df,
                predictions,
                "SYN_FLOOD"
            )
        )

        udp_rate, udp_detected = (
            attack_detection_rate(
                df,
                predictions,
                "UDP_FLOOD"
            )
        )

        port_rate, port_detected = (
            attack_detection_rate(
                df,
                predictions,
                "PORT_SCAN"
            )
        )

        results.append({

            "threshold": float(threshold),

            "accuracy": metrics["accuracy"],

            "precision": metrics["precision"],

            "recall": metrics["recall"],

            "f1": metrics["f1"],

            "false_positive_rate": metrics["fpr"],

            "attack_detection_rate": (
                metrics["detection_rate"]
            ),

            "true_negatives": metrics["tn"],

            "false_positives": metrics["fp"],

            "false_negatives": metrics["fn"],

            "true_positives": metrics["tp"],

            "syn_detection_rate": syn_rate,

            "syn_detected": syn_detected,

            "udp_detection_rate": udp_rate,

            "udp_detected": udp_detected,

            "port_scan_detection_rate": port_rate,

            "port_scan_detected": port_detected,

            "roc_auc": roc_auc,

            "pr_auc": pr_auc,
        })

    results_df = pd.DataFrame(results)

    return results_df, probabilities


# ======================================================================
# FIND BEST THRESHOLDS
# ======================================================================

def find_recommendations(results_df):

    print_header("THRESHOLD RECOMMENDATIONS")

    # ------------------------------------------------------------------
    # 1. Best F1
    # ------------------------------------------------------------------

    best_f1_row = results_df.loc[
        results_df["f1"].idxmax()
    ]

    best_f1_threshold = float(
        best_f1_row["threshold"]
    )

    # ------------------------------------------------------------------
    # 2. Highest recall with FPR <= 1%
    # ------------------------------------------------------------------

    low_fpr = results_df[
        results_df["false_positive_rate"] <= 0.01
    ]

    if len(low_fpr) > 0:

        high_recall_low_fpr = low_fpr.loc[
            low_fpr["recall"].idxmax()
        ]

    else:

        high_recall_low_fpr = best_f1_row

    # ------------------------------------------------------------------
    # 3. Highest recall with FPR <= 0.5%
    # ------------------------------------------------------------------

    strict_fpr = results_df[
        results_df["false_positive_rate"] <= 0.005
    ]

    if len(strict_fpr) > 0:

        strict_row = strict_fpr.loc[
            strict_fpr["recall"].idxmax()
        ]

    else:

        strict_row = best_f1_row

    # ------------------------------------------------------------------
    # 4. High precision threshold
    # ------------------------------------------------------------------

    high_precision = results_df[
        results_df["precision"] >= 0.95
    ]

    if len(high_precision) > 0:

        high_precision_row = high_precision.loc[
            high_precision["recall"].idxmax()
        ]

    else:

        high_precision_row = best_f1_row

    # ------------------------------------------------------------------
    # Print recommendations
    # ------------------------------------------------------------------

    print_section("BEST F1 THRESHOLD")

    print(
        f"Threshold : "
        f"{best_f1_row['threshold']:.2f}"
    )

    print(
        f"Precision : "
        f"{best_f1_row['precision']:.4f}"
    )

    print(
        f"Recall    : "
        f"{best_f1_row['recall']:.4f}"
    )

    print(
        f"F1        : "
        f"{best_f1_row['f1']:.4f}"
    )

    print(
        f"FPR       : "
        f"{best_f1_row['false_positive_rate']:.4f}"
    )

    print_section(
        "BEST RECALL WITH FPR <= 1%"
    )

    print(
        f"Threshold : "
        f"{high_recall_low_fpr['threshold']:.2f}"
    )

    print(
        f"Precision : "
        f"{high_recall_low_fpr['precision']:.4f}"
    )

    print(
        f"Recall    : "
        f"{high_recall_low_fpr['recall']:.4f}"
    )

    print(
        f"F1        : "
        f"{high_recall_low_fpr['f1']:.4f}"
    )

    print(
        f"FPR       : "
        f"{high_recall_low_fpr['false_positive_rate']:.4f}"
    )

    print_section(
        "BEST RECALL WITH FPR <= 0.5%"
    )

    print(
        f"Threshold : "
        f"{strict_row['threshold']:.2f}"
    )

    print(
        f"Precision : "
        f"{strict_row['precision']:.4f}"
    )

    print(
        f"Recall    : "
        f"{strict_row['recall']:.4f}"
    )

    print(
        f"F1        : "
        f"{strict_row['f1']:.4f}"
    )

    print(
        f"FPR       : "
        f"{strict_row['false_positive_rate']:.4f}"
    )

    print_section(
        "BEST HIGH-PRECISION THRESHOLD"
    )

    print(
        f"Threshold : "
        f"{high_precision_row['threshold']:.2f}"
    )

    print(
        f"Precision : "
        f"{high_precision_row['precision']:.4f}"
    )

    print(
        f"Recall    : "
        f"{high_precision_row['recall']:.4f}"
    )

    print(
        f"F1        : "
        f"{high_precision_row['f1']:.4f}"
    )

    print(
        f"FPR       : "
        f"{high_precision_row['false_positive_rate']:.4f}"
    )

    # ------------------------------------------------------------------
    # Overall recommendation
    #
    # For an IDS we prioritize F1 while keeping false positives
    # reasonably controlled.
    # ------------------------------------------------------------------

    candidates = results_df[
        results_df["false_positive_rate"] <= 0.01
    ].copy()

    if len(candidates) > 0:

        # F1 is the primary objective.
        recommended = candidates.loc[
            candidates["f1"].idxmax()
        ]

    else:

        recommended = best_f1_row

    print_header("FINAL RECOMMENDED IDS THRESHOLD")

    print(
        f"Recommended threshold : "
        f"{recommended['threshold']:.2f}"
    )

    print(
        f"Precision             : "
        f"{recommended['precision']:.4f}"
    )

    print(
        f"Recall                : "
        f"{recommended['recall']:.4f}"
    )

    print(
        f"F1                    : "
        f"{recommended['f1']:.4f}"
    )

    print(
        f"False-positive rate   : "
        f"{recommended['false_positive_rate']:.4f}"
    )

    print(
        f"Attack detection rate : "
        f"{recommended['attack_detection_rate']:.4f}"
    )

    return {
        "recommended_threshold": float(
            recommended["threshold"]
        ),

        "recommended_precision": float(
            recommended["precision"]
        ),

        "recommended_recall": float(
            recommended["recall"]
        ),

        "recommended_f1": float(
            recommended["f1"]
        ),

        "recommended_false_positive_rate": float(
            recommended["false_positive_rate"]
        ),

        "recommended_attack_detection_rate": float(
            recommended["attack_detection_rate"]
        ),

        "best_f1_threshold": best_f1_threshold,

        "best_f1": float(
            best_f1_row["f1"]
        ),

        "best_recall_threshold_fpr_1pct": float(
            high_recall_low_fpr["threshold"]
        ),

        "best_recall_fpr_1pct": float(
            high_recall_low_fpr["recall"]
        ),

        "best_recall_threshold_fpr_0_5pct": float(
            strict_row["threshold"]
        ),

        "best_recall_fpr_0_5pct": float(
            strict_row["recall"]
        ),

        "high_precision_threshold": float(
            high_precision_row["threshold"]
        ),

        "high_precision": float(
            high_precision_row["precision"]
        ),

        "high_precision_recall": float(
            high_precision_row["recall"]
        ),
    }


# ======================================================================
# PRINT COMPLETE TABLE
# ======================================================================

def print_results_table(results_df):

    print_header("COMPLETE THRESHOLD RESULTS")

    display_df = results_df[
        [
            "threshold",
            "precision",
            "recall",
            "f1",
            "false_positive_rate",
            "attack_detection_rate",
            "syn_detection_rate",
            "udp_detection_rate",
            "port_scan_detection_rate",
        ]
    ].copy()

    # Convert to percentages for readable output
    percentage_columns = [
        "precision",
        "recall",
        "f1",
        "false_positive_rate",
        "attack_detection_rate",
        "syn_detection_rate",
        "udp_detection_rate",
        "port_scan_detection_rate",
    ]

    for column in percentage_columns:

        display_df[column] = (
            display_df[column] * 100
        ).round(2)

    display_df.columns = [
        "Threshold",
        "Precision %",
        "Recall %",
        "F1 %",
        "FPR %",
        "Detection %",
        "SYN %",
        "UDP %",
        "PortScan %",
    ]

    print(
        display_df.to_string(
            index=False
        )
    )


# ======================================================================
# SAVE RESULTS
# ======================================================================

def save_results(
    results_df,
    recommendations
):

    print_header("SAVING RESULTS")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    results_df.to_csv(
        RESULTS_PATH,
        index=False
    )

    print(
        "Threshold analysis saved to:"
    )

    print(
        f"  {RESULTS_PATH}"
    )

    with open(
        RECOMMENDATION_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            recommendations,
            f,
            indent=2
        )

    print()
    print(
        "Threshold recommendation saved to:"
    )

    print(
        f"  {RECOMMENDATION_PATH}"
    )


# ======================================================================
# MAIN
# ======================================================================

def main():

    print_header(
        "CTU-13 XGBOOST THRESHOLD ANALYSIS"
    )

    print("Configuration:")

    print(
        f"  Model                 : XGBoost"
    )

    print(
        f"  Dataset               : "
        f"{DATA_PATH}"
    )

    print(
        f"  Threshold range      : "
        f"0.05 - 0.90"
    )

    print(
        f"  Threshold step       : "
        f"0.05"
    )

    print(
        f"  Random seed          : "
        f"{RANDOM_SEED}"
    )

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    model = load_model()

    df = load_dataset()

    validate_dataset(df)

    X = prepare_features(df)

    # ------------------------------------------------------------------
    # Analyze
    # ------------------------------------------------------------------

    results_df, probabilities = (
        run_threshold_analysis(
            model,
            df,
            X
        )
    )

    # ------------------------------------------------------------------
    # Recommendations
    # ------------------------------------------------------------------

    recommendations = find_recommendations(
        results_df
    )

    # ------------------------------------------------------------------
    # Complete table
    # ------------------------------------------------------------------

    print_results_table(
        results_df
    )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    save_results(
        results_df,
        recommendations
    )

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------

    print_header(
        "THRESHOLD ANALYSIS COMPLETE"
    )

    print(
        f"Thresholds evaluated : "
        f"{len(THRESHOLDS)}"
    )

    print(
        f"Dataset rows         : "
        f"{len(df):,}"
    )

    print(
        f"ROC-AUC              : "
        f"{results_df['roc_auc'].iloc[0]:.4f}"
    )

    print(
        f"PR-AUC               : "
        f"{results_df['pr_auc'].iloc[0]:.4f}"
    )

    print()

    print(
        "Recommended threshold:"
    )

    print(
        f"  "
        f"{recommendations['recommended_threshold']:.2f}"
    )

    print()

    print(
        "Output files:"
    )

    print(
        f"  {RESULTS_PATH}"
    )

    print(
        f"  {RECOMMENDATION_PATH}"
    )

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


# ======================================================================
# ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    main()