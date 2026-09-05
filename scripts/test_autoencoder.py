"""
CTU-13 AUTOENCODER EVALUATION
=============================

Evaluates the trained CTU-13 Autoencoder on unseen scenarios 11-13.

IMPORTANT:
The Autoencoder was trained using the feature schema stored in:

    models/autoencoder_features.json

The schema uses:

    numeric_features
    categorical_features
    encoded_feature_names

The evaluator MUST reproduce the same 257-dimensional feature
representation before passing data to the scaler/model.
"""

from __future__ import annotations

import json
import os
import random
import warnings
from pathlib import Path

# Reduce TensorFlow console noise.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ======================================================================
# CONFIGURATION
# ======================================================================

SEED = 42

BASE_DIR = Path(__file__).resolve().parent.parent

MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

AUTOENCODER_PATH = (
    MODELS_DIR / "ctu13_autoencoder.keras"
)

SCALER_PATH = (
    MODELS_DIR / "autoencoder_scaler.joblib"
)

FEATURE_SCHEMA_PATH = (
    MODELS_DIR / "autoencoder_features.json"
)

TRAIN_CONFIG_PATH = (
    MODELS_DIR / "autoencoder_config.json"
)

OUTPUT_BENCHMARK = (
    MODELS_DIR / "autoencoder_benchmark.csv"
)

OUTPUT_SCENARIO = (
    MODELS_DIR / "autoencoder_benchmark_per_scenario.csv"
)

OUTPUT_PREDICTIONS = (
    MODELS_DIR / "autoencoder_predictions.csv"
)

OUTPUT_CONFIG = (
    MODELS_DIR / "autoencoder_test_config.json"
)

TEST_SCENARIOS = [11, 12, 13]

# Fallback only if the training configuration does not contain one.
DEFAULT_THRESHOLD = 0.25541964

PREDICT_BATCH_SIZE = 2048


# ======================================================================
# REPRODUCIBILITY
# ======================================================================

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ======================================================================
# DISPLAY
# ======================================================================

def print_header(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def print_section(title: str) -> None:
    print()
    print("-" * 70)
    print(title)
    print("-" * 70)


# ======================================================================
# FILE CHECK
# ======================================================================

def check_required_files() -> None:

    required = [
        AUTOENCODER_PATH,
        SCALER_PATH,
        FEATURE_SCHEMA_PATH,
    ]

    missing = [
        path
        for path in required
        if not path.exists()
    ]

    if missing:

        print()
        print("ERROR: Missing required files:")

        for path in missing:
            print(f"  - {path}")

        raise FileNotFoundError(
            "Required Autoencoder artifacts are missing."
        )


# ======================================================================
# LOAD TRAINING CONFIG
# ======================================================================

def load_training_config() -> dict:

    if not TRAIN_CONFIG_PATH.exists():
        return {}

    try:

        with open(
            TRAIN_CONFIG_PATH,
            "r",
            encoding="utf-8",
        ) as f:

            return json.load(f)

    except Exception as exc:

        print(
            f"Warning: Could not load training config: {exc}"
        )

        return {}


# ======================================================================
# LOAD FEATURE SCHEMA
# ======================================================================

def load_feature_schema() -> dict:

    print_section(
        "LOADING AUTOENCODER FEATURE SCHEMA"
    )

    print(
        f"Path: {FEATURE_SCHEMA_PATH}"
    )

    with open(
        FEATURE_SCHEMA_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        schema = json.load(f)

    if not isinstance(schema, dict):

        raise ValueError(
            "autoencoder_features.json must contain a JSON object."
        )

    # --------------------------------------------------------------
    # THIS IS THE IMPORTANT FIX.
    #
    # Your actual schema stores the final feature list under:
    #
    #     encoded_feature_names
    # --------------------------------------------------------------

    encoded_features = schema.get(
        "encoded_feature_names"
    )

    numeric_features = schema.get(
        "numeric_features",
        []
    )

    categorical_features = schema.get(
        "categorical_features",
        []
    )

    if not isinstance(
        encoded_features,
        list,
    ):

        raise ValueError(
            "Could not find 'encoded_feature_names' "
            "inside autoencoder_features.json."
        )

    if not encoded_features:

        raise ValueError(
            "encoded_feature_names is empty."
        )

    encoded_features = [
        str(x)
        for x in encoded_features
    ]

    numeric_features = [
        str(x)
        for x in numeric_features
    ]

    categorical_features = [
        str(x)
        for x in categorical_features
    ]

    print(
        f"Numeric features    : "
        f"{len(numeric_features)}"
    )

    print(
        f"Categorical features: "
        f"{len(categorical_features)}"
    )

    print(
        f"Encoded features    : "
        f"{len(encoded_features)}"
    )

    print()
    print("Categorical columns:")

    for feature in categorical_features:
        print(f"  {feature}")

    print()
    print("First encoded features:")

    for feature in encoded_features[:20]:
        print(f"  {feature}")

    if len(encoded_features) > 20:
        print("  ...")

    return {
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "encoded_feature_names": encoded_features,
    }


# ======================================================================
# LOAD AUTOENCODER
# ======================================================================

def load_autoencoder():

    print_section(
        "LOADING AUTOENCODER"
    )

    print(
        f"Path: {AUTOENCODER_PATH}"
    )

    model = tf.keras.models.load_model(
        AUTOENCODER_PATH,
        compile=False,
    )

    print(
        "Autoencoder loaded successfully."
    )

    print(
        f"Input shape: {model.input_shape}"
    )

    return model


# ======================================================================
# LOAD SCALER
# ======================================================================

def load_scaler():

    print_section(
        "LOADING SCALER"
    )

    print(
        f"Path: {SCALER_PATH}"
    )

    scaler = joblib.load(
        SCALER_PATH
    )

    print(
        "Scaler loaded successfully."
    )

    if hasattr(
        scaler,
        "n_features_in_",
    ):

        print(
            f"Scaler features: "
            f"{scaler.n_features_in_}"
        )

    return scaler


# ======================================================================
# LOAD SCENARIO
# ======================================================================

def load_scenario(
    scenario: int,
) -> pd.DataFrame:

    path = (
        DATA_DIR
        / f"scenario{scenario}"
        / "ctu13_features.csv"
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Scenario file not found:\n{path}"
        )

    print(
        f"Loading Scenario {scenario}: {path}"
    )

    df = pd.read_csv(path)

    print(
        f"  Rows    : {len(df):,}"
    )

    print(
        f"  Columns : {len(df.columns)}"
    )

    if "Target" not in df.columns:

        raise ValueError(
            f"Scenario {scenario} "
            "does not contain Target."
        )

    return df


# ======================================================================
# BUILD ONE-HOT COLUMN MAP
# ======================================================================

def build_one_hot_schema(
    encoded_features: list[str],
    categorical_features: list[str],
) -> dict[str, list[str]]:

    mapping = {}

    for categorical in categorical_features:

        candidates = []

        prefix = categorical + "_"

        for feature in encoded_features:

            if feature.startswith(prefix):

                candidates.append(feature)

        mapping[categorical] = candidates

    return mapping


# ======================================================================
# PREPARE FEATURES
# ======================================================================

def prepare_features(
    df: pd.DataFrame,
    schema: dict,
) -> pd.DataFrame:

    numeric_features = schema[
        "numeric_features"
    ]

    categorical_features = schema[
        "categorical_features"
    ]

    encoded_features = schema[
        "encoded_feature_names"
    ]

    print(
        f"Preparing {len(encoded_features)} "
        "Autoencoder features..."
    )

    # --------------------------------------------------------------
    # Verify categorical source columns.
    # --------------------------------------------------------------

    missing_categorical = [
        column
        for column in categorical_features
        if column not in df.columns
    ]

    if missing_categorical:

        # Proto/Protocol compatibility.
        if (
            "Proto" in missing_categorical
            and "Protocol" in df.columns
        ):

            missing_categorical.remove(
                "Proto"
            )

        if missing_categorical:

            raise ValueError(
                "Missing categorical source columns:\n"
                + "\n".join(
                    f"  - {x}"
                    for x in missing_categorical
                )
            )

    # --------------------------------------------------------------
    # Create final DataFrame with EXACT schema order.
    # --------------------------------------------------------------

    result = pd.DataFrame(
        0.0,
        index=df.index,
        columns=encoded_features,
        dtype=np.float32,
    )

    # --------------------------------------------------------------
    # NUMERIC FEATURES
    # --------------------------------------------------------------

    missing_numeric = []

    for feature in numeric_features:

        if feature not in df.columns:

            missing_numeric.append(
                feature
            )

            continue

        values = pd.to_numeric(
            df[feature],
            errors="coerce",
        )

        values = values.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        result[feature] = (
            values.astype(np.float32)
        )

    if missing_numeric:

        raise ValueError(
            "Missing numeric features required "
            "by Autoencoder:\n"
            + "\n".join(
                f"  - {x}"
                for x in missing_numeric
            )
        )

    # --------------------------------------------------------------
    # CATEGORICAL FEATURES
    # --------------------------------------------------------------

    for categorical in categorical_features:

        source_column = categorical

        # Raw CTU-13 uses Proto.
        if (
            source_column not in df.columns
            and categorical == "Protocol"
            and "Proto" in df.columns
        ):

            source_column = "Proto"

        if (
            source_column not in df.columns
            and categorical == "Proto"
            and "Protocol" in df.columns
        ):

            source_column = "Protocol"

        values = (
            df[source_column]
            .astype("string")
            .fillna("__MISSING__")
        )

        # Convert to strings exactly like the schema names.
        values = values.astype(str)

        # ----------------------------------------------------------
        # Determine the schema prefix.
        #
        # For example:
        #
        # Proto_esp
        # Proto_icmp
        # Proto_tcp
        #
        # Dir_->
        #
        # State_A_A
        # ----------------------------------------------------------

        prefix = categorical + "_"

        schema_columns = [
            feature
            for feature in encoded_features
            if feature.startswith(prefix)
        ]

        if not schema_columns:
            continue

        # ----------------------------------------------------------
        # Direct one-hot assignment.
        # ----------------------------------------------------------

        for encoded_column in schema_columns:

            category = encoded_column[
                len(prefix):
            ]

            result[encoded_column] = (
                values == category
            ).astype(np.float32)

    # --------------------------------------------------------------
    # CLEAN NUMERIC VALUES
    # --------------------------------------------------------------

    nan_before = int(
        result.isna().sum().sum()
    )

    inf_before = int(
        np.isinf(
            result.to_numpy(
                dtype=np.float32
            )
        ).sum()
    )

    print(
        f"NaN values before cleaning : "
        f"{nan_before:,}"
    )

    print(
        f"Infinite values before cleaning : "
        f"{inf_before:,}"
    )

    result = result.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # --------------------------------------------------------------
    # Median imputation.
    #
    # This is applied ONLY after constructing the exact 257-column
    # representation.
    # --------------------------------------------------------------

    for column in result.columns:

        if result[column].isna().any():

            median = result[column].median()

            if pd.isna(median):
                median = 0.0

            result[column] = (
                result[column]
                .fillna(median)
            )

    result = result.fillna(0.0)

    result = result.astype(
        np.float32
    )

    # --------------------------------------------------------------
    # Final validation.
    # --------------------------------------------------------------

    nan_after = int(
        result.isna().sum().sum()
    )

    inf_after = int(
        np.isinf(
            result.to_numpy(
                dtype=np.float32
            )
        ).sum()
    )

    print(
        f"NaN values after cleaning  : "
        f"{nan_after:,}"
    )

    print(
        f"Infinite values after cleaning : "
        f"{inf_after:,}"
    )

    if nan_after != 0:

        raise ValueError(
            "NaN values remain after feature cleaning."
        )

    if inf_after != 0:

        raise ValueError(
            "Infinite values remain after feature cleaning."
        )

    # --------------------------------------------------------------
    # Exact schema order.
    # --------------------------------------------------------------

    result = result[
        encoded_features
    ]

    print(
        f"Feature matrix: "
        f"{result.shape[0]:,} x "
        f"{result.shape[1]}"
    )

    return result


# ======================================================================
# SCALE FEATURES
# ======================================================================

def scale_features(
    X_df: pd.DataFrame,
    scaler,
) -> np.ndarray:

    print(
        "Scaling features..."
    )

    # --------------------------------------------------------------
    # Important:
    # Use the DataFrame rather than a raw ndarray so that a scaler
    # fitted with feature names can validate the same names/order.
    # --------------------------------------------------------------

    try:

        X = scaler.transform(
            X_df
        )

    except Exception as exc:

        print()
        print(
            "Scaler transform failed."
        )

        print(
            f"Reason: {exc}"
        )

        # Safe fallback for scalers that were fitted without
        # feature names.
        X = scaler.transform(
            X_df.to_numpy(
                dtype=np.float32
            )
        )

    X = np.asarray(
        X,
        dtype=np.float32,
    )

    X = np.nan_to_num(
        X,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    print(
        f"Scaled matrix: "
        f"{X.shape[0]:,} x "
        f"{X.shape[1]}"
    )

    return X


# ======================================================================
# RECONSTRUCTION ERROR
# ======================================================================

def calculate_reconstruction_error(
    model,
    X: np.ndarray,
) -> np.ndarray:

    print(
        "Running Autoencoder..."
    )

    reconstruction = model.predict(
        X,
        batch_size=PREDICT_BATCH_SIZE,
        verbose=0,
    )

    reconstruction = np.asarray(
        reconstruction,
        dtype=np.float32,
    )

    # --------------------------------------------------------------
    # Per-row Mean Squared Error.
    # --------------------------------------------------------------

    errors = np.mean(
        np.square(
            X - reconstruction
        ),
        axis=1,
    )

    errors = np.asarray(
        errors,
        dtype=np.float64,
    )

    errors = np.nan_to_num(
        errors,
        nan=np.inf,
        posinf=np.inf,
        neginf=0.0,
    )

    return errors


# ======================================================================
# METRICS
# ======================================================================

def calculate_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    predictions: np.ndarray,
) -> dict:

    cm = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    )

    tn, fp, fn, tp = cm.ravel()

    accuracy = accuracy_score(
        y_true,
        predictions,
    )

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0,
    )

    try:

        roc_auc = roc_auc_score(
            y_true,
            scores,
        )

    except ValueError:

        roc_auc = float("nan")

    try:

        pr_auc = average_precision_score(
            y_true,
            scores,
        )

    except ValueError:

        pr_auc = float("nan")

    total_negative = tn + fp

    if total_negative:

        fpr = fp / total_negative

    else:

        fpr = 0.0

    total_positive = tp + fn

    if total_positive:

        detection_rate = (
            tp / total_positive
        )

    else:

        detection_rate = 0.0

    return {
        "Accuracy": float(accuracy),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(f1),
        "ROC_AUC": float(roc_auc),
        "PR_AUC": float(pr_auc),
        "FPR": float(fpr),
        "Detection_Rate": float(
            detection_rate
        ),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


# ======================================================================
# PRINT METRICS
# ======================================================================

def print_metrics(
    name: str,
    metrics: dict,
) -> None:

    print()
    print(name)
    print("-" * 40)

    print(
        f"Accuracy : "
        f"{metrics['Accuracy']:.4f}"
    )

    print(
        f"Precision: "
        f"{metrics['Precision']:.4f}"
    )

    print(
        f"Recall   : "
        f"{metrics['Recall']:.4f}"
    )

    print(
        f"F1 Score : "
        f"{metrics['F1']:.4f}"
    )

    print(
        f"ROC-AUC  : "
        f"{metrics['ROC_AUC']:.4f}"
    )

    print(
        f"PR-AUC   : "
        f"{metrics['PR_AUC']:.4f}"
    )

    print(
        f"FPR      : "
        f"{metrics['FPR']:.4f}"
    )

    print(
        f"Detection: "
        f"{metrics['Detection_Rate']:.4f}"
    )


# ======================================================================
# DETERMINE THRESHOLD
# ======================================================================

def determine_threshold(
    train_config: dict,
) -> float:

    possible_keys = [
        "anomaly_threshold",
        "threshold",
        "reconstruction_threshold",
        "best_threshold",
    ]

    for key in possible_keys:

        value = train_config.get(
            key
        )

        if value is not None:

            try:

                return float(value)

            except (
                TypeError,
                ValueError,
            ):

                pass

    return DEFAULT_THRESHOLD


# ======================================================================
# VALIDATE ARTIFACT DIMENSIONS
# ======================================================================

def validate_dimensions(
    model,
    scaler,
    schema_features: list[str],
) -> None:

    expected = len(
        schema_features
    )

    print_section(
        "VALIDATING FEATURE DIMENSIONS"
    )

    print(
        f"Schema features : {expected}"
    )

    # Model.
    model_input = model.input_shape[-1]

    print(
        f"Model input     : {model_input}"
    )

    if (
        model_input is not None
        and int(model_input) != expected
    ):

        raise ValueError(
            "Model/schema mismatch.\n"
            f"Model expects {model_input} "
            f"features, but schema contains "
            f"{expected}."
        )

    # Scaler.
    if hasattr(
        scaler,
        "n_features_in_",
    ):

        scaler_input = int(
            scaler.n_features_in_
        )

        print(
            f"Scaler input    : {scaler_input}"
        )

        if scaler_input != expected:

            raise ValueError(
                "Scaler/schema mismatch.\n"
                f"Scaler expects {scaler_input} "
                f"features, but schema contains "
                f"{expected}."
            )

    print(
        "Feature dimensions validated."
    )


# ======================================================================
# MAIN
# ======================================================================

def main():

    print_header(
        "CTU-13 AUTOENCODER EVALUATION"
    )

    print(
        "Configuration:"
    )

    print(
        "  Training scenarios       : 1 - 10"
    )

    print(
        "  Testing scenarios        : 11 - 13"
    )

    print(
        f"  Random seed              : {SEED}"
    )

    print(
        f"  Prediction batch size    : "
        f"{PREDICT_BATCH_SIZE}"
    )

    # --------------------------------------------------------------
    # Files.
    # --------------------------------------------------------------

    check_required_files()

    # --------------------------------------------------------------
    # Load artifacts.
    # --------------------------------------------------------------

    model = load_autoencoder()

    scaler = load_scaler()

    schema = load_feature_schema()

    train_config = (
        load_training_config()
    )

    encoded_features = schema[
        "encoded_feature_names"
    ]

    # --------------------------------------------------------------
    # Threshold.
    # --------------------------------------------------------------

    threshold = determine_threshold(
        train_config
    )

    print_section(
        "ANOMALY THRESHOLD"
    )

    print(
        f"Threshold: {threshold:.8f}"
    )

    # --------------------------------------------------------------
    # Dimension validation.
    # --------------------------------------------------------------

    validate_dimensions(
        model,
        scaler,
        encoded_features,
    )

    # --------------------------------------------------------------
    # Load scenarios.
    # --------------------------------------------------------------

    print_header(
        "LOADING UNSEEN TEST SCENARIOS"
    )

    scenarios = {}

    total_rows = 0

    for scenario in TEST_SCENARIOS:

        df = load_scenario(
            scenario
        )

        scenarios[scenario] = df

        total_rows += len(df)

    print()

    print(
        f"TOTAL TEST ROWS: "
        f"{total_rows:,}"
    )

    # --------------------------------------------------------------
    # Target distribution.
    # --------------------------------------------------------------

    all_targets = pd.concat(
        [
            df["Target"]
            for df in scenarios.values()
        ],
        ignore_index=True,
    )

    print_section(
        "TEST TARGET DISTRIBUTION"
    )

    print(
        all_targets
        .value_counts()
        .sort_index()
        .to_string()
    )

    # --------------------------------------------------------------
    # Prediction.
    # --------------------------------------------------------------

    print_header(
        "AUTOENCODER PREDICTION"
    )

    scenario_results = []

    prediction_frames = []

    all_y_true = []
    all_scores = []
    all_predictions = []

    for scenario in TEST_SCENARIOS:

        df = scenarios[scenario]

        print_header(
            f"PROCESSING SCENARIO {scenario}"
        )

        print(
            f"Rows: {len(df):,}"
        )

        # ----------------------------------------------------------
        # Features.
        # ----------------------------------------------------------

        X_df = prepare_features(
            df,
            schema,
        )

        # ----------------------------------------------------------
        # Scale.
        # ----------------------------------------------------------

        X = scale_features(
            X_df,
            scaler,
        )

        # ----------------------------------------------------------
        # Reconstruction error.
        # ----------------------------------------------------------

        scores = (
            calculate_reconstruction_error(
                model,
                X,
            )
        )

        # ----------------------------------------------------------
        # Target.
        # ----------------------------------------------------------

        y_true = (
            pd.to_numeric(
                df["Target"],
                errors="coerce",
            )
            .fillna(0)
            .astype(np.int8)
            .to_numpy()
        )

        # ----------------------------------------------------------
        # Prediction.
        # ----------------------------------------------------------

        predictions = (
            scores >= threshold
        ).astype(np.int8)

        # ----------------------------------------------------------
        # Metrics.
        # ----------------------------------------------------------

        metrics = calculate_metrics(
            y_true,
            scores,
            predictions,
        )

        print_section(
            f"SCENARIO {scenario} RESULTS"
        )

        print_metrics(
            "AUTOENCODER",
            metrics,
        )

        # ----------------------------------------------------------
        # Confusion matrix.
        # ----------------------------------------------------------

        print()
        print(
            "CONFUSION MATRIX"
        )

        print("-" * 40)

        print(
            f"[[{metrics['TN']:>8} "
            f"{metrics['FP']:>8}]"
        )

        print(
            f" [{metrics['FN']:>8} "
            f"{metrics['TP']:>8}]]"
        )

        print()
        print(
            f"True Negatives : "
            f"{metrics['TN']:,}"
        )

        print(
            f"False Positives: "
            f"{metrics['FP']:,}"
        )

        print(
            f"False Negatives: "
            f"{metrics['FN']:,}"
        )

        print(
            f"True Positives : "
            f"{metrics['TP']:,}"
        )

        # ----------------------------------------------------------
        # Error statistics.
        # ----------------------------------------------------------

        print()
        print(
            "RECONSTRUCTION ERROR"
        )

        print("-" * 40)

        print(
            f"Mean   : "
            f"{np.mean(scores):.8f}"
        )

        print(
            f"Median : "
            f"{np.median(scores):.8f}"
        )

        print(
            f"Std    : "
            f"{np.std(scores):.8f}"
        )

        print(
            f"Min    : "
            f"{np.min(scores):.8f}"
        )

        print(
            f"Max    : "
            f"{np.max(scores):.8f}"
        )

        print()
        print(
            f"Anomalies detected: "
            f"{predictions.sum():,}"
        )

        print(
            f"Anomaly rate: "
            f"{predictions.mean() * 100:.4f}%"
        )

        # ----------------------------------------------------------
        # Scenario result.
        # ----------------------------------------------------------

        scenario_results.append(
            {
                "Scenario": scenario,
                "Rows": len(df),
                "Threshold": threshold,
                **metrics,
                "Anomalies": int(
                    predictions.sum()
                ),
                "Anomaly_Rate": float(
                    predictions.mean()
                ),
                "Mean_Error": float(
                    np.mean(scores)
                ),
                "Median_Error": float(
                    np.median(scores)
                ),
                "Std_Error": float(
                    np.std(scores)
                ),
                "Min_Error": float(
                    np.min(scores)
                ),
                "Max_Error": float(
                    np.max(scores)
                ),
            }
        )

        # ----------------------------------------------------------
        # Detailed predictions.
        # ----------------------------------------------------------

        prediction_df = pd.DataFrame(
            {
                "Scenario": scenario,
                "Target": y_true,
                "ReconstructionError": scores,
                "Threshold": threshold,
                "Anomaly": predictions,
            }
        )

        useful_columns = [
            "StartTime",
            "SrcAddr",
            "Sport",
            "Proto",
            "Dir",
            "DstAddr",
            "Dport",
            "State",
            "Label",
            "ThreatClass",
        ]

        for column in useful_columns:

            if column in df.columns:

                prediction_df[column] = (
                    df[column]
                    .reset_index(drop=True)
                )

        preferred = [
            "Scenario",
            "StartTime",
            "SrcAddr",
            "Sport",
            "Proto",
            "Dir",
            "DstAddr",
            "Dport",
            "State",
            "Label",
            "ThreatClass",
            "Target",
            "ReconstructionError",
            "Threshold",
            "Anomaly",
        ]

        existing_preferred = [
            column
            for column in preferred
            if column in prediction_df.columns
        ]

        remaining = [
            column
            for column in prediction_df.columns
            if column not in existing_preferred
        ]

        prediction_df = prediction_df[
            existing_preferred + remaining
        ]

        prediction_frames.append(
            prediction_df
        )

        # ----------------------------------------------------------
        # Overall arrays.
        # ----------------------------------------------------------

        all_y_true.append(
            y_true
        )

        all_scores.append(
            scores
        )

        all_predictions.append(
            predictions
        )

        # Free memory.
        del X_df
        del X

    # ==================================================================
    # OVERALL
    # ==================================================================

    print_header(
        "OVERALL AUTOENCODER RESULTS"
    )

    y_true_all = np.concatenate(
        all_y_true
    )

    scores_all = np.concatenate(
        all_scores
    )

    predictions_all = np.concatenate(
        all_predictions
    )

    overall_metrics = calculate_metrics(
        y_true_all,
        scores_all,
        predictions_all,
    )

    print_metrics(
        "AUTOENCODER",
        overall_metrics,
    )

    # ==================================================================
    # CONFUSION MATRIX
    # ==================================================================

    print_section(
        "OVERALL CONFUSION MATRIX"
    )

    print(
        f"[[{overall_metrics['TN']:>8} "
        f"{overall_metrics['FP']:>8}]"
    )

    print(
        f" [{overall_metrics['FN']:>8} "
        f"{overall_metrics['TP']:>8}]]"
    )

    print()
    print(
        f"True Negatives : "
        f"{overall_metrics['TN']:,}"
    )

    print(
        f"False Positives: "
        f"{overall_metrics['FP']:,}"
    )

    print(
        f"False Negatives: "
        f"{overall_metrics['FN']:,}"
    )

    print(
        f"True Positives : "
        f"{overall_metrics['TP']:,}"
    )

    # ==================================================================
    # ERROR
    # ==================================================================

    print_section(
        "OVERALL RECONSTRUCTION ERROR"
    )

    print(
        f"Mean   : "
        f"{np.mean(scores_all):.8f}"
    )

    print(
        f"Median : "
        f"{np.median(scores_all):.8f}"
    )

    print(
        f"Std    : "
        f"{np.std(scores_all):.8f}"
    )

    print(
        f"Min    : "
        f"{np.min(scores_all):.8f}"
    )

    print(
        f"Max    : "
        f"{np.max(scores_all):.8f}"
    )

    print()
    print(
        f"Anomalies detected: "
        f"{predictions_all.sum():,}"
    )

    print(
        f"Overall anomaly rate: "
        f"{predictions_all.mean() * 100:.4f}%"
    )

    # ==================================================================
    # PER SCENARIO
    # ==================================================================

    print_header(
        "PER-SCENARIO AUTOENCODER SUMMARY"
    )

    scenario_df = pd.DataFrame(
        scenario_results
    )

    display_columns = [
        "Scenario",
        "Rows",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC_AUC",
        "PR_AUC",
        "FPR",
        "Detection_Rate",
        "Anomalies",
    ]

    print(
        scenario_df[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # ==================================================================
    # OVERALL BENCHMARK CSV
    # ==================================================================

    benchmark_row = {
        "Model": "Autoencoder",
        "Scenarios": "11-13",
        "Rows": len(y_true_all),
        "Threshold": threshold,
        **overall_metrics,
        "Anomalies": int(
            predictions_all.sum()
        ),
        "Anomaly_Rate": float(
            predictions_all.mean()
        ),
        "Mean_Error": float(
            np.mean(scores_all)
        ),
        "Median_Error": float(
            np.median(scores_all)
        ),
        "Std_Error": float(
            np.std(scores_all)
        ),
        "Min_Error": float(
            np.min(scores_all)
        ),
        "Max_Error": float(
            np.max(scores_all)
        ),
    }

    benchmark_df = pd.DataFrame(
        [benchmark_row]
    )

    # ==================================================================
    # SAVE
    # ==================================================================

    print_header(
        "SAVING RESULTS"
    )

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    benchmark_df.to_csv(
        OUTPUT_BENCHMARK,
        index=False,
    )

    print(
        "Overall results saved to:"
    )

    print(
        f"  {OUTPUT_BENCHMARK}"
    )

    scenario_df.to_csv(
        OUTPUT_SCENARIO,
        index=False,
    )

    print(
        "Per-scenario results saved to:"
    )

    print(
        f"  {OUTPUT_SCENARIO}"
    )

    predictions_df = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    predictions_df.to_csv(
        OUTPUT_PREDICTIONS,
        index=False,
    )

    print(
        "Detailed predictions saved to:"
    )

    print(
        f"  {OUTPUT_PREDICTIONS}"
    )

    # ==================================================================
    # SAVE CONFIG
    # ==================================================================

    output_config = {
        "model": "CTU13_Autoencoder",
        "training_scenarios": "1-10",
        "testing_scenarios": TEST_SCENARIOS,
        "random_seed": SEED,
        "threshold": threshold,
        "feature_count": len(
            encoded_features
        ),
        "numeric_feature_count": len(
            schema["numeric_features"]
        ),
        "categorical_feature_count": len(
            schema["categorical_features"]
        ),
        "prediction_batch_size": (
            PREDICT_BATCH_SIZE
        ),
        "test_rows": len(
            y_true_all
        ),
        "metrics": overall_metrics,
        "artifacts": {
            "model": str(
                AUTOENCODER_PATH
            ),
            "scaler": str(
                SCALER_PATH
            ),
            "feature_schema": str(
                FEATURE_SCHEMA_PATH
            ),
            "training_config": str(
                TRAIN_CONFIG_PATH
            ),
        },
        "outputs": {
            "benchmark": str(
                OUTPUT_BENCHMARK
            ),
            "per_scenario": str(
                OUTPUT_SCENARIO
            ),
            "predictions": str(
                OUTPUT_PREDICTIONS
            ),
        },
    }

    with open(
        OUTPUT_CONFIG,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output_config,
            f,
            indent=2,
            default=str,
        )

    print(
        "Configuration saved to:"
    )

    print(
        f"  {OUTPUT_CONFIG}"
    )

    # ==================================================================
    # FINAL
    # ==================================================================

    print_header(
        "AUTOENCODER EVALUATION COMPLETE"
    )

    print(
        "Training:"
    )

    print(
        "  Scenario 1 - Scenario 10"
    )

    print()
    print(
        "Testing:"
    )

    print(
        "  Scenario 11 - Scenario 13"
    )

    print()
    print(
        f"Test rows       : "
        f"{len(y_true_all):,}"
    )

    print(
        f"Input features  : "
        f"{len(encoded_features)}"
    )

    print(
        f"Threshold       : "
        f"{threshold:.8f}"
    )

    print()
    print(
        f"Accuracy        : "
        f"{overall_metrics['Accuracy']:.4f}"
    )

    print(
        f"Precision       : "
        f"{overall_metrics['Precision']:.4f}"
    )

    print(
        f"Recall          : "
        f"{overall_metrics['Recall']:.4f}"
    )

    print(
        f"F1              : "
        f"{overall_metrics['F1']:.4f}"
    )

    print(
        f"ROC-AUC         : "
        f"{overall_metrics['ROC_AUC']:.4f}"
    )

    print(
        f"PR-AUC          : "
        f"{overall_metrics['PR_AUC']:.4f}"
    )

    print()
    print(
        "Output files:"
    )

    print(
        f"  {OUTPUT_BENCHMARK}"
    )

    print(
        f"  {OUTPUT_SCENARIO}"
    )

    print(
        f"  {OUTPUT_PREDICTIONS}"
    )

    print(
        f"  {OUTPUT_CONFIG}"
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