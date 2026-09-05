"""
CTU-13 XGBoost + Autoencoder Gated Hybrid IDS

Architecture
------------
XGBoost:
    Primary supervised detector.

Autoencoder:
    Secondary anomaly detector.

Gating:
    Autoencoder is used only when XGBoost is uncertain.

Important:
    XGBoost features are ALWAYS reordered according to
    the actual feature names stored inside the trained model.
"""

from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

try:
    import tensorflow as tf
except ImportError:
    raise ImportError(
        "\nTensorFlow is not installed.\n"
        "Run:\n"
        "python -m pip install tensorflow\n"
    )

try:
    import xgboost as xgb
except ImportError:
    raise ImportError(
        "\nXGBoost is not installed.\n"
        "Run:\n"
        "python -m pip install xgboost\n"
    )


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

XGB_MODEL_PATH = (
    MODEL_DIR / "ctu13_multiscenario_xgboost.json"
)

AE_MODEL_PATH = (
    MODEL_DIR / "ctu13_autoencoder.keras"
)

AE_SCALER_PATH = (
    MODEL_DIR / "autoencoder_scaler.joblib"
)

AE_SCHEMA_PATH = (
    MODEL_DIR / "autoencoder_features.json"
)


# ============================================================
# CONFIGURATION
# ============================================================

TEST_SCENARIOS = [11, 12, 13]

RANDOM_SEED = 42

XGB_THRESHOLD = 0.20

AE_THRESHOLD = 0.25541964

UNCERTAINTY_LOW = 0.05
UNCERTAINTY_HIGH = 0.35

XGB_WEIGHT = 0.90
AE_WEIGHT = 0.10

HYBRID_THRESHOLD = 0.20

BATCH_SIZE = 2048


# ============================================================
# CTU-13 NUMERIC FEATURES
# ============================================================

NUMERIC_FEATURES = [
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


CATEGORICAL_FEATURES = [
    "Proto",
    "Dir",
    "State",
]


# ============================================================
# PRINT HELPERS
# ============================================================

def header(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def section(title):
    print()
    print("-" * 70)
    print(title)
    print("-" * 70)


# ============================================================
# MODEL FILE CHECK
# ============================================================

def check_model_files():

    header("CHECKING MODEL FILES")

    files = {
        "XGBoost": XGB_MODEL_PATH,
        "Autoencoder": AE_MODEL_PATH,
        "Autoencoder scaler": AE_SCALER_PATH,
        "Autoencoder schema": AE_SCHEMA_PATH,
    }

    for name, path in files.items():

        if path.exists():

            print(f"[OK] {name}")
            print(f"     {path}")

        else:

            print(f"[MISSING] {name}")
            print(f"          {path}")

            raise FileNotFoundError(
                f"\nRequired file missing:\n{path}"
            )

    print()
    print("All required model files found.")


# ============================================================
# LOAD MODELS
# ============================================================

def load_models():

    header("LOADING MODELS")

    # --------------------------------------------------------
    # XGBoost
    # --------------------------------------------------------

    section("LOADING XGBOOST")

    print(f"Path: {XGB_MODEL_PATH}")

    xgb_model = xgb.XGBClassifier()

    xgb_model.load_model(
        str(XGB_MODEL_PATH)
    )

    print("XGBoost loaded successfully.")

    # --------------------------------------------------------
    # Get EXACT XGBoost feature names
    # --------------------------------------------------------

    booster = xgb_model.get_booster()

    xgb_feature_names = booster.feature_names

    if not xgb_feature_names:

        raise ValueError(
            "The XGBoost model does not contain feature names."
        )

    print(
        f"XGBoost features: "
        f"{len(xgb_feature_names)}"
    )

    for i, feature in enumerate(
        xgb_feature_names,
        start=1
    ):

        print(
            f"  {i:2d}. {feature}"
        )

    # --------------------------------------------------------
    # Autoencoder
    # --------------------------------------------------------

    section("LOADING AUTOENCODER")

    print(f"Path: {AE_MODEL_PATH}")

    autoencoder = tf.keras.models.load_model(
        str(AE_MODEL_PATH),
        compile=False
    )

    print(
        "Autoencoder loaded successfully."
    )

    print(
        f"Input shape: "
        f"{autoencoder.input_shape}"
    )

    # --------------------------------------------------------
    # Scaler
    # --------------------------------------------------------

    section("LOADING AUTOENCODER SCALER")

    print(f"Path: {AE_SCALER_PATH}")

    scaler = joblib.load(
        AE_SCALER_PATH
    )

    print(
        "Autoencoder scaler loaded successfully."
    )

    scaler_dim = getattr(
        scaler,
        "n_features_in_",
        None
    )

    print(
        f"Scaler features: "
        f"{scaler_dim}"
    )

    return (
        xgb_model,
        xgb_feature_names,
        autoencoder,
        scaler,
    )


# ============================================================
# LOAD AUTOENCODER SCHEMA
# ============================================================

def load_schema():

    section(
        "LOADING AUTOENCODER FEATURE SCHEMA"
    )

    print(
        f"Path: {AE_SCHEMA_PATH}"
    )

    with open(
        AE_SCHEMA_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        schema = json.load(f)

    print("Schema loaded.")

    print()
    print("Top-level schema keys:")

    for key in schema.keys():

        print(
            f"  {key}"
        )

    numeric = schema.get(
        "numeric_features",
        []
    )

    categorical = schema.get(
        "categorical_features",
        []
    )

    encoded = schema.get(
        "encoded_feature_names",
        []
    )

    print()
    print(
        f"Numeric features: "
        f"{len(numeric)}"
    )

    for feature in numeric:

        print(
            f"  {feature}"
        )

    print()
    print(
        f"Categorical features: "
        f"{len(categorical)}"
    )

    for feature in categorical:

        print(
            f"  {feature}"
        )

    print()
    print(
        f"Encoded features: "
        f"{len(encoded)}"
    )

    print(
        "First encoded features:"
    )

    for feature in encoded[:20]:

        print(
            f"  {feature}"
        )

    if len(encoded) > 20:

        print(
            f"  ... "
            f"{len(encoded) - 20} more"
        )

    return schema


# ============================================================
# VALIDATE MODELS
# ============================================================

def validate_models(
    autoencoder,
    scaler,
    schema,
    xgb_feature_names,
):

    section(
        "VALIDATING MODEL DIMENSIONS"
    )

    ae_dim = int(
        autoencoder.input_shape[-1]
    )

    scaler_dim = getattr(
        scaler,
        "n_features_in_",
        None
    )

    encoded_dim = len(
        schema.get(
            "encoded_feature_names",
            []
        )
    )

    xgb_dim = len(
        xgb_feature_names
    )

    print(
        f"Autoencoder expects: {ae_dim}"
    )

    print(
        f"Scaler expects     : {scaler_dim}"
    )

    print(
        f"Schema contains    : {encoded_dim}"
    )

    print(
        f"XGBoost expects    : {xgb_dim}"
    )

    if ae_dim != encoded_dim:

        raise ValueError(
            "\nAutoencoder/schema mismatch.\n"
            f"Autoencoder: {ae_dim}\n"
            f"Schema: {encoded_dim}"
        )

    if (
        scaler_dim is not None
        and scaler_dim != ae_dim
    ):

        raise ValueError(
            "\nAutoencoder/scaler mismatch.\n"
            f"Autoencoder: {ae_dim}\n"
            f"Scaler: {scaler_dim}"
        )

    print()
    print(
        "Dimension validation successful."
    )


# ============================================================
# LOAD SCENARIO
# ============================================================

def load_scenario(scenario):

    path = (
        DATA_DIR
        / f"scenario{scenario}"
        / "ctu13_features.csv"
    )

    print()
    print(
        f"Loading Scenario {scenario}: "
        f"{path}"
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Scenario file not found:\n{path}"
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


# ============================================================
# NUMERIC CLEANING
# ============================================================

def clean_numeric(
    dataframe
):

    X = dataframe.copy()

    for column in X.columns:

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce"
        )

    nan_before = int(
        X.isna().sum().sum()
    )

    values = X.to_numpy(
        dtype=np.float64,
        na_value=np.nan
    )

    inf_before = int(
        np.isinf(values).sum()
    )

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    for column in X.columns:

        median = X[column].median()

        if pd.isna(median):

            median = 0.0

        X[column] = (
            X[column]
            .fillna(median)
        )

    X = X.fillna(0.0)

    return (
        X.astype(np.float32),
        nan_before,
        inf_before,
    )


# ============================================================
# XGBOOST FEATURE PREPARATION
# ============================================================

def prepare_xgb_features(
    df,
    xgb_feature_names
):

    print(
        "Preparing XGBoost features..."
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Use EXACT model feature names/order.
    # --------------------------------------------------------

    output = pd.DataFrame(
        index=df.index
    )

    missing = []

    for feature in xgb_feature_names:

        if feature in df.columns:

            output[feature] = (
                df[feature]
            )

        else:

            missing.append(
                feature
            )

    # --------------------------------------------------------
    # CTU-13 aliases / derived mappings
    # --------------------------------------------------------

    alias_map = {

        "Duration":
            "Dur",

        "Protocol":
            "Proto",

        "SrcPkts":
            "TotPkts",

        "DstPkts":
            "TotPkts",

        "SrcRate":
            "PacketsPerSecond",

        "DstRate":
            "PacketsPerSecond",

        "SrcLoad":
            "SourceOutboundRatio",

        "DstLoad":
            "SourceOutboundRatio",

        "SrcLoss":
            "SrcByteRatio",

        "DstLoss":
            "DstByteRatio",

        "pLoss":
            "SrcByteRatio",

        "pRtx":
            "SrcByteRatio",

        "SrcWin":
            "SrcBytes",

        "DstWin":
            "DstBytes",

        "TcpRtt":
            "InterArrivalTime",

        "SynAck":
            "InterArrivalTime",

        "AckDat":
            "PairInterArrivalTime",

        "Mean":
            "AvgPacketSize",

        "StdDev":
            "AvgPacketSize",

        "Sum":
            "TotBytes",

        "Min":
            "AvgPacketSize",

        "Max":
            "AvgPacketSize",

        "sTtl":
            "sTos",

        "dTtl":
            "dTos",

        "sHops":
            "sTos",

        "dHops":
            "dTos",
    }

    for feature in list(missing):

        source = alias_map.get(
            feature
        )

        if (
            source is not None
            and source in df.columns
        ):

            output[feature] = (
                df[source]
            )

            missing.remove(
                feature
            )

    # --------------------------------------------------------
    # Derived packet counts
    # --------------------------------------------------------

    if (
        "SrcPkts" in missing
        and "TotPkts" in df.columns
    ):

        output["SrcPkts"] = (
            pd.to_numeric(
                df["TotPkts"],
                errors="coerce"
            ) * 0.5
        )

        missing.remove(
            "SrcPkts"
        )

    if (
        "DstPkts" in missing
        and "TotPkts" in df.columns
    ):

        output["DstPkts"] = (
            pd.to_numeric(
                df["TotPkts"],
                errors="coerce"
            ) * 0.5
        )

        missing.remove(
            "DstPkts"
        )

    # --------------------------------------------------------
    # Any remaining unavailable feature
    # --------------------------------------------------------

    if missing:

        print()
        print(
            "WARNING: Features unavailable "
            "in CTU-13 CSV:"
        )

        for feature in missing:

            print(
                f"  - {feature}"
            )

            output[feature] = 0.0

    # --------------------------------------------------------
    # THE CRITICAL FIX
    #
    # Reorder EXACTLY to model order.
    # --------------------------------------------------------

    output = output[
        xgb_feature_names
    ]

    X, nan_before, inf_before = (
        clean_numeric(output)
    )

    print(
        f"NaN values before cleaning : "
        f"{nan_before:,}"
    )

    print(
        f"Infinite values before cleaning : "
        f"{inf_before:,}"
    )

    print(
        f"XGBoost matrix: "
        f"{X.shape[0]:,} x "
        f"{X.shape[1]}"
    )

    return X


# ============================================================
# AUTOENCODER MATRIX
# ============================================================

def build_autoencoder_matrix(
    df,
    schema
):

    numeric_features = schema[
        "numeric_features"
    ]

    categorical_features = schema[
        "categorical_features"
    ]

    encoded_features = schema[
        "encoded_feature_names"
    ]

    result = pd.DataFrame(
        0.0,
        index=df.index,
        columns=encoded_features,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Numeric features
    # --------------------------------------------------------

    for feature in numeric_features:

        if feature not in df.columns:

            print(
                f"WARNING: Missing numeric "
                f"feature: {feature}"
            )

            continue

        result[feature] = pd.to_numeric(
            df[feature],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Categorical features
    # --------------------------------------------------------

    for column in categorical_features:

        if column not in df.columns:

            continue

        values = (
            df[column]
            .astype("string")
            .fillna("__MISSING__")
            .astype(str)
        )

        for encoded_name in encoded_features:

            # Standard OneHotEncoder naming:
            #
            # Proto_TCP
            # Proto_UDP
            # Dir_<
            # State_SF

            prefix = (
                f"{column}_"
            )

            if not encoded_name.startswith(
                prefix
            ):

                continue

            category = (
                encoded_name[
                    len(prefix):
                ]
            )

            result[
                encoded_name
            ] = (
                values == category
            ).astype(
                np.float32
            )

    # --------------------------------------------------------
    # Handle alternate Protocol naming
    # --------------------------------------------------------

    if "Proto" in df.columns:

        values = (
            df["Proto"]
            .astype("string")
            .fillna("__MISSING__")
            .astype(str)
        )

        for encoded_name in encoded_features:

            if encoded_name.startswith(
                "Protocol_"
            ):

                category = (
                    encoded_name[
                        len("Protocol_"):
                    ]
                )

                result[
                    encoded_name
                ] = (
                    values == category
                ).astype(
                    np.float32
                )

    # --------------------------------------------------------
    # Clean
    # --------------------------------------------------------

    result = result.replace(
        [np.inf, -np.inf],
        np.nan
    )

    result = result.fillna(
        0.0
    )

    return result.astype(
        np.float32
    )


# ============================================================
# RECONSTRUCTION ERROR
# ============================================================

def reconstruction_error(
    model,
    X
):

    reconstructed = model.predict(
        X,
        batch_size=BATCH_SIZE,
        verbose=0
    )

    error = np.mean(
        np.square(
            X - reconstructed
        ),
        axis=1
    )

    return error


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    predictions,
    scores
):

    metrics = {}

    metrics["Accuracy"] = float(
        accuracy_score(
            y_true,
            predictions
        )
    )

    metrics["Precision"] = float(
        precision_score(
            y_true,
            predictions,
            zero_division=0
        )
    )

    metrics["Recall"] = float(
        recall_score(
            y_true,
            predictions,
            zero_division=0
        )
    )

    metrics["F1"] = float(
        f1_score(
            y_true,
            predictions,
            zero_division=0
        )
    )

    try:

        metrics["ROC_AUC"] = float(
            roc_auc_score(
                y_true,
                scores
            )
        )

    except Exception:

        metrics["ROC_AUC"] = 0.0

    try:

        metrics["PR_AUC"] = float(
            average_precision_score(
                y_true,
                scores
            )
        )

    except Exception:

        metrics["PR_AUC"] = 0.0

    return metrics


# ============================================================
# PRINT METRICS
# ============================================================

def print_metrics(
    name,
    metrics
):

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


# ============================================================
# MAIN
# ============================================================

def main():

    np.random.seed(
        RANDOM_SEED
    )

    tf.random.set_seed(
        RANDOM_SEED
    )

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    header(
        "CTU-13 XGBOOST + AUTOENCODER "
        "GATED HYBRID INTRUSION DETECTION"
    )

    print("Configuration:")

    print(
        "  Training scenarios : 1 - 10"
    )

    print(
        "  Testing scenarios  : 11 - 13"
    )

    print(
        f"  XGBoost threshold   : "
        f"{XGB_THRESHOLD:.2f}"
    )

    print(
        f"  Autoencoder threshold: "
        f"{AE_THRESHOLD:.8f}"
    )

    print(
        f"  Uncertainty range   : "
        f"{UNCERTAINTY_LOW:.2f} - "
        f"{UNCERTAINTY_HIGH:.2f}"
    )

    print(
        "  Hybrid mode         : GATED"
    )

    print(
        f"  XGBoost weight      : "
        f"{XGB_WEIGHT:.2f}"
    )

    print(
        f"  Autoencoder weight  : "
        f"{AE_WEIGHT:.2f}"
    )

    # --------------------------------------------------------
    # CHECK FILES
    # --------------------------------------------------------

    check_model_files()

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    (
        xgb_model,
        xgb_feature_names,
        autoencoder,
        scaler,
    ) = load_models()

    # --------------------------------------------------------
    # SCHEMA
    # --------------------------------------------------------

    schema = load_schema()

    validate_models(
        autoencoder,
        scaler,
        schema,
        xgb_feature_names
    )

    # --------------------------------------------------------
    # LOAD TEST DATA
    # --------------------------------------------------------

    header(
        "LOADING UNSEEN TEST SCENARIOS"
    )

    scenario_data = {}

    total_rows = 0

    for scenario in TEST_SCENARIOS:

        df = load_scenario(
            scenario
        )

        scenario_data[
            scenario
        ] = df

        total_rows += len(df)

    print()
    print(
        f"TOTAL TEST ROWS: "
        f"{total_rows:,}"
    )

    # --------------------------------------------------------
    # Target distribution
    # --------------------------------------------------------

    header(
        "TEST TARGET DISTRIBUTION"
    )

    all_targets = pd.concat(
        [
            df["Target"]
            for df in scenario_data.values()
        ],
        ignore_index=True
    )

    print(
        all_targets.value_counts()
        .sort_index()
    )

    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    overall_y = []

    overall_xgb_pred = []
    overall_xgb_score = []

    overall_ae_pred = []
    overall_ae_score = []

    overall_hybrid_pred = []
    overall_hybrid_score = []

    all_predictions = []

    per_scenario = []

    # --------------------------------------------------------
    # Process scenarios
    # --------------------------------------------------------

    header(
        "HYBRID PREDICTION"
    )

    for scenario, df in scenario_data.items():

        section(
            f"PROCESSING SCENARIO {scenario}"
        )

        print(
            f"Rows: {len(df):,}"
        )

        y_true = (
            pd.to_numeric(
                df["Target"],
                errors="coerce"
            )
            .fillna(0)
            .astype(int)
            .to_numpy()
        )

        # ====================================================
        # XGBOOST
        # ====================================================

        X_xgb = prepare_xgb_features(
            df,
            xgb_feature_names
        )

        print()
        print(
            "Running XGBoost..."
        )

        xgb_probability = (
            xgb_model
            .predict_proba(
                X_xgb
            )[:, 1]
        )

        xgb_prediction = (
            xgb_probability
            >= XGB_THRESHOLD
        ).astype(int)

        # ====================================================
        # AUTOENCODER
        # ====================================================

        print()
        print(
            "Preparing Autoencoder features..."
        )

        X_ae = (
            build_autoencoder_matrix(
                df,
                schema
            )
        )

        print(
            f"Autoencoder matrix: "
            f"{X_ae.shape[0]:,} x "
            f"{X_ae.shape[1]}"
        )

        print(
            "Scaling Autoencoder features..."
        )

        X_ae_scaled = (
            scaler.transform(
                X_ae
            )
        )

        X_ae_scaled = np.nan_to_num(
            X_ae_scaled,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        ).astype(
            np.float32
        )

        print(
            "Running Autoencoder..."
        )

        ae_error = (
            reconstruction_error(
                autoencoder,
                X_ae_scaled
            )
        )

        ae_prediction = (
            ae_error
            >= AE_THRESHOLD
        ).astype(int)

        # ----------------------------------------------------
        # Normalize AE score
        # ----------------------------------------------------

        ae_score = (
            ae_error
            / max(
                AE_THRESHOLD,
                1e-12
            )
        )

        ae_score = np.clip(
            ae_score,
            0.0,
            1.0
        )

        # ====================================================
        # GATED HYBRID
        # ====================================================

        hybrid_prediction = (
            xgb_prediction.copy()
        )

        hybrid_score = (
            xgb_probability.copy()
        )

        uncertain = (
            (
                xgb_probability
                >= UNCERTAINTY_LOW
            )
            &
            (
                xgb_probability
                <= UNCERTAINTY_HIGH
            )
        )

        uncertain_count = int(
            uncertain.sum()
        )

        print()
        print(
            f"XGBoost uncertain rows: "
            f"{uncertain_count:,}"
        )

        if uncertain.any():

            blended = (
                XGB_WEIGHT
                * xgb_probability[
                    uncertain
                ]
                +
                AE_WEIGHT
                * ae_score[
                    uncertain
                ]
            )

            hybrid_score[
                uncertain
            ] = blended

            hybrid_prediction[
                uncertain
            ] = (
                blended
                >= HYBRID_THRESHOLD
            ).astype(int)

        # ====================================================
        # METRICS
        # ====================================================

        xgb_metrics = calculate_metrics(
            y_true,
            xgb_prediction,
            xgb_probability
        )

        ae_metrics = calculate_metrics(
            y_true,
            ae_prediction,
            ae_error
        )

        hybrid_metrics = calculate_metrics(
            y_true,
            hybrid_prediction,
            hybrid_score
        )

        print_metrics(
            "XGBoost",
            xgb_metrics
        )

        print_metrics(
            "Autoencoder",
            ae_metrics
        )

        print_metrics(
            "GATED HYBRID",
            hybrid_metrics
        )

        # ====================================================
        # CONFUSION MATRIX
        # ====================================================

        cm = confusion_matrix(
            y_true,
            hybrid_prediction,
            labels=[0, 1]
        )

        tn, fp, fn, tp = (
            cm.ravel()
        )

        print()
        print(
            "HYBRID CONFUSION MATRIX"
        )

        print("-" * 40)

        print(cm)

        print()

        print(
            f"True Negatives : "
            f"{tn:,}"
        )

        print(
            f"False Positives: "
            f"{fp:,}"
        )

        print(
            f"False Negatives: "
            f"{fn:,}"
        )

        print(
            f"True Positives : "
            f"{tp:,}"
        )

        # ====================================================
        # DISTRIBUTION
        # ====================================================

        print()
        print(
            "PREDICTION DISTRIBUTION"
        )

        print(
            f"XGBoost ATTACK      : "
            f"{xgb_prediction.sum():,}"
        )

        print(
            f"Autoencoder ANOMALY : "
            f"{ae_prediction.sum():,}"
        )

        print(
            f"Hybrid ATTACK       : "
            f"{hybrid_prediction.sum():,}"
        )

        # ====================================================
        # SAVE SCENARIO RESULTS
        # ====================================================

        per_scenario.extend(
            [
                {
                    "Scenario": scenario,
                    "Model": "XGBoost",
                    **xgb_metrics,
                },
                {
                    "Scenario": scenario,
                    "Model": "Autoencoder",
                    **ae_metrics,
                },
                {
                    "Scenario": scenario,
                    "Model": "Hybrid",
                    **hybrid_metrics,
                },
            ]
        )

        scenario_predictions = pd.DataFrame(
            {
                "Scenario": scenario,
                "Target": y_true,

                "XGBoostProbability":
                    xgb_probability,

                "XGBoostPrediction":
                    xgb_prediction,

                "AutoencoderError":
                    ae_error,

                "AutoencoderScore":
                    ae_score,

                "AutoencoderPrediction":
                    ae_prediction,

                "XGBoostUncertain":
                    uncertain.astype(int),

                "HybridScore":
                    hybrid_score,

                "HybridPrediction":
                    hybrid_prediction,
            }
        )

        all_predictions.append(
            scenario_predictions
        )

        # ====================================================
        # OVERALL
        # ====================================================

        overall_y.extend(
            y_true.tolist()
        )

        overall_xgb_pred.extend(
            xgb_prediction.tolist()
        )

        overall_xgb_score.extend(
            xgb_probability.tolist()
        )

        overall_ae_pred.extend(
            ae_prediction.tolist()
        )

        overall_ae_score.extend(
            ae_error.tolist()
        )

        overall_hybrid_pred.extend(
            hybrid_prediction.tolist()
        )

        overall_hybrid_score.extend(
            hybrid_score.tolist()
        )

    # ========================================================
    # ARRAYS
    # ========================================================

    overall_y = np.asarray(
        overall_y
    )

    overall_xgb_pred = np.asarray(
        overall_xgb_pred
    )

    overall_xgb_score = np.asarray(
        overall_xgb_score
    )

    overall_ae_pred = np.asarray(
        overall_ae_pred
    )

    overall_ae_score = np.asarray(
        overall_ae_score
    )

    overall_hybrid_pred = np.asarray(
        overall_hybrid_pred
    )

    overall_hybrid_score = np.asarray(
        overall_hybrid_score
    )

    # ========================================================
    # OVERALL METRICS
    # ========================================================

    header(
        "OVERALL HYBRID RESULTS"
    )

    xgb_overall = calculate_metrics(
        overall_y,
        overall_xgb_pred,
        overall_xgb_score
    )

    ae_overall = calculate_metrics(
        overall_y,
        overall_ae_pred,
        overall_ae_score
    )

    hybrid_overall = calculate_metrics(
        overall_y,
        overall_hybrid_pred,
        overall_hybrid_score
    )

    print_metrics(
        "XGBoost",
        xgb_overall
    )

    print_metrics(
        "Autoencoder",
        ae_overall
    )

    print_metrics(
        "GATED XGBOOST + AUTOENCODER",
        hybrid_overall
    )

    # ========================================================
    # OVERALL CONFUSION MATRICES
    # ========================================================

    header(
        "OVERALL CONFUSION MATRICES"
    )

    xgb_cm = confusion_matrix(
        overall_y,
        overall_xgb_pred,
        labels=[0, 1]
    )

    ae_cm = confusion_matrix(
        overall_y,
        overall_ae_pred,
        labels=[0, 1]
    )

    hybrid_cm = confusion_matrix(
        overall_y,
        overall_hybrid_pred,
        labels=[0, 1]
    )

    print("XGBoost:")
    print(xgb_cm)

    print()
    print("Autoencoder:")
    print(ae_cm)

    print()
    print("Hybrid:")
    print(hybrid_cm)

    # ========================================================
    # IMPROVEMENT
    # ========================================================

    recall_change = (
        hybrid_overall["Recall"]
        -
        xgb_overall["Recall"]
    )

    precision_change = (
        hybrid_overall["Precision"]
        -
        xgb_overall["Precision"]
    )

    f1_change = (
        hybrid_overall["F1"]
        -
        xgb_overall["F1"]
    )

    header(
        "HYBRID CHANGE VS XGBOOST"
    )

    print(
        f"Recall change    : "
        f"{recall_change:+.4f}"
    )

    print(
        f"Precision change : "
        f"{precision_change:+.4f}"
    )

    print(
        f"F1 change        : "
        f"{f1_change:+.4f}"
    )

    # ========================================================
    # PER-SCENARIO
    # ========================================================

    header(
        "PER-SCENARIO SUMMARY"
    )

    per_scenario_df = pd.DataFrame(
        per_scenario
    )

    print(
        per_scenario_df.to_string(
            index=False
        )
    )

    # ========================================================
    # SAVE
    # ========================================================

    header(
        "SAVING RESULTS"
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Overall
    # --------------------------------------------------------

    overall_df = pd.DataFrame(
        [
            {
                "Model": "XGBoost",
                **xgb_overall,
            },
            {
                "Model": "Autoencoder",
                **ae_overall,
            },
            {
                "Model": "GatedHybrid",
                **hybrid_overall,
            },
        ]
    )

    overall_path = (
        MODEL_DIR
        / "autoencoder_xgb_hybrid_benchmark.csv"
    )

    overall_df.to_csv(
        overall_path,
        index=False
    )

    print(
        f"Overall results:\n"
        f"  {overall_path}"
    )

    # --------------------------------------------------------
    # Per scenario
    # --------------------------------------------------------

    per_scenario_path = (
        MODEL_DIR
        / "autoencoder_xgb_hybrid_per_scenario.csv"
    )

    per_scenario_df.to_csv(
        per_scenario_path,
        index=False
    )

    print(
        f"Per-scenario results:\n"
        f"  {per_scenario_path}"
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    predictions_df = pd.concat(
        all_predictions,
        ignore_index=True
    )

    prediction_path = (
        MODEL_DIR
        / "autoencoder_xgb_hybrid_predictions.csv"
    )

    predictions_df.to_csv(
        prediction_path,
        index=False
    )

    print(
        f"Predictions:\n"
        f"  {prediction_path}"
    )

    # --------------------------------------------------------
    # Config
    # --------------------------------------------------------

    config = {

        "project":
            "CTU-13",

        "training_scenarios":
            "1-10",

        "testing_scenarios":
            "11-13",

        "random_seed":
            RANDOM_SEED,

        "xgboost_threshold":
            XGB_THRESHOLD,

        "autoencoder_threshold":
            AE_THRESHOLD,

        "uncertainty_low":
            UNCERTAINTY_LOW,

        "uncertainty_high":
            UNCERTAINTY_HIGH,

        "xgboost_weight":
            XGB_WEIGHT,

        "autoencoder_weight":
            AE_WEIGHT,

        "hybrid_threshold":
            HYBRID_THRESHOLD,

        "xgboost_features":
            list(xgb_feature_names),

        "autoencoder_features":
            len(
                schema[
                    "encoded_feature_names"
                ]
            ),

        "test_rows":
            int(
                len(predictions_df)
            ),

        "xgboost_metrics":
            xgb_overall,

        "autoencoder_metrics":
            ae_overall,

        "hybrid_metrics":
            hybrid_overall,

        "change_vs_xgboost": {

            "recall":
                float(
                    recall_change
                ),

            "precision":
                float(
                    precision_change
                ),

            "f1":
                float(
                    f1_change
                ),
        },
    }

    config_path = (
        MODEL_DIR
        / "autoencoder_xgb_hybrid_config.json"
    )

    with open(
        config_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            config,
            f,
            indent=2,
            allow_nan=False
        )

    print(
        f"Configuration:\n"
        f"  {config_path}"
    )

    # ========================================================
    # FINAL
    # ========================================================

    header(
        "AUTOENCODER + XGBOOST HYBRID COMPLETE"
    )

    print(
        f"Test rows: "
        f"{len(predictions_df):,}"
    )

    print()

    print(
        f"XGBoost F1     : "
        f"{xgb_overall['F1']:.4f}"
    )

    print(
        f"Autoencoder F1 : "
        f"{ae_overall['F1']:.4f}"
    )

    print(
        f"Hybrid F1      : "
        f"{hybrid_overall['F1']:.4f}"
    )

    print()

    print(
        f"XGBoost Recall  : "
        f"{xgb_overall['Recall']:.4f}"
    )

    print(
        f"Hybrid Recall   : "
        f"{hybrid_overall['Recall']:.4f}"
    )

    print()

    print(
        "DONE"
    )

    print(
        "=" * 70
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()