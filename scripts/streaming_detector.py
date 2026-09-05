"""
CTU-13 STREAMING HYBRID INTRUSION DETECTOR

Architecture:

    CTU-13 CSV
          |
          v
    Read small chunks
          |
          +----------------------+
          |                      |
          v                      v
      XGBoost               Autoencoder
      30 features           257 features
          |                      |
          v                      v
    Attack probability    Reconstruction error
          |                      |
          +----------+-----------+
                     |
                     v
              Hybrid decision
                     |
                     v
                Alert engine
                     |
                     v
              JSONL / CSV output

Important:
- XGBoost uses the exact feature order stored in the model.
- Autoencoder uses autoencoder_features.json.
- Autoencoder uses the saved scaler.
- Data is processed in chunks.
- The complete test dataset is NOT loaded into memory.
"""

import json
import warnings
from pathlib import Path
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

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

OUTPUT_JSONL = (
    OUTPUT_DIR / "streaming_alerts.jsonl"
)

OUTPUT_CSV = (
    OUTPUT_DIR / "streaming_results.csv"
)

OUTPUT_SUMMARY = (
    OUTPUT_DIR / "streaming_summary.json"
)


# ============================================================
# CONFIGURATION
# ============================================================

TEST_SCENARIOS = [11, 12, 13]

CHUNK_SIZE = 5000

PREDICTION_BATCH_SIZE = 2048

XGB_THRESHOLD = 0.20

AE_THRESHOLD = 0.25541964

# XGBoost is the primary detector.
XGB_WEIGHT = 0.90

# Autoencoder provides anomaly evidence.
AE_WEIGHT = 0.10

# If XGBoost is clearly confident, trust it.
# If it is uncertain, use AE evidence.
UNCERTAINTY_LOW = 0.05
UNCERTAINTY_HIGH = 0.35

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)


# ============================================================
# RAW AUTOENCODER FEATURES
# ============================================================

DEFAULT_NUMERIC_FEATURES = [
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


DEFAULT_CATEGORICAL_FEATURES = [
    "Proto",
    "Dir",
    "State",
]


# ============================================================
# DISPLAY
# ============================================================

def banner(title):
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
# CHECK FILES
# ============================================================

def check_files():

    banner("CHECKING REQUIRED FILES")

    required = {
        "XGBoost model": XGB_MODEL_PATH,
        "Autoencoder": AE_MODEL_PATH,
        "Autoencoder scaler": AE_SCALER_PATH,
        "Autoencoder schema": AE_SCHEMA_PATH,
    }

    missing = []

    for name, path in required.items():

        if path.exists():

            print(f"[OK] {name}")
            print(f"     {path}")

        else:

            print(f"[MISSING] {name}")
            print(f"          {path}")

            missing.append(name)

    if missing:

        raise FileNotFoundError(
            "\nMissing required model files:\n"
            + "\n".join(
                f"  - {name}"
                for name in missing
            )
        )


# ============================================================
# LOAD XGBOOST
# ============================================================

def load_xgboost():

    section("LOADING XGBOOST")

    print(f"Path: {XGB_MODEL_PATH}")

    import xgboost as xgb

    model = xgb.XGBClassifier()

    model.load_model(
        str(XGB_MODEL_PATH)
    )

    print("XGBoost loaded successfully.")

    # --------------------------------------------------------
    # Get exact feature names from model
    # --------------------------------------------------------

    feature_names = None

    try:

        booster = model.get_booster()

        feature_names = booster.feature_names

    except Exception:

        feature_names = None

    if not feature_names:

        raise ValueError(
            "Could not obtain XGBoost feature names "
            "from the trained model."
        )

    feature_names = [
        str(x)
        for x in feature_names
    ]

    print(
        f"XGBoost features: {len(feature_names)}"
    )

    for i, feature in enumerate(
        feature_names,
        start=1
    ):

        print(
            f"  {i:2d}. {feature}"
        )

    return model, feature_names


# ============================================================
# LOAD AUTOENCODER
# ============================================================

def load_autoencoder():

    section("LOADING AUTOENCODER")

    print(f"Path: {AE_MODEL_PATH}")

    import tensorflow as tf

    model = tf.keras.models.load_model(
        str(AE_MODEL_PATH),
        compile=False
    )

    print(
        "Autoencoder loaded successfully."
    )

    print(
        f"Input shape: {model.input_shape}"
    )

    return model


# ============================================================
# LOAD SCALER
# ============================================================

def load_scaler():

    section("LOADING AUTOENCODER SCALER")

    print(
        f"Path: {AE_SCALER_PATH}"
    )

    scaler = joblib.load(
        AE_SCALER_PATH
    )

    print(
        "Autoencoder scaler loaded successfully."
    )

    dimension = getattr(
        scaler,
        "n_features_in_",
        None
    )

    if dimension is not None:

        print(
            f"Scaler features: {dimension}"
        )

    else:

        print(
            "Scaler does not expose "
            "n_features_in_."
        )

    return scaler


# ============================================================
# LOAD AUTOENCODER SCHEMA
# ============================================================

def load_schema():

    section("LOADING AUTOENCODER SCHEMA")

    print(
        f"Path: {AE_SCHEMA_PATH}"
    )

    with open(
        AE_SCHEMA_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        schema = json.load(file)

    if not isinstance(schema, dict):

        raise ValueError(
            "Autoencoder schema must be a JSON object."
        )

    numeric_features = schema.get(
        "numeric_features",
        DEFAULT_NUMERIC_FEATURES
    )

    categorical_features = schema.get(
        "categorical_features",
        DEFAULT_CATEGORICAL_FEATURES
    )

    encoded_features = (
        schema.get("encoded_feature_names")
        or schema.get("encoded_features")
        or schema.get("feature_names")
    )

    if not isinstance(
        numeric_features,
        list
    ):

        raise ValueError(
            "Invalid numeric_features."
        )

    if not isinstance(
        categorical_features,
        list
    ):

        raise ValueError(
            "Invalid categorical_features."
        )

    if not isinstance(
        encoded_features,
        list
    ):

        raise ValueError(
            "Could not find encoded feature names "
            "in autoencoder_features.json."
        )

    numeric_features = [
        str(x)
        for x in numeric_features
    ]

    categorical_features = [
        str(x)
        for x in categorical_features
    ]

    encoded_features = [
        str(x)
        for x in encoded_features
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

    print("\nCategorical columns:")

    for feature in categorical_features:

        print(
            f"  {feature}"
        )

    return {
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "encoded_features": encoded_features,
    }


# ============================================================
# VALIDATE MODELS
# ============================================================

def validate_models(
    xgb_model,
    xgb_features,
    ae_model,
    scaler,
    schema
):

    section("VALIDATING MODEL DIMENSIONS")

    # --------------------------------------------------------
    # XGBoost
    # --------------------------------------------------------

    xgb_dimension = len(
        xgb_features
    )

    print(
        f"XGBoost expects     : "
        f"{xgb_dimension}"
    )

    if xgb_dimension != 30:

        print(
            "WARNING: XGBoost model does not "
            "contain exactly 30 features."
        )

    # --------------------------------------------------------
    # Autoencoder
    # --------------------------------------------------------

    ae_shape = ae_model.input_shape

    ae_dimension = int(
        ae_shape[-1]
    )

    scaler_dimension = getattr(
        scaler,
        "n_features_in_",
        None
    )

    schema_dimension = len(
        schema["encoded_features"]
    )

    print(
        f"Autoencoder expects : "
        f"{ae_dimension}"
    )

    print(
        f"Scaler expects      : "
        f"{scaler_dimension}"
    )

    print(
        f"Schema contains     : "
        f"{schema_dimension}"
    )

    if ae_dimension != schema_dimension:

        raise ValueError(
            "\nAutoencoder/schema dimension mismatch.\n"
            f"Model: {ae_dimension}\n"
            f"Schema: {schema_dimension}"
        )

    if (
        scaler_dimension is not None
        and scaler_dimension != ae_dimension
    ):

        raise ValueError(
            "\nAutoencoder/scaler dimension mismatch.\n"
            f"Model: {ae_dimension}\n"
            f"Scaler: {scaler_dimension}"
        )

    print(
        "\nDimension validation successful."
    )


# ============================================================
# CLEAN NUMERIC SERIES
# ============================================================

def clean_numeric_series(series):

    values = pd.to_numeric(
        series,
        errors="coerce"
    )

    values = values.replace(
        [np.inf, -np.inf],
        np.nan
    )

    return values


# ============================================================
# PREPARE XGBOOST FEATURES
# ============================================================

def prepare_xgb_features(
    df,
    feature_names
):

    missing = [
        feature
        for feature in feature_names
        if feature not in df.columns
    ]

    if missing:

        raise ValueError(
            "\nMissing XGBoost features:\n"
            + "\n".join(
                f"  - {feature}"
                for feature in missing
            )
        )

    # --------------------------------------------------------
    # CRITICAL:
    # Use EXACT model feature order.
    # --------------------------------------------------------

    X = df.loc[
        :,
        feature_names
    ].copy()

    for column in feature_names:

        X[column] = clean_numeric_series(
            X[column]
        )

        median = X[column].median()

        if pd.isna(median):

            median = 0.0

        X[column] = (
            X[column]
            .fillna(median)
        )

    X = X.astype(
        np.float32
    )

    return X


# ============================================================
# EXTRACT CATEGORY FROM ENCODED NAME
# ============================================================

def extract_category(
    encoded_name,
    categorical_column
):

    name = str(
        encoded_name
    )

    column = str(
        categorical_column
    )

    prefixes = [
        column + "_",
        column + "=",
        column + ":",
        column + "[",
    ]

    for prefix in prefixes:

        if name.startswith(prefix):

            value = name[
                len(prefix):
            ]

            if value.endswith("]"):

                value = value[:-1]

            return value

    return None


# ============================================================
# BUILD AUTOENCODER MATRIX
# ============================================================

def build_autoencoder_matrix(
    df,
    schema
):

    numeric_features = (
        schema["numeric_features"]
    )

    categorical_features = (
        schema["categorical_features"]
    )

    encoded_features = (
        schema["encoded_features"]
    )

    # --------------------------------------------------------
    # Validate raw columns
    # --------------------------------------------------------

    required = (
        list(numeric_features)
        + list(categorical_features)
    )

    missing = [
        feature
        for feature in required
        if feature not in df.columns
    ]

    if missing:

        raise ValueError(
            "\nMissing Autoencoder raw features:\n"
            + "\n".join(
                f"  - {feature}"
                for feature in missing
            )
        )

    # --------------------------------------------------------
    # Numeric data
    # --------------------------------------------------------

    numeric_data = {}

    for feature in numeric_features:

        values = clean_numeric_series(
            df[feature]
        )

        median = values.median()

        if pd.isna(median):

            median = 0.0

        values = values.fillna(
            median
        )

        numeric_data[feature] = (
            values.astype(
                np.float32
            )
        )

    # --------------------------------------------------------
    # Categorical data
    # --------------------------------------------------------

    categorical_data = {}

    for feature in categorical_features:

        values = (
            df[feature]
            .astype("string")
            .fillna("")
            .astype(str)
        )

        categorical_data[feature] = (
            values
        )

    # --------------------------------------------------------
    # Create encoded matrix
    # --------------------------------------------------------

    result = np.zeros(
        (
            len(df),
            len(encoded_features)
        ),
        dtype=np.float32
    )

    numeric_set = set(
        numeric_features
    )

    # --------------------------------------------------------
    # Process every encoded column
    # --------------------------------------------------------

    for index, encoded_name in enumerate(
        encoded_features
    ):

        # ----------------------------------------------------
        # Numeric feature
        # ----------------------------------------------------

        if encoded_name in numeric_set:

            result[:, index] = (
                numeric_data[
                    encoded_name
                ].to_numpy(
                    dtype=np.float32
                )
            )

            continue

        # ----------------------------------------------------
        # Categorical feature
        # ----------------------------------------------------

        matched_column = None
        category_value = None

        for column in categorical_features:

            value = extract_category(
                encoded_name,
                column
            )

            if value is not None:

                matched_column = column
                category_value = value

                break

        if matched_column is None:

            raise ValueError(
                "\nCould not decode Autoencoder "
                f"feature: {encoded_name}\n"
                "Check autoencoder_features.json."
            )

        result[:, index] = (
            categorical_data[
                matched_column
            ]
            .eq(category_value)
            .to_numpy(
                dtype=np.float32
            )
        )

    return pd.DataFrame(
        result,
        columns=encoded_features,
        index=df.index
    )


# ============================================================
# PREPARE AUTOENCODER FEATURES
# ============================================================

def prepare_autoencoder_features(
    df,
    schema,
    scaler
):

    X = build_autoencoder_matrix(
        df,
        schema
    )

    X_values = X.to_numpy(
        dtype=np.float32
    )

    X_scaled = scaler.transform(
        X_values
    )

    X_scaled = np.asarray(
        X_scaled,
        dtype=np.float32
    )

    X_scaled = np.nan_to_num(
        X_scaled,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    return X_scaled


# ============================================================
# AUTOENCODER RECONSTRUCTION ERROR
# ============================================================

def calculate_reconstruction_error(
    model,
    X
):

    errors = []

    for start in range(
        0,
        len(X),
        PREDICTION_BATCH_SIZE
    ):

        end = min(
            start
            + PREDICTION_BATCH_SIZE,
            len(X)
        )

        batch = X[
            start:end
        ]

        reconstructed = model.predict(
            batch,
            batch_size=PREDICTION_BATCH_SIZE,
            verbose=0
        )

        error = np.mean(
            np.square(
                batch
                - reconstructed
            ),
            axis=1
        )

        errors.append(
            error.astype(
                np.float32
            )
        )

    if not errors:

        return np.empty(
            0,
            dtype=np.float32
        )

    return np.concatenate(
        errors
    )


# ============================================================
# NORMALIZE AE ANOMALY SCORE
# ============================================================

def autoencoder_score(
    reconstruction_errors
):

    """
    Convert reconstruction error into a bounded
    0-1 anomaly score.

    The trained AE threshold represents the point
    at which a flow becomes anomalous.

    score = error / threshold

    Values above threshold approach 1.
    """

    scores = (
        reconstruction_errors
        / max(
            AE_THRESHOLD,
            1e-12
        )
    )

    scores = np.clip(
        scores,
        0.0,
        1.0
    )

    return scores.astype(
        np.float32
    )


# ============================================================
# HYBRID DECISION
# ============================================================

def calculate_hybrid_scores(
    xgb_probability,
    ae_score
):

    # --------------------------------------------------------
    # Weighted evidence
    # --------------------------------------------------------

    weighted_score = (
        XGB_WEIGHT
        * xgb_probability
        +
        AE_WEIGHT
        * ae_score
    )

    # --------------------------------------------------------
    # GATED LOGIC
    #
    # XGBoost remains primary.
    # AE contributes especially when XGBoost is uncertain.
    # --------------------------------------------------------

    hybrid_score = weighted_score.copy()

    uncertain = (
        (xgb_probability >= UNCERTAINTY_LOW)
        &
        (xgb_probability <= UNCERTAINTY_HIGH)
    )

    # When XGBoost is uncertain, increase the influence
    # of the AE anomaly signal.
    hybrid_score[uncertain] = (
        0.60
        * xgb_probability[uncertain]
        +
        0.40
        * ae_score[uncertain]
    )

    hybrid_score = np.clip(
        hybrid_score,
        0.0,
        1.0
    )

    return hybrid_score.astype(
        np.float32
    )


# ============================================================
# THREAT CLASSIFICATION
# ============================================================

def classify_threat(
    xgb_probability,
    ae_score,
    hybrid_score
):

    result = []

    for xgb_p, ae_p, hybrid_p in zip(
        xgb_probability,
        ae_score,
        hybrid_score
    ):

        if hybrid_p >= 0.70:

            result.append(
                "MALICIOUS"
            )

        elif hybrid_p >= 0.35:

            result.append(
                "SUSPICIOUS"
            )

        else:

            result.append(
                "BENIGN"
            )

    return np.array(
        result,
        dtype=object
    )


# ============================================================
# SEVERITY
# ============================================================

def calculate_severity(
    threat,
    score
):

    if threat == "MALICIOUS":

        if score >= 0.90:

            return "CRITICAL"

        if score >= 0.80:

            return "HIGH"

        return "MEDIUM"

    if threat == "SUSPICIOUS":

        return "MEDIUM"

    return "LOW"


# ============================================================
# DETECTION REASONS
# ============================================================

def generate_reason(
    xgb_probability,
    ae_score,
    ae_error,
    threat
):

    reasons = []

    if xgb_probability >= XGB_THRESHOLD:

        reasons.append(
            "XGBoost detected elevated attack probability"
        )

    if ae_error >= AE_THRESHOLD:

        reasons.append(
            "Autoencoder detected anomalous flow reconstruction"
        )

    if (
        xgb_probability >= UNCERTAINTY_LOW
        and
        xgb_probability <= UNCERTAINTY_HIGH
    ):

        if ae_score >= 0.50:

            reasons.append(
                "Autoencoder provided additional anomaly evidence"
            )

    if not reasons:

        reasons.append(
            "No strong malicious indicators detected"
        )

    return reasons


# ============================================================
# SAFE VALUE
# ============================================================

def safe_value(value):

    if pd.isna(value):

        return None

    if isinstance(
        value,
        (np.integer,)
    ):

        return int(value)

    if isinstance(
        value,
        (np.floating,)
    ):

        return float(value)

    return value


# ============================================================
# CREATE ALERT
# ============================================================

def create_alert(
    row,
    scenario,
    xgb_probability,
    ae_score,
    ae_error,
    hybrid_score,
    threat
):

    src_ip = safe_value(
        row.get("SrcAddr")
    )

    dst_ip = safe_value(
        row.get("DstAddr")
    )

    protocol = safe_value(
        row.get("Proto")
    )

    direction = safe_value(
        row.get("Dir")
    )

    state = safe_value(
        row.get("State")
    )

    src_port = safe_value(
        row.get("Sport")
    )

    dst_port = safe_value(
        row.get("Dport")
    )

    severity = calculate_severity(
        threat,
        hybrid_score
    )

    reasons = generate_reason(
        xgb_probability,
        ae_score,
        ae_error,
        threat
    )

    alert = {

        "timestamp":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "scenario":
            int(scenario),

        "source_ip":
            src_ip,

        "destination_ip":
            dst_ip,

        "source_port":
            src_port,

        "destination_port":
            dst_port,

        "protocol":
            protocol,

        "direction":
            direction,

        "state":
            state,

        "threat":
            threat,

        "severity":
            severity,

        "confidence":
            round(
                float(hybrid_score),
                6
            ),

        "xgboost_probability":
            round(
                float(xgb_probability),
                6
            ),

        "autoencoder_score":
            round(
                float(ae_score),
                6
            ),

        "reconstruction_error":
            round(
                float(ae_error),
                8
            ),

        "reasons":
            reasons,
    }

    return alert


# ============================================================
# PROCESS ONE CHUNK
# ============================================================

def process_chunk(
    df,
    scenario,
    xgb_model,
    xgb_features,
    ae_model,
    scaler,
    schema
):

    # --------------------------------------------------------
    # XGBoost
    # --------------------------------------------------------

    X_xgb = prepare_xgb_features(
        df,
        xgb_features
    )

    xgb_probability = (
        xgb_model
        .predict_proba(
            X_xgb
        )[:, 1]
    )

    # --------------------------------------------------------
    # Autoencoder
    # --------------------------------------------------------

    X_ae = prepare_autoencoder_features(
        df,
        schema,
        scaler
    )

    reconstruction_errors = (
        calculate_reconstruction_error(
            ae_model,
            X_ae
        )
    )

    ae_scores = autoencoder_score(
        reconstruction_errors
    )

    # --------------------------------------------------------
    # Hybrid
    # --------------------------------------------------------

    hybrid_scores = calculate_hybrid_scores(
        xgb_probability,
        ae_scores
    )

    threats = classify_threat(
        xgb_probability,
        ae_scores,
        hybrid_scores
    )

    return (
        xgb_probability,
        ae_scores,
        reconstruction_errors,
        hybrid_scores,
        threats
    )


# ============================================================
# PROCESS SCENARIO
# ============================================================

def process_scenario(
    scenario,
    xgb_model,
    xgb_features,
    ae_model,
    scaler,
    schema,
    jsonl_file,
    csv_file
):

    section(
        f"STREAMING SCENARIO {scenario}"
    )

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
        f"Path: {path}"
    )

    print(
        f"Chunk size: {CHUNK_SIZE:,}"
    )

    total_rows = 0

    benign_count = 0
    suspicious_count = 0
    malicious_count = 0

    total_xgb_alerts = 0
    total_ae_alerts = 0

    csv_header_written = False

    for chunk_number, df in enumerate(
        pd.read_csv(
            path,
            chunksize=CHUNK_SIZE
        ),
        start=1
    ):

        print(
            f"\nChunk {chunk_number}"
        )

        print(
            f"Rows: {len(df):,}"
        )

        (
            xgb_probability,
            ae_scores,
            reconstruction_errors,
            hybrid_scores,
            threats
        ) = process_chunk(
            df,
            scenario,
            xgb_model,
            xgb_features,
            ae_model,
            scaler,
            schema
        )

        # ----------------------------------------------------
        # Counts
        # ----------------------------------------------------

        benign = int(
            np.sum(
                threats == "BENIGN"
            )
        )

        suspicious = int(
            np.sum(
                threats == "SUSPICIOUS"
            )
        )

        malicious = int(
            np.sum(
                threats == "MALICIOUS"
            )
        )

        benign_count += benign
        suspicious_count += suspicious
        malicious_count += malicious

        xgb_alerts = int(
            np.sum(
                xgb_probability
                >= XGB_THRESHOLD
            )
        )

        ae_alerts = int(
            np.sum(
                reconstruction_errors
                >= AE_THRESHOLD
            )
        )

        total_xgb_alerts += xgb_alerts
        total_ae_alerts += ae_alerts

        total_rows += len(df)

        print(
            f"BENIGN     : {benign:,}"
        )

        print(
            f"SUSPICIOUS : {suspicious:,}"
        )

        print(
            f"MALICIOUS  : {malicious:,}"
        )

        print(
            f"XGBoost alerts: {xgb_alerts:,}"
        )

        print(
            f"AE alerts     : {ae_alerts:,}"
        )

        # ----------------------------------------------------
        # Create output dataframe
        # ----------------------------------------------------

        result = pd.DataFrame({

            "scenario":
                scenario,

            "source_ip":
                df["SrcAddr"].values
                if "SrcAddr" in df.columns
                else "",

            "destination_ip":
                df["DstAddr"].values
                if "DstAddr" in df.columns
                else "",

            "protocol":
                df["Proto"].values
                if "Proto" in df.columns
                else "",

            "source_port":
                df["Sport"].values
                if "Sport" in df.columns
                else np.nan,

            "destination_port":
                df["Dport"].values
                if "Dport" in df.columns
                else np.nan,

            "xgboost_probability":
                xgb_probability,

            "autoencoder_score":
                ae_scores,

            "reconstruction_error":
                reconstruction_errors,

            "hybrid_score":
                hybrid_scores,

            "threat":
                threats,

            "target":
                df["Target"].values
                if "Target" in df.columns
                else np.nan,

        })

        # ----------------------------------------------------
        # Save CSV incrementally
        # ----------------------------------------------------

        result.to_csv(
            csv_file,
            mode="a",
            header=not csv_header_written,
            index=False
        )

        csv_header_written = True

        # ----------------------------------------------------
        # Save only suspicious/malicious alerts
        # ----------------------------------------------------

        for index in range(
            len(df)
        ):

            threat = threats[index]

            if threat == "BENIGN":

                continue

            alert = create_alert(
                row=df.iloc[index],
                scenario=scenario,
                xgb_probability=
                    xgb_probability[index],
                ae_score=
                    ae_scores[index],
                ae_error=
                    reconstruction_errors[index],
                hybrid_score=
                    hybrid_scores[index],
                threat=threat
            )

            jsonl_file.write(
                json.dumps(
                    alert,
                    allow_nan=False
                )
                + "\n"
            )

        jsonl_file.flush()
        csv_file.flush()

    print(
        f"\nScenario {scenario} complete."
    )

    print(
        f"Rows processed: {total_rows:,}"
    )

    return {
        "scenario":
            scenario,

        "rows":
            total_rows,

        "benign":
            benign_count,

        "suspicious":
            suspicious_count,

        "malicious":
            malicious_count,

        "xgboost_alerts":
            total_xgb_alerts,

        "autoencoder_alerts":
            total_ae_alerts,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    banner(
        "CTU-13 STREAMING HYBRID "
        "INTRUSION DETECTION"
    )

    print(
        "Configuration:"
    )

    print(
        f"Testing scenarios       : "
        f"{TEST_SCENARIOS}"
    )

    print(
        f"Chunk size              : "
        f"{CHUNK_SIZE:,}"
    )

    print(
        f"XGBoost threshold       : "
        f"{XGB_THRESHOLD}"
    )

    print(
        f"Autoencoder threshold   : "
        f"{AE_THRESHOLD}"
    )

    print(
        f"XGBoost weight          : "
        f"{XGB_WEIGHT}"
    )

    print(
        f"Autoencoder weight      : "
        f"{AE_WEIGHT}"
    )

    print(
        f"Uncertainty range       : "
        f"{UNCERTAINTY_LOW} - "
        f"{UNCERTAINTY_HIGH}"
    )

    print(
        f"Random seed             : "
        f"{RANDOM_SEED}"
    )

    # --------------------------------------------------------
    # Files
    # --------------------------------------------------------

    check_files()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------

    xgb_model, xgb_features = (
        load_xgboost()
    )

    ae_model = load_autoencoder()

    scaler = load_scaler()

    schema = load_schema()

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_models(
        xgb_model,
        xgb_features,
        ae_model,
        scaler,
        schema
    )

    # --------------------------------------------------------
    # Prepare output files
    # --------------------------------------------------------

    if OUTPUT_JSONL.exists():

        OUTPUT_JSONL.unlink()

    if OUTPUT_CSV.exists():

        OUTPUT_CSV.unlink()

    scenario_results = []

    # --------------------------------------------------------
    # Streaming execution
    # --------------------------------------------------------

    banner(
        "STARTING STREAMING DETECTION"
    )

    with open(
        OUTPUT_JSONL,
        "a",
        encoding="utf-8"
    ) as jsonl_file, open(
        OUTPUT_CSV,
        "a",
        encoding="utf-8"
    ) as csv_file:

        for scenario in TEST_SCENARIOS:

            result = process_scenario(
                scenario=scenario,
                xgb_model=xgb_model,
                xgb_features=xgb_features,
                ae_model=ae_model,
                scaler=scaler,
                schema=schema,
                jsonl_file=jsonl_file,
                csv_file=csv_file
            )

            scenario_results.append(
                result
            )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    total_rows = sum(
        x["rows"]
        for x in scenario_results
    )

    total_benign = sum(
        x["benign"]
        for x in scenario_results
    )

    total_suspicious = sum(
        x["suspicious"]
        for x in scenario_results
    )

    total_malicious = sum(
        x["malicious"]
        for x in scenario_results
    )

    total_xgb_alerts = sum(
        x["xgboost_alerts"]
        for x in scenario_results
    )

    total_ae_alerts = sum(
        x["autoencoder_alerts"]
        for x in scenario_results
    )

    summary = {

        "project":
            "CTU-13 Hybrid IDS",

        "mode":
            "Streaming",

        "testing_scenarios":
            TEST_SCENARIOS,

        "chunk_size":
            CHUNK_SIZE,

        "total_rows":
            total_rows,

        "benign":
            total_benign,

        "suspicious":
            total_suspicious,

        "malicious":
            total_malicious,

        "xgboost_alerts":
            total_xgb_alerts,

        "autoencoder_alerts":
            total_ae_alerts,

        "models": {

            "xgboost":
                str(
                    XGB_MODEL_PATH
                ),

            "autoencoder":
                str(
                    AE_MODEL_PATH
                ),

            "scaler":
                str(
                    AE_SCALER_PATH
                ),

            "schema":
                str(
                    AE_SCHEMA_PATH
                ),
        },

        "configuration": {

            "xgboost_threshold":
                XGB_THRESHOLD,

            "autoencoder_threshold":
                AE_THRESHOLD,

            "xgboost_weight":
                XGB_WEIGHT,

            "autoencoder_weight":
                AE_WEIGHT,

            "uncertainty_low":
                UNCERTAINTY_LOW,

            "uncertainty_high":
                UNCERTAINTY_HIGH,
        },

        "scenario_results":
            scenario_results,
    }

    with open(
        OUTPUT_SUMMARY,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=2
        )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    banner(
        "STREAMING DETECTION COMPLETE"
    )

    print(
        f"Total rows processed : "
        f"{total_rows:,}"
    )

    print(
        f"Benign               : "
        f"{total_benign:,}"
    )

    print(
        f"Suspicious           : "
        f"{total_suspicious:,}"
    )

    print(
        f"Malicious            : "
        f"{total_malicious:,}"
    )

    print(
        f"XGBoost alerts       : "
        f"{total_xgb_alerts:,}"
    )

    print(
        f"Autoencoder alerts   : "
        f"{total_ae_alerts:,}"
    )

    print()
    print(
        "Output files:"
    )

    print(
        f"  Alerts : {OUTPUT_JSONL}"
    )

    print(
        f"  Results: {OUTPUT_CSV}"
    )

    print(
        f"  Summary: {OUTPUT_SUMMARY}"
    )

    print()
    print(
        "The detector processed the test data "
        "incrementally using chunks."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()