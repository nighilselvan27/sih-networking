from datetime import datetime, timezone
from pathlib import Path
import json
import os

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import xgboost as xgb


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

XGB_MODEL_PATH = MODELS_DIR / "ctu13_multiscenario_xgboost.json"
AE_MODEL_PATH = MODELS_DIR / "ctu13_autoencoder.keras"
AE_SCALER_PATH = MODELS_DIR / "autoencoder_scaler.joblib"
AE_SCHEMA_PATH = MODELS_DIR / "autoencoder_features.json"


# ============================================================
# CONFIGURATION
# ============================================================

XGB_THRESHOLD = 0.20
AE_THRESHOLD = 0.25541964

XGB_WEIGHT = 0.90
AE_WEIGHT = 0.10

# ------------------------------------------------------------
# Minimum training frequency for a one-hot category to be usable.
#
# The autoencoder was trained with pandas.get_dummies, which has no
# concept of "unknown category": a category the model never saw simply
# has no column, and reindexing produces an all-zero block.
#
# A category that IS in the schema but occurred in a negligible number
# of training rows is effectively the same situation. Its StandardScaler
# scale_ is tiny, so a live 1.0 explodes into hundreds of sigma - an
# input the autoencoder has no learned representation for.
#
# Example: State_UNKNOWN has mean_ = 1.01e-05 (about 3 of ~300k rows),
# scale_ = 0.00317976, so a live 1.0 becomes +314.49 sigma and alone
# accounts for ~86% of the reconstruction error.
#
# Categories below this frequency are resolved to an all-zero block,
# which is exactly what training-time encoding did for unseen values.
# ------------------------------------------------------------

MIN_CATEGORY_FREQUENCY = 1e-4

# Set AE_DEBUG=0 in the environment to silence the per-flow debug block.
AE_DEBUG = os.getenv("AE_DEBUG", "1") == "1"

# Number of rows in the TOP SCALED FEATURES table.
AE_DEBUG_TOP_N = 15


# ============================================================
# GLOBAL MODELS
# ============================================================

xgb_model = None
ae_model = None
ae_scaler = None
ae_schema = None

xgb_feature_names = []
ae_feature_names = []

# {feature: {category: column_index}}
ae_category_columns = {}

# {feature: {category: training_frequency}}
ae_category_frequency = {}

# Highest-gain XGBoost features, derived from the trained booster at load
# time and used as the "supporting evidence" set on each alert.
xgb_top_features = []

models_loaded = False


# ============================================================
# LOAD MODELS
# ============================================================

def load_models():

    global xgb_model
    global ae_model
    global ae_scaler
    global ae_schema
    global xgb_feature_names
    global xgb_top_features
    global ae_feature_names
    global ae_category_columns
    global ae_category_frequency
    global models_loaded

    print("=" * 70)
    print("LOADING IDS MODELS")
    print("=" * 70)

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    required_files = [
        XGB_MODEL_PATH,
        AE_MODEL_PATH,
        AE_SCALER_PATH,
        AE_SCHEMA_PATH,
    ]

    for path in required_files:

        if not path.exists():
            raise FileNotFoundError(
                f"Required model file not found:\n{path}"
            )

    # --------------------------------------------------------
    # XGBoost
    # --------------------------------------------------------

    print("\nLoading XGBoost...")
    print(f"Path: {XGB_MODEL_PATH}")

    xgb_model = xgb.XGBClassifier()

    xgb_model.load_model(str(XGB_MODEL_PATH))

    print("XGBoost loaded successfully.")

    # Get feature names
    try:
        xgb_feature_names = list(
            xgb_model.get_booster().feature_names
        )
    except Exception:
        xgb_feature_names = []

    if not xgb_feature_names:

        xgb_feature_names = [
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

    print(f"XGBoost features: {len(xgb_feature_names)}")

    # --------------------------------------------------------
    # Supporting-evidence feature set.
    #
    # Taken from the trained booster's own gain ranking, so the
    # alert reports the features THIS model actually weights most.
    # Read-only: nothing is computed, fitted or invented here.
    # --------------------------------------------------------

    try:

        gain = xgb_model.get_booster().get_score(
            importance_type="gain"
        )

        xgb_top_features = [
            name
            for name, _ in sorted(
                gain.items(),
                key=lambda item: -item[1],
            )[:8]
        ]

    except Exception:

        xgb_top_features = []

    if not xgb_top_features:

        xgb_top_features = xgb_feature_names[:8]

    print(
        f"Evidence features (top gain): "
        f"{', '.join(xgb_top_features)}"
    )

    # --------------------------------------------------------
    # Autoencoder
    # --------------------------------------------------------

    print("\nLoading Autoencoder...")
    print(f"Path: {AE_MODEL_PATH}")

    ae_model = tf.keras.models.load_model(
        AE_MODEL_PATH,
        compile=False
    )

    print("Autoencoder loaded successfully.")
    print(f"Input shape: {ae_model.input_shape}")

    # --------------------------------------------------------
    # Scaler
    # --------------------------------------------------------

    print("\nLoading Autoencoder scaler...")
    print(f"Path: {AE_SCALER_PATH}")

    ae_scaler = joblib.load(AE_SCALER_PATH)

    print("Autoencoder scaler loaded successfully.")
    print(f"Scaler features: {ae_scaler.n_features_in_}")

    # --------------------------------------------------------
    # Schema
    # --------------------------------------------------------

    print("\nLoading Autoencoder schema...")
    print(f"Path: {AE_SCHEMA_PATH}")

    with open(
        AE_SCHEMA_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        ae_schema = json.load(f)

    print("Autoencoder schema loaded successfully.")

    ae_feature_names = ae_schema["encoded_feature_names"]

    # --------------------------------------------------------
    # Categorical vocabulary
    #
    # The 257 encoded columns are: 30 numeric, then one-hot blocks
    # named "<feature>_<category>". We rebuild the exact vocabulary
    # from the schema, and read each category's training frequency
    # straight off the fitted scaler (for a 0/1 one-hot column the
    # StandardScaler mean_ IS the training frequency).
    #
    # Nothing is re-fitted and no new artifact is produced.
    # --------------------------------------------------------

    ae_category_columns = {}
    ae_category_frequency = {}

    for feature in ae_schema.get("categorical_features", []):

        prefix = f"{feature}_"

        columns = {}
        frequency = {}

        for index, name in enumerate(ae_feature_names):

            if not name.startswith(prefix):
                continue

            category = name[len(prefix):]

            columns[category] = index
            frequency[category] = float(ae_scaler.mean_[index])

        ae_category_columns[feature] = columns
        ae_category_frequency[feature] = frequency

        usable = sum(
            1
            for value in frequency.values()
            if value >= MIN_CATEGORY_FREQUENCY
        )

        print(
            f"  {feature:<6}: {len(columns)} categories "
            f"({usable} above the {MIN_CATEGORY_FREQUENCY:g} "
            f"frequency floor)"
        )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    ae_input_dimension = ae_model.input_shape[-1]
    scaler_dimension = ae_scaler.n_features_in_
    schema_dimension = ae_schema["input_dimension"]
    encoded_dimension = len(ae_feature_names)

    print("\nModel validation:")

    print(
        f"  XGBoost features : "
        f"{len(xgb_feature_names)}"
    )

    print(
        f"  Autoencoder      : "
        f"{ae_input_dimension}"
    )

    print(
        f"  Scaler           : "
        f"{scaler_dimension}"
    )

    print(
        f"  Schema           : "
        f"{schema_dimension}"
    )

    print(
        f"  Encoded features : "
        f"{encoded_dimension}"
    )

    if ae_input_dimension != scaler_dimension:
        raise ValueError(
            "Autoencoder and scaler dimensions do not match."
        )

    if ae_input_dimension != schema_dimension:
        raise ValueError(
            "Autoencoder and schema dimensions do not match."
        )

    if ae_input_dimension != encoded_dimension:
        raise ValueError(
            "Autoencoder and encoded feature dimensions do not match."
        )

    print("\n" + "=" * 70)
    print("ALL MODELS READY")
    print("=" * 70)


    models_loaded = True


# ============================================================
# PREPARE XGBOOST FEATURES
# ============================================================

def prepare_xgb_features(data):

    if isinstance(data, dict):

        df = pd.DataFrame([data])

    elif isinstance(data, pd.DataFrame):

        df = data.copy()

    else:

        df = pd.DataFrame(data)

    result = pd.DataFrame(index=df.index)

    for feature in xgb_feature_names:

        if feature in df.columns:

            result[feature] = pd.to_numeric(
                df[feature],
                errors="coerce"
            ).fillna(0)

        else:

            result[feature] = 0.0

    result = result.replace(
        [np.inf, -np.inf],
        0
    )

    result = result.fillna(0)

    return result


# ============================================================
# CATEGORICAL ENCODING FOR AUTOENCODER
# ============================================================


def resolve_category(feature, raw_value):
    """
    Map a raw categorical value onto its one-hot column index.

    Returns (column_index, note). column_index is None when the value
    has no usable representation, in which case the whole one-hot block
    stays zero - which is exactly what pandas.get_dummies + reindex
    produced at training time for a category it had never seen.
    """

    columns = ae_category_columns.get(feature, {})
    frequency = ae_category_frequency.get(feature, {})

    if raw_value is None:
        value = ""
    else:
        value = str(raw_value).strip()

    # Training categories for Proto are lowercase (tcp, udp, icmp, ...).
    if feature == "Proto":
        value = value.lower()

    if value not in columns:

        return None, (
            f"{feature}={raw_value!r} -> all-zero block "
            f"[not-in-vocabulary]"
        )

    observed = frequency.get(value, 0.0)

    if observed < MIN_CATEGORY_FREQUENCY:

        return None, (
            f"{feature}={raw_value!r} -> all-zero block "
            f"[degenerate, train freq={observed:.3e} "
            f"< {MIN_CATEGORY_FREQUENCY:g}]"
        )

    return columns[value], (
        f"{feature}={raw_value!r} -> {feature}_{value} "
        f"[train freq={observed:.4f}]"
    )


def prepare_autoencoder_features(data):
    """
    Build the exact 257-column encoded matrix the autoencoder expects.

    Column order is taken verbatim from the schema, so it matches the
    matrix the StandardScaler was fitted on.

    Returns (DataFrame in schema order, list of resolution notes).
    """

    if isinstance(data, dict):

        df = pd.DataFrame([data])

    elif isinstance(data, pd.DataFrame):

        df = data.copy()

    else:

        df = pd.DataFrame(data)

    numeric_features = ae_schema.get(
        "numeric_features",
        []
    )

    categorical_features = ae_schema.get(
        "categorical_features",
        []
    )

    encoded_feature_names = ae_schema[
        "encoded_feature_names"
    ]

    rows = len(df)

    # --------------------------------------------------------
    # Allocate the full matrix once.
    #
    # Every unused one-hot column is legitimately zero, so starting
    # from zeros and filling by index is both correct and much
    # cheaper than concatenating ~227 single-column frames.
    # --------------------------------------------------------

    matrix = np.zeros(
        (rows, len(encoded_feature_names)),
        dtype=np.float32,
    )

    column_index = {
        name: index
        for index, name in enumerate(encoded_feature_names)
    }

    # --------------------------------------------------------
    # Numeric features
    # --------------------------------------------------------

    for feature in numeric_features:

        index = column_index.get(feature)

        if index is None:
            continue

        if feature in df.columns:

            values = pd.to_numeric(
                df[feature],
                errors="coerce",
            )

            values = values.replace(
                [np.inf, -np.inf],
                np.nan,
            )

            values = values.fillna(0.0)

            matrix[:, index] = values.to_numpy(
                dtype=np.float32
            )

    # --------------------------------------------------------
    # Categorical features
    # --------------------------------------------------------

    notes = []

    for feature in categorical_features:

        if feature in df.columns:

            raw_values = df[feature].tolist()

        else:

            raw_values = [None] * rows

        for row, raw_value in enumerate(raw_values):

            index, note = resolve_category(
                feature,
                raw_value,
            )

            if index is not None:

                matrix[row, index] = 1.0

            if row == 0:

                notes.append(note)

    encoded = pd.DataFrame(
        matrix,
        index=df.index,
        columns=encoded_feature_names,
    )

    return encoded, notes


# ============================================================
# AUTOENCODER SCORE
# ============================================================

def print_ae_debug(X, scaled, error, notes):
    """
    Per-flow autoencoder diagnostics.

    Shows how many encoded features are non-zero, how each categorical
    value was resolved, the reconstruction error against the threshold,
    and the features that dominate the scaled input vector.
    """

    print()
    print("=" * 74)
    print("[AE DEBUG]")
    print("=" * 74)

    print(
        f"Non-zero encoded features : "
        f"{int(np.count_nonzero(X[0]))} / {X.shape[1]}"
    )

    for note in notes:
        print(f"Categorical resolution    : {note}")

    print(
        f"Raw min / max             : "
        f"{float(np.min(X)):.4f} / {float(np.max(X)):.4f}"
    )

    print(
        f"Scaled min / max          : "
        f"{float(np.min(scaled)):.4f} / {float(np.max(scaled)):.4f}"
    )

    print(
        f"Scaled mean / std         : "
        f"{float(np.mean(scaled)):.4f} / {float(np.std(scaled)):.4f}"
    )

    print(
        f"Reconstruction error      : {error:.8f}"
    )

    print(
        f"Threshold                 : {AE_THRESHOLD}"
    )

    print(
        f"Anomalous                 : {error >= AE_THRESHOLD}"
    )

    # --------------------------------------------------------
    # TOP SCALED FEATURES
    # --------------------------------------------------------

    print()
    print("TOP SCALED FEATURES")

    print(
        f"{'rank':>4}  "
        f"{'feature_name':<30}"
        f"{'raw value':>18}"
        f"{'scaled value':>16}"
    )

    print("-" * 74)

    order = np.argsort(
        -np.abs(scaled[0])
    )[:AE_DEBUG_TOP_N]

    for rank, index in enumerate(order, start=1):

        print(
            f"{rank:>4}  "
            f"{ae_feature_names[index]:<30}"
            f"{float(X[0][index]):>18.4f}"
            f"{float(scaled[0][index]):>16.4f}"
        )

    print("=" * 74)
    print()


def calculate_autoencoder_score(X, notes=None):

    X = np.asarray(
        X,
        dtype=np.float32
    )

    if X.ndim == 1:

        X = X.reshape(1, -1)

    # --------------------------------------------------------
    # Validate dimension
    # --------------------------------------------------------

    expected = ae_model.input_shape[-1]

    if X.shape[1] != expected:

        raise ValueError(
            f"Autoencoder received {X.shape[1]} "
            f"features but expects {expected}."
        )

    # --------------------------------------------------------
    # Scaling
    # --------------------------------------------------------

    scaled = ae_scaler.transform(X)

    scaled = np.nan_to_num(
        scaled,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)

    # --------------------------------------------------------
    # Reconstruction
    # --------------------------------------------------------

    reconstruction = ae_model.predict(
        scaled,
        verbose=0
    )

    reconstruction = np.asarray(
        reconstruction,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Reconstruction error
    # --------------------------------------------------------

    error = np.mean(
        np.square(
            scaled - reconstruction
        ),
        axis=1
    )

    error_value = float(error[0])

    if AE_DEBUG:

        print_ae_debug(
            X,
            scaled,
            error_value,
            notes or [],
        )

    return error_value


# ============================================================
# XGBOOST SCORE
# ============================================================

def calculate_xgb_score(data):

    X = prepare_xgb_features(data)

    probability = xgb_model.predict_proba(
        X
    )

    # Positive class
    if probability.shape[1] >= 2:

        score = float(
            probability[0][1]
        )

    else:

        score = float(
            probability[0][0]
        )

    return score


# ============================================================
# HYBRID PREDICTION
# ============================================================

def predict(data):

    if not models_loaded:

        load_models()

    # --------------------------------------------------------
    # XGBoost
    # --------------------------------------------------------

    xgb_score = calculate_xgb_score(
        data
    )

    # --------------------------------------------------------
    # Autoencoder
    # --------------------------------------------------------

    ae_input, ae_notes = prepare_autoencoder_features(
        data
    )

    ae_array = ae_input.to_numpy(
        dtype=np.float32
    )

    ae_error = calculate_autoencoder_score(
        ae_array,
        ae_notes,
    )

    ae_non_zero = int(
        np.count_nonzero(ae_array[0])
    )

    # --------------------------------------------------------
    # Individual decisions
    # --------------------------------------------------------

    xgb_malicious = (
        xgb_score >= XGB_THRESHOLD
    )

    ae_anomalous = (
        ae_error >= AE_THRESHOLD
    )

    # --------------------------------------------------------
    # Normalize AE score for hybrid confidence
    #
    # This prevents reconstruction error from dominating
    # because its scale is different from XGBoost probability.
    # --------------------------------------------------------

    if AE_THRESHOLD > 0:

        ae_score_normalized = min(
            ae_error / AE_THRESHOLD,
            1.0
        )

    else:

        ae_score_normalized = 0.0

    # --------------------------------------------------------
    # GATED HYBRID
    #
    # XGBoost is primary.
    # Autoencoder acts as anomaly support.
    # --------------------------------------------------------

    if xgb_malicious:

        prediction = 1

        gated = True

    elif ae_anomalous and xgb_score >= 0.05:

        prediction = 1

        gated = True

    else:

        prediction = 0

        gated = False

    # --------------------------------------------------------
    # Hybrid score
    # --------------------------------------------------------

    hybrid_score = (
        XGB_WEIGHT * xgb_score
        +
        AE_WEIGHT * ae_score_normalized
    )

    hybrid_score = float(
        np.clip(
            hybrid_score,
            0.0,
            1.0
        )
    )

    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    if prediction == 1:

        confidence = hybrid_score

    else:

        confidence = 1.0 - hybrid_score

    confidence = float(
        np.clip(
            confidence,
            0.0,
            1.0
        )
    )

    # --------------------------------------------------------
    # Label
    # --------------------------------------------------------

    label = (
        "MALICIOUS"
        if prediction == 1
        else "BENIGN"
    )

    # --------------------------------------------------------
    # Risk
    # --------------------------------------------------------

    if prediction == 1:

        if confidence >= 0.85:

            risk = "CRITICAL"

        elif confidence >= 0.65:

            risk = "HIGH"

        else:

            risk = "MEDIUM"

    else:

        if hybrid_score >= 0.10:

            risk = "LOW"

        else:

            risk = "SAFE"

    # --------------------------------------------------------
    # Explanation
    # --------------------------------------------------------

    if prediction == 1:

        if xgb_malicious and ae_anomalous:

            explanation = (
                "Malicious traffic detected. "
                "The supervised XGBoost model identified "
                "malicious characteristics and the "
                "autoencoder also detected anomalous "
                "traffic behaviour."
            )

        elif xgb_malicious:

            explanation = (
                "Malicious traffic detected by XGBoost. "
                "The observed flow characteristics are "
                "consistent with malicious traffic."
            )

        else:

            explanation = (
                "Anomalous traffic detected by the hybrid "
                "detector. The traffic differs from the "
                "learned normal behaviour."
            )

    else:

        explanation = (
            "Benign traffic. Traffic characteristics "
            "appear consistent with benign behaviour."
        )

    # --------------------------------------------------------
    # Standardized alert fields
    #
    # These surface information the pipeline ALREADY carries.
    # live_capture.py's build_payload() sends "timestamp" and
    # "flow_id", and FlowInput allows extra keys, so both arrive
    # here untouched. Nothing about capture, flow keying, feature
    # extraction or the prediction itself is changed.
    # --------------------------------------------------------

    source = data if isinstance(data, dict) else {}

    # Timestamp: prefer the capture-side value (when the flow was
    # observed); fall back to a real server-side time otherwise.

    capture_timestamp = source.get("timestamp")

    if capture_timestamp:

        timestamp = str(capture_timestamp)
        timestamp_source = "capture"

    else:

        timestamp = datetime.now(timezone.utc).isoformat()
        timestamp_source = "server"

    # Flow id: reuse live_capture.py's identifier when supplied,
    # otherwise rebuild it in exactly the same format.

    flow_id = source.get("flow_id")

    if not flow_id:

        metadata = source.get("metadata") or {}

        src_ip = (
            source.get("SrcAddr")
            or metadata.get("src_ip")
            or "unknown"
        )

        dst_ip = (
            source.get("DstAddr")
            or metadata.get("dst_ip")
            or "unknown"
        )

        def _port(value):
            try:
                return int(float(value))
            except (TypeError, ValueError):
                return 0

        flow_id = (
            f"{src_ip}:{_port(source.get('Sport'))}-"
            f"{dst_ip}:{_port(source.get('Dport'))}-"
            f"{str(source.get('Proto', '')).upper()}"
        )

    flow_id = str(flow_id)

    # Threat class: the CTU-13 ThreatClass vocabulary is exactly
    # {BENIGN, BOTNET}. This is a relabelling of the existing binary
    # prediction - the decision itself is unchanged.

    threat_class = "BOTNET" if prediction == 1 else "BENIGN"

    # Supporting evidence: this flow's ACTUAL values for the features
    # the trained model weights most, plus the three categoricals the
    # autoencoder consumed. Values are read from the request; no new
    # metric is derived.

    supporting_features = {}

    for feature in xgb_top_features:

        if feature in source:

            supporting_features[feature] = source[feature]

    for feature in ("Proto", "Dir", "State"):

        if feature in source:

            supporting_features[feature] = source[feature]

    # --------------------------------------------------------
    # Standardized response
    # --------------------------------------------------------

    result = {

        # ----------------------------------------------------
        # Standardized alert schema
        # ----------------------------------------------------

        "timestamp": timestamp,

        "flow_id": flow_id,

        "threat_class": threat_class,

        "supporting_features": supporting_features,


        "prediction": int(prediction),

        "label": label,

        "confidence": round(
            confidence,
            6
        ),

        "xgboost_score": round(
            xgb_score,
            6
        ),

        "autoencoder_score": round(
            ae_error,
            8
        ),

        "autoencoder_normalized": round(
            ae_score_normalized,
            6
        ),

        "hybrid_score": round(
            hybrid_score,
            6
        ),

        "risk": risk,

        "gated": bool(gated),

        "xgboost_malicious": bool(
            xgb_malicious
        ),

        "autoencoder_anomalous": bool(
            ae_anomalous
        ),

        "explanation": explanation,

        "evidence": {

            "xgboost_threshold":
                XGB_THRESHOLD,

            "autoencoder_threshold":
                AE_THRESHOLD,

            "xgboost_weight":
                XGB_WEIGHT,

            "autoencoder_weight":
                AE_WEIGHT,
        },

        # ----------------------------------------------------
        # Aliases required by schemas.PredictionResponse.
        #
        # These carry the same values under the names the API
        # contract (and live_capture.py) already expect. Every
        # original key above is preserved unchanged.
        # ----------------------------------------------------

        "xgboost_probability": round(
            xgb_score,
            6
        ),

        "xgboost_anomaly": bool(
            xgb_malicious
        ),

        "autoencoder_anomaly": bool(
            ae_anomalous
        ),

        "risk_level": risk,

        "details": {

            "xgboost_threshold":
                XGB_THRESHOLD,

            "autoencoder_threshold":
                AE_THRESHOLD,

            "xgboost_weight":
                XGB_WEIGHT,

            "autoencoder_weight":
                AE_WEIGHT,

            "autoencoder_non_zero_features":
                ae_non_zero,

            "autoencoder_encoded_features":
                len(ae_feature_names),

            "categorical_resolution":
                ae_notes,

            "timestamp_source":
                timestamp_source,

            "evidence_features":
                list(xgb_top_features),
        },
    }

    return result


# ============================================================
# MODEL INFORMATION
# ============================================================

def model_info():

    if not models_loaded:

        load_models()

    return {

        "status": "ready",

        "xgboost": {

            "features":
                len(xgb_feature_names),

            "threshold":
                XGB_THRESHOLD,

            "weight":
                XGB_WEIGHT,
        },

        "autoencoder": {

            "features":
                len(ae_feature_names),

            "threshold":
                AE_THRESHOLD,

            "weight":
                AE_WEIGHT,
        },

        "hybrid": {

            "mode":
                "GATED",

            "xgboost_weight":
                XGB_WEIGHT,

            "autoencoder_weight":
                AE_WEIGHT,
        }
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    load_models()

    print(
        json.dumps(
            model_info(),
            indent=2
        )
    )