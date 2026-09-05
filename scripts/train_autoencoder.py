"""
CTU-13 Autoencoder Training
===========================

Trains an unsupervised Autoencoder using BENIGN traffic from
CTU-13 scenarios 1-10.

The script:

1. Loads CTU-13 feature CSVs
2. Selects benign traffic only
3. Uses the ACTUAL CTU-13 feature schema
4. Encodes categorical network features
5. Handles NaN / Inf values
6. Standardizes features
7. Trains a dense Autoencoder
8. Calculates reconstruction errors
9. Automatically selects an anomaly threshold
10. Saves:
      models/ctu13_autoencoder.keras
      models/autoencoder_scaler.joblib
      models/autoencoder_features.json
      models/autoencoder_config.json
      models/autoencoder_training_history.csv
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from tensorflow import keras
from tensorflow.keras import layers


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

AUTOENCODER_PATH = MODEL_DIR / "ctu13_autoencoder.keras"
SCALER_PATH = MODEL_DIR / "autoencoder_scaler.joblib"
FEATURES_PATH = MODEL_DIR / "autoencoder_features.json"
CONFIG_PATH = MODEL_DIR / "autoencoder_config.json"
HISTORY_PATH = MODEL_DIR / "autoencoder_training_history.csv"

TRAIN_SCENARIOS = range(1, 11)

MAX_BENIGN_ROWS_PER_SCENARIO = 30_000

EPOCHS = 50
BATCH_SIZE = 512
VALIDATION_SPLIT = 0.20

THRESHOLD_PERCENTILE = 99.0

RANDOM_SEED = 42


# ============================================================
# ACTUAL CTU-13 SCHEMA
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

TARGET_COLUMN = "Target"


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed() -> None:
    os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)


# ============================================================
# PRINTING
# ============================================================

def banner(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# LOAD DATA
# ============================================================

def load_benign_training_data() -> pd.DataFrame:

    banner("LOADING BENIGN TRAINING DATA")

    frames = []

    for scenario in TRAIN_SCENARIOS:

        path = DATA_DIR / f"scenario{scenario}" / "ctu13_features.csv"

        print(f"Loading Scenario {scenario}: {path}")

        if not path.exists():
            print("  WARNING: File not found. Skipping.")
            continue

        df = pd.read_csv(path)

        print(f"  Rows    : {len(df):,}")
        print(f"  Columns : {len(df.columns)}")

        if TARGET_COLUMN not in df.columns:
            raise ValueError(
                f"Target column '{TARGET_COLUMN}' missing "
                f"from scenario {scenario}"
            )

        benign = df[df[TARGET_COLUMN] == 0].copy()

        print(f"  Benign rows: {len(benign):,}")

        if len(benign) > MAX_BENIGN_ROWS_PER_SCENARIO:

            benign = benign.sample(
                n=MAX_BENIGN_ROWS_PER_SCENARIO,
                random_state=RANDOM_SEED,
            )

            print(
                f"  Sampled: {len(benign):,}"
            )

        else:
            print(
                f"  Using all: {len(benign):,}"
            )

        frames.append(benign)

        del df

    if not frames:
        raise RuntimeError(
            "No training data was loaded."
        )

    data = pd.concat(
        frames,
        ignore_index=True,
    )

    print()
    print("-" * 70)
    print("BENIGN TRAINING DATA SUMMARY")
    print("-" * 70)

    print(
        f"Total benign rows : {len(data):,}"
    )

    return data


# ============================================================
# FEATURE PREPARATION
# ============================================================

def prepare_features(
    df: pd.DataFrame,
):
    banner("PREPARING FEATURES")

    missing_numeric = [
        col
        for col in NUMERIC_FEATURES
        if col not in df.columns
    ]

    missing_categorical = [
        col
        for col in CATEGORICAL_FEATURES
        if col not in df.columns
    ]

    if missing_numeric or missing_categorical:

        print()
        print("Missing features:")

        for col in missing_numeric:
            print(f"  - {col}")

        for col in missing_categorical:
            print(f"  - {col}")

        raise ValueError(
            "Required CTU-13 features are missing."
        )

    # --------------------------------------------------------
    # Numeric features
    # --------------------------------------------------------

    numeric_df = df[NUMERIC_FEATURES].copy()

    for col in NUMERIC_FEATURES:
        numeric_df[col] = pd.to_numeric(
            numeric_df[col],
            errors="coerce",
        )

    numeric_df = numeric_df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    numeric_nan = int(
        numeric_df.isna().sum().sum()
    )

    print(
        f"Numeric NaN values before cleaning : "
        f"{numeric_nan:,}"
    )

    # Median imputation
    numeric_df = numeric_df.fillna(
        numeric_df.median()
    )

    # --------------------------------------------------------
    # Categorical features
    # --------------------------------------------------------

    categorical_df = df[
        CATEGORICAL_FEATURES
    ].copy()

    for col in CATEGORICAL_FEATURES:

        categorical_df[col] = (
            categorical_df[col]
            .fillna("UNKNOWN")
            .astype(str)
            .str.strip()
        )

    print(
        f"Categorical features              : "
        f"{len(CATEGORICAL_FEATURES)}"
    )

    # --------------------------------------------------------
    # One-hot encoding
    # --------------------------------------------------------

    print("Encoding categorical features...")

    categorical_encoded = pd.get_dummies(
        categorical_df,
        columns=CATEGORICAL_FEATURES,
        dtype=np.float32,
    )

    print(
        f"Categorical encoded columns       : "
        f"{categorical_encoded.shape[1]}"
    )

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    feature_df = pd.concat(
        [
            numeric_df,
            categorical_encoded,
        ],
        axis=1,
    )

    # Ensure everything is numeric
    feature_df = feature_df.apply(
        pd.to_numeric,
        errors="coerce",
    )

    feature_df = feature_df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    final_nan = int(
        feature_df.isna().sum().sum()
    )

    print(
        f"Final NaN values                 : "
        f"{final_nan:,}"
    )

    if final_nan > 0:

        feature_df = feature_df.fillna(0)

    feature_names = feature_df.columns.tolist()

    print()
    print(
        f"Final feature matrix: "
        f"{feature_df.shape[0]:,} x "
        f"{feature_df.shape[1]:,}"
    )

    return (
        feature_df.astype(np.float32),
        feature_names,
    )


# ============================================================
# BUILD AUTOENCODER
# ============================================================

def build_autoencoder(input_dim: int):

    # Keep architecture relatively small because
    # CTU-13 feature dimensionality is modest.

    latent_dim = max(
        8,
        min(32, input_dim // 3),
    )

    hidden_1 = max(
        32,
        min(128, input_dim * 2),
    )

    hidden_2 = max(
        16,
        min(64, input_dim),
    )

    inputs = keras.Input(
        shape=(input_dim,),
        name="network_features",
    )

    # Encoder
    x = layers.Dense(
        hidden_1,
        activation="relu",
        name="encoder_dense_1",
    )(inputs)

    x = layers.BatchNormalization()(x)

    x = layers.Dense(
        hidden_2,
        activation="relu",
        name="encoder_dense_2",
    )(x)

    latent = layers.Dense(
        latent_dim,
        activation="relu",
        name="latent_space",
    )(x)

    # Decoder
    x = layers.Dense(
        hidden_2,
        activation="relu",
        name="decoder_dense_1",
    )(latent)

    x = layers.BatchNormalization()(x)

    x = layers.Dense(
        hidden_1,
        activation="relu",
        name="decoder_dense_2",
    )(x)

    outputs = layers.Dense(
        input_dim,
        activation="linear",
        name="reconstruction",
    )(x)

    model = keras.Model(
        inputs,
        outputs,
        name="CTU13_Autoencoder",
    )

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="mse",
    )

    return model


# ============================================================
# RECONSTRUCTION ERROR
# ============================================================

def reconstruction_errors(
    model,
    X: np.ndarray,
) -> np.ndarray:

    reconstructed = model.predict(
        X,
        batch_size=BATCH_SIZE,
        verbose=0,
    )

    errors = np.mean(
        np.square(
            X - reconstructed
        ),
        axis=1,
    )

    return errors


# ============================================================
# SAVE JSON
# ============================================================

def save_json(path: Path, data) -> None:

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            data,
            f,
            indent=2,
        )


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed()

    banner("CTU-13 AUTOENCODER TRAINING")

    print("Configuration:")
    print(
        f"  Training scenarios       : "
        f"1 - 10"
    )
    print(
        f"  Max benign rows/scenario: "
        f"{MAX_BENIGN_ROWS_PER_SCENARIO:,}"
    )
    print(
        f"  Epochs                   : "
        f"{EPOCHS}"
    )
    print(
        f"  Batch size               : "
        f"{BATCH_SIZE}"
    )
    print(
        f"  Validation split         : "
        f"{VALIDATION_SPLIT}"
    )
    print(
        f"  Threshold percentile     : "
        f"{THRESHOLD_PERCENTILE}"
    )
    print(
        f"  Random seed              : "
        f"{RANDOM_SEED}"
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    benign_data = load_benign_training_data()

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    X_df, feature_names = prepare_features(
        benign_data
    )

    del benign_data

    # --------------------------------------------------------
    # Scale
    # --------------------------------------------------------

    banner("STANDARDIZING FEATURES")

    scaler = StandardScaler()

    X = scaler.fit_transform(
        X_df.values
    ).astype(np.float32)

    print(
        f"Scaled matrix: "
        f"{X.shape[0]:,} x "
        f"{X.shape[1]:,}"
    )

    # --------------------------------------------------------
    # Build
    # --------------------------------------------------------

    banner("BUILDING AUTOENCODER")

    model = build_autoencoder(
        X.shape[1]
    )

    model.summary()

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    banner("TRAINING AUTOENCODER")

    callbacks = [

        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=7,
            restore_best_weights=True,
            verbose=1,
        ),

        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        X,
        X,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        shuffle=True,
        callbacks=callbacks,
        verbose=1,
    )

    # --------------------------------------------------------
    # Reconstruction errors
    # --------------------------------------------------------

    banner("CALCULATING RECONSTRUCTION ERRORS")

    errors = reconstruction_errors(
        model,
        X,
    )

    print()
    print(
        f"Mean reconstruction error   : "
        f"{np.mean(errors):.8f}"
    )

    print(
        f"Median reconstruction error : "
        f"{np.median(errors):.8f}"
    )

    print(
        f"Std reconstruction error    : "
        f"{np.std(errors):.8f}"
    )

    print(
        f"Minimum reconstruction error: "
        f"{np.min(errors):.8f}"
    )

    print(
        f"Maximum reconstruction error: "
        f"{np.max(errors):.8f}"
    )

    # --------------------------------------------------------
    # Threshold
    # --------------------------------------------------------

    banner("CALCULATING ANOMALY THRESHOLD")

    threshold = float(
        np.percentile(
            errors,
            THRESHOLD_PERCENTILE,
        )
    )

    print(
        f"Threshold percentile : "
        f"{THRESHOLD_PERCENTILE}%"
    )

    print(
        f"Anomaly threshold    : "
        f"{threshold:.8f}"
    )

    training_anomalies = (
        errors >= threshold
    )

    print(
        f"Training rows above threshold: "
        f"{training_anomalies.sum():,}"
    )

    print(
        f"Training anomaly rate        : "
        f"{training_anomalies.mean() * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    banner("SAVING AUTOENCODER")

    model.save(
        AUTOENCODER_PATH
    )

    print(
        f"Autoencoder saved to:\n"
        f"  {AUTOENCODER_PATH}"
    )

    # --------------------------------------------------------
    # Save scaler
    # --------------------------------------------------------

    joblib.dump(
        scaler,
        SCALER_PATH,
    )

    print(
        f"Scaler saved to:\n"
        f"  {SCALER_PATH}"
    )

    # --------------------------------------------------------
    # Save feature schema
    # --------------------------------------------------------

    feature_schema = {
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "encoded_feature_names": feature_names,
        "input_dimension": len(feature_names),
    }

    save_json(
        FEATURES_PATH,
        feature_schema,
    )

    print(
        f"Feature schema saved to:\n"
        f"  {FEATURES_PATH}"
    )

    # --------------------------------------------------------
    # Save config
    # --------------------------------------------------------

    config = {

        "model": "CTU13_Autoencoder",

        "training_scenarios": [
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
        ],

        "max_benign_rows_per_scenario":
            MAX_BENIGN_ROWS_PER_SCENARIO,

        "epochs": EPOCHS,

        "batch_size": BATCH_SIZE,

        "validation_split":
            VALIDATION_SPLIT,

        "threshold_percentile":
            THRESHOLD_PERCENTILE,

        "threshold":
            threshold,

        "input_dimension":
            len(feature_names),

        "numeric_features":
            NUMERIC_FEATURES,

        "categorical_features":
            CATEGORICAL_FEATURES,

        "random_seed":
            RANDOM_SEED,
    }

    save_json(
        CONFIG_PATH,
        config,
    )

    print(
        f"Configuration saved to:\n"
        f"  {CONFIG_PATH}"
    )

    # --------------------------------------------------------
    # Save training history
    # --------------------------------------------------------

    history_df = pd.DataFrame(
        history.history
    )

    history_df.insert(
        0,
        "epoch",
        np.arange(
            1,
            len(history_df) + 1,
        ),
    )

    history_df.to_csv(
        HISTORY_PATH,
        index=False,
    )

    print(
        f"Training history saved to:\n"
        f"  {HISTORY_PATH}"
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    banner("AUTOENCODER TRAINING COMPLETE")

    print(
        f"Training rows       : "
        f"{len(X):,}"
    )

    print(
        f"Input features      : "
        f"{len(feature_names):,}"
    )

    print(
        f"Training epochs     : "
        f"{len(history_df):,}"
    )

    print(
        f"Threshold percentile: "
        f"{THRESHOLD_PERCENTILE}%"
    )

    print(
        f"Anomaly threshold   : "
        f"{threshold:.8f}"
    )

    print()
    print("Output files:")

    print(
        f"  {AUTOENCODER_PATH}"
    )

    print(
        f"  {SCALER_PATH}"
    )

    print(
        f"  {FEATURES_PATH}"
    )

    print(
        f"  {CONFIG_PATH}"
    )

    print(
        f"  {HISTORY_PATH}"
    )

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()