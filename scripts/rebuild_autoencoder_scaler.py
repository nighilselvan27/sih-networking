import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

SCHEMA_PATH = MODELS_DIR / "autoencoder_features.json"
OUTPUT_PATH = MODELS_DIR / "autoencoder_scaler.joblib"

TRAIN_SCENARIOS = range(1, 11)

RANDOM_SEED = 42
MAX_BENIGN_ROWS_PER_SCENARIO = 30_000


# ============================================================
# HELPERS
# ============================================================

def print_header(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def load_schema():
    print_header("LOADING AUTOENCODER SCHEMA")

    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(
            f"Schema not found:\n{SCHEMA_PATH}"
        )

    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema = json.load(f)

    numeric_features = schema["numeric_features"]
    categorical_features = schema["categorical_features"]
    encoded_feature_names = schema["encoded_feature_names"]
    input_dimension = schema["input_dimension"]

    print(f"Numeric features    : {len(numeric_features)}")
    print(f"Categorical features: {len(categorical_features)}")
    print(f"Encoded features   : {len(encoded_feature_names)}")
    print(f"Input dimension     : {input_dimension}")

    if len(encoded_feature_names) != input_dimension:
        raise ValueError(
            "Schema inconsistency: encoded feature count does not "
            "match input_dimension."
        )

    return (
        schema,
        numeric_features,
        categorical_features,
        encoded_feature_names,
    )


# ============================================================
# LOAD TRAINING DATA
# ============================================================

def load_training_data(
    numeric_features,
    categorical_features,
):
    print_header("LOADING BENIGN TRAINING DATA")

    rng = np.random.default_rng(RANDOM_SEED)

    frames = []

    required_columns = set(
        numeric_features + categorical_features + ["Target"]
    )

    for scenario in TRAIN_SCENARIOS:

        path = (
            DATA_DIR
            / f"scenario{scenario}"
            / "ctu13_features.csv"
        )

        print(f"\nScenario {scenario}")
        print(f"Path: {path}")

        if not path.exists():
            raise FileNotFoundError(
                f"Training data not found:\n{path}"
            )

        df = pd.read_csv(path)

        print(f"Rows    : {len(df):,}")
        print(f"Columns : {len(df.columns)}")

        missing = [
            col
            for col in required_columns
            if col not in df.columns
        ]

        if missing:
            raise ValueError(
                f"Scenario {scenario} is missing columns:\n"
                + "\n".join(f"  - {x}" for x in missing)
            )

        # ----------------------------------------------------
        # Keep benign traffic only
        # ----------------------------------------------------

        benign = df[df["Target"] == 0].copy()

        print(f"Benign rows: {len(benign):,}")

        if len(benign) > MAX_BENIGN_ROWS_PER_SCENARIO:
            benign = benign.sample(
                n=MAX_BENIGN_ROWS_PER_SCENARIO,
                random_state=RANDOM_SEED,
            )

            print(
                f"Sampled   : "
                f"{len(benign):,}"
            )

        frames.append(
            benign[
                numeric_features + categorical_features
            ].copy()
        )

    result = pd.concat(
        frames,
        ignore_index=True,
    )

    print()
    print("-" * 70)
    print(
        f"TOTAL BENIGN TRAINING ROWS: "
        f"{len(result):,}"
    )
    print("-" * 70)

    return result


# ============================================================
# PREPARE 257 FEATURES
# ============================================================

def prepare_encoded_features(
    df,
    numeric_features,
    categorical_features,
    encoded_feature_names,
):
    print_header("PREPARING 257-DIMENSIONAL FEATURES")

    data = df.copy()

    # --------------------------------------------------------
    # Numeric columns
    # --------------------------------------------------------

    numeric = data[numeric_features].copy()

    for column in numeric_features:
        numeric[column] = pd.to_numeric(
            numeric[column],
            errors="coerce",
        )

    numeric = numeric.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    numeric = numeric.fillna(0.0)

    # --------------------------------------------------------
    # Categorical columns
    #
    # IMPORTANT:
    # get_dummies produces one-hot columns.
    # We then explicitly reindex to the saved schema.
    # This guarantees the exact 257-column order.
    # --------------------------------------------------------

    categorical = data[categorical_features].copy()

    for column in categorical_features:
        categorical[column] = (
            categorical[column]
            .fillna("UNKNOWN")
            .astype(str)
        )

    categorical_encoded = pd.get_dummies(
        categorical,
        columns=categorical_features,
        dtype=float,
    )

    # --------------------------------------------------------
    # Combine numeric + categorical
    # --------------------------------------------------------

    encoded = pd.concat(
        [
            numeric.reset_index(drop=True),
            categorical_encoded.reset_index(drop=True),
        ],
        axis=1,
    )

    # --------------------------------------------------------
    # Force EXACT schema/order
    # --------------------------------------------------------

    encoded = encoded.reindex(
        columns=encoded_feature_names,
        fill_value=0.0,
    )

    encoded = encoded.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    encoded = encoded.fillna(0.0)

    encoded = encoded.astype(np.float32)

    print(f"Generated feature matrix:")
    print(f"Rows    : {len(encoded):,}")
    print(f"Columns : {len(encoded.columns)}")

    if len(encoded.columns) != len(encoded_feature_names):
        raise ValueError(
            "Encoded feature dimension mismatch."
        )

    # Verify exact order
    if encoded.columns.tolist() != encoded_feature_names:
        raise ValueError(
            "Encoded feature order does not match schema."
        )

    print("Feature schema/order verified.")

    return encoded


# ============================================================
# BUILD SCALER
# ============================================================

def build_scaler(encoded):
    print_header("BUILDING STANDARD SCALER")

    scaler = StandardScaler()

    scaler.fit(encoded.values)

    print(
        f"Scaler fitted on "
        f"{encoded.shape[0]:,} rows x "
        f"{encoded.shape[1]} features."
    )

    # --------------------------------------------------------
    # Save feature names manually.
    #
    # This is important because the backend previously
    # encountered a missing feature_names_in_ attribute.
    # --------------------------------------------------------

    scaler.feature_names_in_ = np.asarray(
        encoded.columns.tolist(),
        dtype=object,
    )

    joblib.dump(
        scaler,
        OUTPUT_PATH,
    )

    print()
    print(f"[OK] Scaler saved:")
    print(f"     {OUTPUT_PATH}")

    return scaler


# ============================================================
# VERIFY
# ============================================================

def verify_scaler(
    scaler,
    encoded_feature_names,
):
    print_header("VERIFYING SAVED SCALER")

    loaded = joblib.load(
        OUTPUT_PATH
    )

    print(
        f"Scaler feature count: "
        f"{len(loaded.feature_names_in_)}"
    )

    print(
        f"Schema feature count: "
        f"{len(encoded_feature_names)}"
    )

    if len(loaded.feature_names_in_) != len(
        encoded_feature_names
    ):
        raise ValueError(
            "Saved scaler feature count does not "
            "match schema."
        )

    if loaded.feature_names_in_.tolist() != (
        encoded_feature_names
    ):
        raise ValueError(
            "Saved scaler feature order does not "
            "match schema."
        )

    print()
    print("[OK] Feature count matches.")
    print("[OK] Feature order matches.")
    print("[OK] Scaler successfully verified.")


# ============================================================
# MAIN
# ============================================================

def main():

    print_header(
        "CTU-13 AUTOENCODER SCALER RECONSTRUCTION"
    )

    print("Purpose:")
    print(
        "Reconstruct the missing StandardScaler using "
        "the existing training scenarios and the "
        "existing 257-feature autoencoder schema."
    )

    print()
    print(f"Project directory : {BASE_DIR}")
    print(f"Data directory    : {DATA_DIR}")
    print(f"Models directory  : {MODELS_DIR}")
    print(f"Output scaler     : {OUTPUT_PATH}")

    (
        schema,
        numeric_features,
        categorical_features,
        encoded_feature_names,
    ) = load_schema()

    training_data = load_training_data(
        numeric_features,
        categorical_features,
    )

    encoded = prepare_encoded_features(
        training_data,
        numeric_features,
        categorical_features,
        encoded_feature_names,
    )

    scaler = build_scaler(
        encoded
    )

    verify_scaler(
        scaler,
        encoded_feature_names,
    )

    print_header("COMPLETE")

    print(
        "The missing autoencoder scaler has been "
        "reconstructed and saved."
    )

    print()
    print(
        "Next command:"
    )
    print()
    print(
        "    python backend/main.py"
    )
    print()


if __name__ == "__main__":
    main()