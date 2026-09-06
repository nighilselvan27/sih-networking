from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy import sparse


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

SCHEMA_PATH = MODELS_DIR / "autoencoder_features.json"
OUTPUT_PATH = MODELS_DIR / "autoencoder_scaler.joblib"

TRAINING_SCENARIOS = range(1, 11)

MAX_BENIGN_ROWS_PER_SCENARIO = 30000
RANDOM_STATE = 42


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("CTU-13 AUTOENCODER SCALER CREATION")
print("=" * 70)

print(f"Project root : {PROJECT_ROOT}")
print(f"Data directory : {DATA_DIR}")
print(f"Models directory : {MODELS_DIR}")
print(f"Output : {OUTPUT_PATH}")


# ============================================================
# CHECK FILES
# ============================================================

if not SCHEMA_PATH.exists():
    raise FileNotFoundError(
        f"Schema not found:\n{SCHEMA_PATH}"
    )


# ============================================================
# LOAD SCHEMA
# ============================================================

print("\n" + "=" * 70)
print("LOADING AUTOENCODER SCHEMA")
print("=" * 70)

with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
    schema = json.load(f)

numeric_features = schema["numeric_features"]
categorical_features = schema["categorical_features"]
encoded_feature_names = schema["encoded_feature_names"]

expected_dimension = schema["input_dimension"]

print(f"Numeric features    : {len(numeric_features)}")
print(f"Categorical features: {len(categorical_features)}")
print(f"Encoded features    : {len(encoded_feature_names)}")
print(f"Input dimension     : {expected_dimension}")


# ============================================================
# VALIDATE SCHEMA
# ============================================================

if len(encoded_feature_names) != expected_dimension:
    raise ValueError(
        f"Schema mismatch:\n"
        f"Encoded features = {len(encoded_feature_names)}\n"
        f"Expected = {expected_dimension}"
    )


# ============================================================
# LOAD TRAINING DATA
# ============================================================

print("\n" + "=" * 70)
print("LOADING BENIGN TRAINING DATA")
print("=" * 70)

all_encoded_parts = []

for scenario in TRAINING_SCENARIOS:

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

    print(f"Rows loaded: {len(df):,}")

    # --------------------------------------------------------
    # Select benign traffic
    # --------------------------------------------------------

    if "Target" in df.columns:
        benign = df[df["Target"] == 0].copy()

    elif "Label" in df.columns:
        benign = df[
            df["Label"]
            .astype(str)
            .str.lower()
            .isin(["benign", "normal"])
        ].copy()

    else:
        raise ValueError(
            "Could not find Target or Label column."
        )

    print(f"Benign rows: {len(benign):,}")

    # --------------------------------------------------------
    # Sample
    # --------------------------------------------------------

    if len(benign) > MAX_BENIGN_ROWS_PER_SCENARIO:

        benign = benign.sample(
            n=MAX_BENIGN_ROWS_PER_SCENARIO,
            random_state=RANDOM_STATE
        )

        print(
            f"Sampled: {len(benign):,}"
        )

    # --------------------------------------------------------
    # Check required columns
    # --------------------------------------------------------

    required_columns = (
        numeric_features
        + categorical_features
    )

    missing = [
        col
        for col in required_columns
        if col not in benign.columns
    ]

    if missing:
        raise ValueError(
            f"Scenario {scenario} is missing columns:\n"
            + "\n".join(missing)
        )

    # --------------------------------------------------------
    # Numeric features
    # --------------------------------------------------------

    numeric = benign[numeric_features].copy()

    numeric = numeric.apply(
        pd.to_numeric,
        errors="coerce"
    )

    numeric = numeric.fillna(0)

    # --------------------------------------------------------
    # Categorical features
    # --------------------------------------------------------

    categorical = benign[categorical_features].copy()

    for col in categorical_features:
        categorical[col] = (
            categorical[col]
            .fillna("UNKNOWN")
            .astype(str)
        )

    # --------------------------------------------------------
    # One-hot encoding
    #
    # IMPORTANT:
    # We use the categories from the schema by explicitly
    # creating the encoded columns.
    # --------------------------------------------------------

    encoded = pd.DataFrame(
        0.0,
        index=benign.index,
        columns=encoded_feature_names
    )

    # Numeric columns
    for col in numeric_features:

        encoded[col] = numeric[col].values

    # Categorical columns
    for col in categorical_features:

        for value in categorical[col].unique():

            encoded_name = f"{col}_{value}"

            if encoded_name in encoded.columns:
                encoded.loc[
                    categorical[col] == value,
                    encoded_name
                ] = 1.0

    # --------------------------------------------------------
    # Ensure exact feature order
    # --------------------------------------------------------

    encoded = encoded[
        encoded_feature_names
    ]

    all_encoded_parts.append(encoded)

    print(
        f"Encoded matrix: "
        f"{encoded.shape[0]:,} x {encoded.shape[1]}"
    )


# ============================================================
# COMBINE TRAINING DATA
# ============================================================

print("\n" + "=" * 70)
print("COMBINING TRAINING DATA")
print("=" * 70)

X = pd.concat(
    all_encoded_parts,
    axis=0,
    ignore_index=True
)

print(f"Total rows    : {len(X):,}")
print(f"Total features: {X.shape[1]}")


# ============================================================
# FINAL VALIDATION
# ============================================================

if X.shape[1] != expected_dimension:

    raise ValueError(
        f"Final feature dimension mismatch.\n"
        f"Got: {X.shape[1]}\n"
        f"Expected: {expected_dimension}"
    )

print("\nFeature dimension validation: OK")


# ============================================================
# FIT SCALER
# ============================================================

print("\n" + "=" * 70)
print("FITTING STANDARD SCALER")
print("=" * 70)

scaler = StandardScaler()

scaler.fit(X.values)

print("Scaler fitted successfully.")

print(
    f"Scaler features: "
    f"{scaler.n_features_in_}"
)


# ============================================================
# VALIDATE SCALER
# ============================================================

if scaler.n_features_in_ != expected_dimension:

    raise ValueError(
        f"Scaler dimension mismatch.\n"
        f"Scaler: {scaler.n_features_in_}\n"
        f"Expected: {expected_dimension}"
    )


# ============================================================
# SAVE
# ============================================================

MODELS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

joblib.dump(
    scaler,
    OUTPUT_PATH
)

print("\n" + "=" * 70)
print("SCALER SAVED")
print("=" * 70)

print(f"File: {OUTPUT_PATH}")

print(
    f"Size: "
    f"{OUTPUT_PATH.stat().st_size:,} bytes"
)


# ============================================================
# RELOAD TEST
# ============================================================

print("\n" + "=" * 70)
print("VERIFYING SAVED SCALER")
print("=" * 70)

loaded_scaler = joblib.load(
    OUTPUT_PATH
)

print(
    f"Loaded scaler features: "
    f"{loaded_scaler.n_features_in_}"
)

if loaded_scaler.n_features_in_ != expected_dimension:

    raise ValueError(
        "Saved scaler failed dimension validation."
    )

print("\n" + "=" * 70)
print("SUCCESS")
print("=" * 70)

print(
    "\nautoencoder_scaler.joblib "
    "was created successfully."
)

print(
    "\nFinal artifact:"
)

print(
    OUTPUT_PATH
)