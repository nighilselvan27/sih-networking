import pandas as pd
import numpy as np
from pathlib import Path

from xgboost import XGBClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("data")
MODEL_DIR = Path("models")

TRAIN_SCENARIOS = list(range(1, 11))
TEST_SCENARIOS = [11, 12, 13]

TARGET = "Target"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD SCENARIO
# ============================================================

def load_scenario(scenario):

    path = (
        DATA_DIR
        / f"scenario{scenario}"
        / "ctu13_features.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"\nMissing feature file:\n{path}"
        )

    print(
        f"Loading Scenario {scenario}: {path}"
    )

    df = pd.read_csv(path)

    print(
        f"  Rows: {len(df):,}"
    )

    print(
        f"  Columns: {len(df.columns)}"
    )

    if TARGET not in df.columns:
        raise ValueError(
            f"Target column '{TARGET}' "
            f"not found in Scenario {scenario}"
        )

    return df


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_features(df):

    print("  Preparing features...")

    df = df.copy()

    # --------------------------------------------------------
    # Columns that must NOT enter the ML model
    # --------------------------------------------------------

    drop_columns = [
        TARGET,

        # Original label information
        "Label",
        "ThreatClass",

        # Timestamp
        "StartTime",

        # Network identity / direction
        "SrcAddr",
        "DstAddr",
        "Dir",

        # Raw categorical fields
        "Proto",
        "State",

        # Other possible non-ML identifiers
        "_Scenario",
    ]

    existing_drop = [
        col
        for col in drop_columns
        if col in df.columns
    ]

    if existing_drop:
        print(
            "  Dropping:"
        )

        for col in existing_drop:
            print(
                f"    {col}"
            )

        df = df.drop(
            columns=existing_drop
        )

    # --------------------------------------------------------
    # Convert remaining columns to numeric
    # --------------------------------------------------------

    for col in df.columns:

        if not pd.api.types.is_numeric_dtype(
            df[col]
        ):

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    # --------------------------------------------------------
    # Replace infinity
    # --------------------------------------------------------

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # --------------------------------------------------------
    # Fill missing numeric values
    # --------------------------------------------------------

    df = df.fillna(0)

    # --------------------------------------------------------
    # Final safety check
    # --------------------------------------------------------

    bad_columns = [
        col
        for col in df.columns
        if not pd.api.types.is_numeric_dtype(
            df[col]
        )
    ]

    if bad_columns:

        raise ValueError(
            "\nNon-numeric columns still remain:\n"
            + "\n".join(bad_columns)
        )

    print(
        f"  Final numeric features: {len(df.columns)}"
    )

    return df


# ============================================================
# LOAD TRAINING DATA
# ============================================================

print()
print("=" * 70)
print("CTU-13 CROSS-SCENARIO TRAINING")
print("=" * 70)

print()
print("=" * 70)
print("LOADING TRAINING SCENARIOS")
print("=" * 70)

train_frames = []

for scenario in TRAIN_SCENARIOS:

    df = load_scenario(
        scenario
    )

    df["_Scenario"] = scenario

    train_frames.append(df)

    print()


train_df = pd.concat(
    train_frames,
    ignore_index=True
)

print()
print(
    f"TOTAL TRAINING ROWS: "
    f"{len(train_df):,}"
)


# ============================================================
# LOAD TEST DATA
# ============================================================

print()
print("=" * 70)
print("LOADING TEST SCENARIOS")
print("=" * 70)

test_frames = []

for scenario in TEST_SCENARIOS:

    df = load_scenario(
        scenario
    )

    df["_Scenario"] = scenario

    test_frames.append(df)

    print()


test_df = pd.concat(
    test_frames,
    ignore_index=True
)

print()
print(
    f"TOTAL TEST ROWS: "
    f"{len(test_df):,}"
)


# ============================================================
# TARGET
# ============================================================

print()
print("=" * 70)
print("TARGET")
print("=" * 70)

y_train = pd.to_numeric(
    train_df[TARGET],
    errors="coerce"
).fillna(0).astype(int)


y_test = pd.to_numeric(
    test_df[TARGET],
    errors="coerce"
).fillna(0).astype(int)


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

print()
print("TRAINING DISTRIBUTION")

print(
    y_train.value_counts()
    .sort_index()
)


print()
print("TEST DISTRIBUTION")

print(
    y_test.value_counts()
    .sort_index()
)


# ============================================================
# PREPARE FEATURES
# ============================================================

print()
print("=" * 70)
print("PREPARING TRAINING FEATURES")
print("=" * 70)

X_train = prepare_features(
    train_df
)


print()
print("=" * 70)
print("PREPARING TEST FEATURES")
print("=" * 70)

X_test = prepare_features(
    test_df
)


# ============================================================
# ALIGN TRAIN / TEST FEATURES
# ============================================================

print()
print("=" * 70)
print("ALIGNING FEATURES")
print("=" * 70)

# Test must contain exactly the same features
# as training.

X_test = X_test.reindex(
    columns=X_train.columns,
    fill_value=0
)

print(
    f"Training features: "
    f"{len(X_train.columns)}"
)

print(
    f"Testing features : "
    f"{len(X_test.columns)}"
)


# ============================================================
# FINAL DATA TYPE CHECK
# ============================================================

print()
print("=" * 70)
print("DATA TYPE CHECK")
print("=" * 70)

non_numeric_train = X_train.select_dtypes(
    exclude=[np.number]
).columns.tolist()

non_numeric_test = X_test.select_dtypes(
    exclude=[np.number]
).columns.tolist()


if non_numeric_train:

    print(
        "ERROR: Non-numeric training columns:"
    )

    print(
        non_numeric_train
    )

    raise ValueError(
        "Training data contains non-numeric columns."
    )


if non_numeric_test:

    print(
        "ERROR: Non-numeric testing columns:"
    )

    print(
        non_numeric_test
    )

    raise ValueError(
        "Testing data contains non-numeric columns."
    )


print(
    "All features are numeric."
)


# ============================================================
# SHOW FEATURES
# ============================================================

print()
print("=" * 70)
print("FINAL FEATURES")
print("=" * 70)

for i, feature in enumerate(
    X_train.columns,
    start=1
):

    print(
        f"{i:2d}. {feature}"
    )


# ============================================================
# CLASS IMBALANCE
# ============================================================

negative = (
    y_train == 0
).sum()

positive = (
    y_train == 1
).sum()


if positive == 0:

    raise ValueError(
        "No BOTNET samples found."
    )


scale_pos_weight = (
    negative / positive
)


print()
print("=" * 70)
print("CLASS IMBALANCE")
print("=" * 70)

print(
    f"BENIGN samples : {negative:,}"
)

print(
    f"BOTNET samples : {positive:,}"
)

print(
    f"Scale positive weight: "
    f"{scale_pos_weight:.2f}"
)


# ============================================================
# TRAIN XGBOOST
# ============================================================

print()
print("=" * 70)
print("TRAINING XGBOOST")
print("=" * 70)

model = XGBClassifier(

    n_estimators=400,

    max_depth=8,

    learning_rate=0.08,

    subsample=0.85,

    colsample_bytree=0.85,

    objective="binary:logistic",

    eval_metric="logloss",

    scale_pos_weight=scale_pos_weight,

    tree_method="hist",

    random_state=42,

    n_jobs=-1
)


model.fit(
    X_train,
    y_train
)


print(
    "Training completed."
)


# ============================================================
# PREDICTION
# ============================================================

print()
print("=" * 70)
print("PREDICTING UNSEEN SCENARIOS")
print("=" * 70)

y_pred = model.predict(
    X_test
)

y_prob = model.predict_proba(
    X_test
)[:, 1]


# ============================================================
# OVERALL METRICS
# ============================================================

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

roc_auc = roc_auc_score(
    y_test,
    y_prob
)

pr_auc = average_precision_score(
    y_test,
    y_prob
)


print()
print("=" * 70)
print("OVERALL TEST RESULTS")
print("=" * 70)

print(
    f"Accuracy : {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"F1 Score : {f1:.4f}"
)

print(
    f"ROC-AUC  : {roc_auc:.4f}"
)

print(
    f"PR-AUC   : {pr_auc:.4f}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print()
print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "BENIGN",
            "BOTNET"
        ],
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print()
print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(cm)


if cm.shape == (2, 2):

    tn, fp, fn, tp = cm.ravel()

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
# PER-SCENARIO RESULTS
# ============================================================

print()
print("=" * 70)
print("PER-SCENARIO RESULTS")
print("=" * 70)


scenario_results = []


offset = 0


for i, scenario in enumerate(
    TEST_SCENARIOS
):

    scenario_df = test_frames[i]

    size = len(
        scenario_df
    )

    start = offset

    end = offset + size


    scenario_y = y_test.iloc[
        start:end
    ]

    scenario_pred = y_pred[
        start:end
    ]

    scenario_prob = y_prob[
        start:end
    ]


    scenario_accuracy = accuracy_score(
        scenario_y,
        scenario_pred
    )

    scenario_precision = precision_score(
        scenario_y,
        scenario_pred,
        zero_division=0
    )

    scenario_recall = recall_score(
        scenario_y,
        scenario_pred,
        zero_division=0
    )

    scenario_f1 = f1_score(
        scenario_y,
        scenario_pred,
        zero_division=0
    )


    if len(
        np.unique(scenario_y)
    ) == 2:

        scenario_roc = roc_auc_score(
            scenario_y,
            scenario_prob
        )

    else:

        scenario_roc = np.nan


    print()
    print(
        f"Scenario {scenario}"
    )

    print(
        "-" * 40
    )

    print(
        f"Rows      : {size:,}"
    )

    print(
        f"Accuracy  : "
        f"{scenario_accuracy:.4f}"
    )

    print(
        f"Precision : "
        f"{scenario_precision:.4f}"
    )

    print(
        f"Recall    : "
        f"{scenario_recall:.4f}"
    )

    print(
        f"F1 Score  : "
        f"{scenario_f1:.4f}"
    )

    print(
        f"ROC-AUC   : "
        f"{scenario_roc:.4f}"
    )


    scenario_results.append({

        "Scenario": scenario,

        "Rows": size,

        "Accuracy": scenario_accuracy,

        "Precision": scenario_precision,

        "Recall": scenario_recall,

        "F1": scenario_f1,

        "ROC_AUC": scenario_roc

    })


    offset = end


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print()
print("=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)


importance = pd.Series(
    model.feature_importances_,
    index=X_train.columns
).sort_values(
    ascending=False
)


print(
    importance
)


# ============================================================
# SAVE MODEL
# ============================================================

model_path = (
    MODEL_DIR
    / "ctu13_multiscenario_xgboost.json"
)


model.save_model(
    model_path
)


# ============================================================
# SAVE FEATURE LIST
# ============================================================

feature_path = (
    MODEL_DIR
    / "ctu13_multiscenario_features.txt"
)


with open(
    feature_path,
    "w",
    encoding="utf-8"
) as f:

    for feature in X_train.columns:

        f.write(
            feature + "\n"
        )


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    scenario_results
)


results_path = (
    MODEL_DIR
    / "ctu13_multiscenario_results.csv"
)


results_df.to_csv(
    results_path,
    index=False
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

prediction_df = pd.DataFrame({

    "Scenario":
        test_df["_Scenario"].values,

    "Actual":
        y_test.values,

    "Predicted":
        y_pred,

    "BotnetProbability":
        y_prob

})


prediction_path = (
    MODEL_DIR
    / "ctu13_multiscenario_predictions.csv"
)


prediction_df.to_csv(
    prediction_path,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("CROSS-SCENARIO TRAINING COMPLETE")
print("=" * 70)

print()

print(
    "TRAINING SCENARIOS:"
)

print(
    "Scenario 1 - Scenario 10"
)

print()

print(
    "TEST SCENARIOS:"
)

print(
    "Scenario 11 - Scenario 13"
)

print()

print(
    f"Training rows: "
    f"{len(train_df):,}"
)

print(
    f"Testing rows : "
    f"{len(test_df):,}"
)

print()

print(
    f"Accuracy : {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"F1 Score : {f1:.4f}"
)

print(
    f"ROC-AUC  : {roc_auc:.4f}"
)

print(
    f"PR-AUC   : {pr_auc:.4f}"
)

print()

print(
    f"Model saved to:"
)

print(
    model_path
)

print()

print(
    "Done."
)