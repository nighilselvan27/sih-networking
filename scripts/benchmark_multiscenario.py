import pandas as pd
import numpy as np
import time
from pathlib import Path

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
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
# LOAD DATA
# ============================================================

def load_scenario(scenario):

    path = (
        DATA_DIR
        / f"scenario{scenario}"
        / "ctu13_features.csv"
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Missing file:\n{path}"
        )

    print(
        f"Loading Scenario {scenario}: "
        f"{path}"
    )

    df = pd.read_csv(path)

    print(
        f"  Rows: {len(df):,}"
    )

    return df


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_features(df):

    df = df.copy()

    # Columns that must not be used by ML
    drop_columns = [

        TARGET,

        "Label",
        "ThreatClass",

        "StartTime",

        "SrcAddr",
        "DstAddr",

        "Dir",

        "Proto",
        "State",

        "_Scenario",
    ]

    existing = [
        col
        for col in drop_columns
        if col in df.columns
    ]

    df = df.drop(
        columns=existing
    )

    # Convert everything remaining to numeric
    for col in df.columns:

        if not pd.api.types.is_numeric_dtype(
            df[col]
        ):

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    # Remove infinity
    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Missing values
    df = df.fillna(0)

    # Safety check
    bad_columns = (
        df.select_dtypes(
            exclude=[np.number]
        ).columns.tolist()
    )

    if bad_columns:

        raise ValueError(
            "Non-numeric columns found:\n"
            + str(bad_columns)
        )

    return df


# ============================================================
# LOAD TRAINING DATA
# ============================================================

print()
print("=" * 70)
print("CTU-13 MULTI-SCENARIO MODEL BENCHMARK")
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

y_train = (
    pd.to_numeric(
        train_df[TARGET],
        errors="coerce"
    )
    .fillna(0)
    .astype(int)
)

y_test = (
    pd.to_numeric(
        test_df[TARGET],
        errors="coerce"
    )
    .fillna(0)
    .astype(int)
)


# ============================================================
# PREPARE FEATURES
# ============================================================

print()
print("=" * 70)
print("PREPARING FEATURES")
print("=" * 70)

X_train = prepare_features(
    train_df
)

X_test = prepare_features(
    test_df
)


# ============================================================
# ALIGN FEATURES
# ============================================================

X_test = X_test.reindex(
    columns=X_train.columns,
    fill_value=0
)


print()
print(
    f"Features: "
    f"{len(X_train.columns)}"
)

print(
    "All features are numeric."
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

scale_pos_weight = (
    negative / positive
)


print()
print("=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)

print(
    f"BENIGN : {negative:,}"
)

print(
    f"BOTNET : {positive:,}"
)

print(
    f"Scale positive weight: "
    f"{scale_pos_weight:.2f}"
)


# ============================================================
# DEFINE MODELS
# ============================================================

models = {

    "XGBoost": XGBClassifier(

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
    ),


    "Random Forest": RandomForestClassifier(

        n_estimators=250,

        max_depth=20,

        min_samples_leaf=2,

        class_weight="balanced",

        random_state=42,

        n_jobs=-1
    ),


    "HistGradientBoosting":
        HistGradientBoostingClassifier(

            max_iter=400,

            learning_rate=0.08,

            max_leaf_nodes=63,

            l2_regularization=1.0,

            random_state=42
        )
}


# ============================================================
# STORAGE
# ============================================================

overall_results = []

scenario_results = []


# ============================================================
# TRAIN + TEST EACH MODEL
# ============================================================

for model_name, model in models.items():

    print()
    print("=" * 70)
    print(
        f"TRAINING: {model_name}"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    start_time = time.time()

    model.fit(
        X_train,
        y_train
    )

    train_time = (
        time.time()
        - start_time
    )

    print(
        f"Training completed in "
        f"{train_time:.2f} seconds"
    )

    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    print(
        "Running test prediction..."
    )

    prediction_start = time.time()

    y_pred = model.predict(
        X_test
    )

    y_prob = model.predict_proba(
        X_test
    )[:, 1]

    prediction_time = (
        time.time()
        - prediction_start
    )

    # --------------------------------------------------------
    # OVERALL METRICS
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # PRINT OVERALL
    # --------------------------------------------------------

    print()
    print(
        f"{model_name} RESULTS"
    )

    print(
        "-" * 50
    )

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

    print(
        f"Train Time: {train_time:.2f}s"
    )

    print(
        f"Prediction Time: "
        f"{prediction_time:.2f}s"
    )

    # --------------------------------------------------------
    # STORE OVERALL
    # --------------------------------------------------------

    overall_results.append({

        "Model": model_name,

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1": f1,

        "ROC_AUC": roc_auc,

        "PR_AUC": pr_auc,

        "Train_Time_Sec": train_time,

        "Prediction_Time_Sec":
            prediction_time
    })

    # --------------------------------------------------------
    # PER SCENARIO
    # --------------------------------------------------------

    offset = 0

    for i, scenario in enumerate(
        TEST_SCENARIOS
    ):

        scenario_df = (
            test_frames[i]
        )

        size = len(
            scenario_df
        )

        start = offset

        end = (
            offset
            + size
        )

        scenario_y = (
            y_test.iloc[
                start:end
            ]
        )

        scenario_pred = (
            y_pred[
                start:end
            ]
        )

        scenario_prob = (
            y_prob[
                start:end
            ]
        )

        scenario_accuracy = (
            accuracy_score(
                scenario_y,
                scenario_pred
            )
        )

        scenario_precision = (
            precision_score(
                scenario_y,
                scenario_pred,
                zero_division=0
            )
        )

        scenario_recall = (
            recall_score(
                scenario_y,
                scenario_pred,
                zero_division=0
            )
        )

        scenario_f1 = (
            f1_score(
                scenario_y,
                scenario_pred,
                zero_division=0
            )
        )

        if len(
            np.unique(
                scenario_y
            )
        ) == 2:

            scenario_roc_auc = (
                roc_auc_score(
                    scenario_y,
                    scenario_prob
                )
            )

        else:

            scenario_roc_auc = np.nan

        scenario_results.append({

            "Model": model_name,

            "Scenario": scenario,

            "Rows": size,

            "Accuracy":
                scenario_accuracy,

            "Precision":
                scenario_precision,

            "Recall":
                scenario_recall,

            "F1":
                scenario_f1,

            "ROC_AUC":
                scenario_roc_auc
        })

        offset = end

    # --------------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------------

    safe_name = (
        model_name
        .lower()
        .replace(" ", "_")
    )

    model_path = (
        MODEL_DIR
        / f"multiscenario_{safe_name}.joblib"
    )

    try:

        import joblib

        joblib.dump(
            model,
            model_path
        )

        print()
        print(
            f"Model saved: "
            f"{model_path}"
        )

    except Exception as e:

        print(
            f"Could not save model: "
            f"{e}"
        )


# ============================================================
# OVERALL COMPARISON
# ============================================================

results_df = pd.DataFrame(
    overall_results
)

results_df = results_df.sort_values(
    by="F1",
    ascending=False
)


print()
print("=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)

print()

print(
    results_df.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}"
    )
)


# ============================================================
# SCENARIO COMPARISON
# ============================================================

scenario_df = pd.DataFrame(
    scenario_results
)


print()
print("=" * 70)
print("PER-SCENARIO COMPARISON")
print("=" * 70)


for scenario in TEST_SCENARIOS:

    print()
    print(
        f"SCENARIO {scenario}"
    )

    print(
        "-" * 70
    )

    temp = (
        scenario_df[
            scenario_df["Scenario"]
            == scenario
        ]
    )

    print(
        temp.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}"
        )
    )


# ============================================================
# SAVE RESULTS
# ============================================================

overall_path = (
    MODEL_DIR
    / "multiscenario_benchmark.csv"
)

scenario_path = (
    MODEL_DIR
    / "multiscenario_benchmark_per_scenario.csv"
)


results_df.to_csv(
    overall_path,
    index=False
)

scenario_df.to_csv(
    scenario_path,
    index=False
)


# ============================================================
# BEST MODEL
# ============================================================

best_model = (
    results_df.iloc[0]
)


print()
print("=" * 70)
print("BEST MODEL")
print("=" * 70)

print()

print(
    f"Model: "
    f"{best_model['Model']}"
)

print(
    f"Accuracy: "
    f"{best_model['Accuracy']:.4f}"
)

print(
    f"Precision: "
    f"{best_model['Precision']:.4f}"
)

print(
    f"Recall: "
    f"{best_model['Recall']:.4f}"
)

print(
    f"F1: "
    f"{best_model['F1']:.4f}"
)

print(
    f"ROC-AUC: "
    f"{best_model['ROC_AUC']:.4f}"
)

print(
    f"PR-AUC: "
    f"{best_model['PR_AUC']:.4f}"
)

print()
print(
    f"Overall results saved to:"
)

print(
    overall_path
)

print()

print(
    f"Per-scenario results saved to:"
)

print(
    scenario_path
)

print()
print("=" * 70)
print("BENCHMARK COMPLETE")
print("=" * 70)