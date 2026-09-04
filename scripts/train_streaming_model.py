import os
import json
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)
from xgboost import XGBClassifier


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = "data/scenario1/ctu13_streaming_features.csv"
MODEL_FILE = "models/ctu13_streaming_xgboost.json"

TARGET = "Target"

# Columns that must NOT be used for ML
DROP_COLUMNS = [
    "Target",
    "ThreatClass",
    "StartTime",
    "SrcAddr",
    "DstAddr",
]


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("LOADING STREAMING DATASET")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)

print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")


# ============================================================
# VERIFY TARGET
# ============================================================

if TARGET not in df.columns:
    raise ValueError(
        f"Target column '{TARGET}' not found.\n"
        f"Available columns: {list(df.columns)}"
    )

print("\nTarget distribution:")
print(df[TARGET].value_counts())


# ============================================================
# PREPARE FEATURES
# ============================================================

drop_cols = [c for c in DROP_COLUMNS if c in df.columns]

X = df.drop(columns=drop_cols, errors="ignore")
y = df[TARGET].astype(int)

# Keep only numeric features
X = X.select_dtypes(include=[np.number])

# Replace invalid values
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(0)

print("\nFEATURES")
print("-" * 60)

print(f"Number of features: {X.shape[1]}")

for col in X.columns:
    print(col)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

print("\n" + "=" * 60)
print("TRAIN / TEST SPLIT")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"Training samples: {len(X_train):,}")
print(f"Testing samples:  {len(X_test):,}")

print("\nTraining distribution:")
print(y_train.value_counts())

print("\nTesting distribution:")
print(y_test.value_counts())


# ============================================================
# CLASS WEIGHT
# ============================================================

negative = (y_train == 0).sum()
positive = (y_train == 1).sum()

scale_pos_weight = negative / positive

print(f"\nScale positive weight: {scale_pos_weight:.2f}")


# ============================================================
# XGBOOST
# ============================================================

print("\n" + "=" * 60)
print("TRAINING FINAL STREAMING XGBOOST")
print("=" * 60)

model = XGBClassifier(
    n_estimators=300,
    max_depth=7,
    learning_rate=0.08,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Training completed.")


# ============================================================
# PREDICTION
# ============================================================

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["BENIGN", "BOTNET"],
        digits=4
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(confusion_matrix(y_test, y_pred))


# ============================================================
# METRICS
# ============================================================

roc_auc = roc_auc_score(y_test, y_prob)
pr_auc = average_precision_score(y_test, y_prob)

print("\n" + "=" * 60)
print("METRICS")
print("=" * 60)

print(f"ROC-AUC: {roc_auc:.4f}")
print(f"PR-AUC : {pr_auc:.4f}")


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)

importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print(importance)


# ============================================================
# SAVE MODEL
# ============================================================

os.makedirs("models", exist_ok=True)

model.save_model(MODEL_FILE)

# Save feature order
feature_file = "models/ctu13_streaming_features.json"

with open(feature_file, "w") as f:
    json.dump(list(X.columns), f, indent=2)

print("\n" + "=" * 60)
print("MODEL SAVED")
print("=" * 60)

print(f"Model:    {MODEL_FILE}")
print(f"Features: {feature_file}")