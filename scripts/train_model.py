import pandas as pd
import numpy as np
from pathlib import Path

from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)

INPUT = Path("data/scenario1/ctu13_features.csv")
MODEL_DIR = Path("models")

MODEL_DIR.mkdir(exist_ok=True)

print("Loading feature dataset...")

df = pd.read_csv(INPUT)

df["StartTime"] = pd.to_datetime(df["StartTime"], errors="coerce")

df = df.sort_values("StartTime").reset_index(drop=True)

FEATURES = [
    "Dur",
    "ProtocolCode",
    "Sport",
    "Dport",
    "StateCode",
    "TotPkts",
    "TotBytes",
    "SrcBytes",
    "DstBytes",
    "PacketsPerSecond",
    "BytesPerSecond",
    "AvgPacketSize",
    "SrcByteRatio",
    "DstByteRatio",
]

X = df[FEATURES]
y = df["Target"]

# --------------------------------------------------
# Time-based split
# --------------------------------------------------

split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("\n========== SPLIT ==========")

print(f"Training samples: {len(X_train):,}")
print(f"Testing samples:  {len(X_test):,}")

print("\nTraining class distribution:")
print(y_train.value_counts())

print("\nTesting class distribution:")
print(y_test.value_counts())

# --------------------------------------------------
# Handle class imbalance
# --------------------------------------------------

negative = (y_train == 0).sum()
positive = (y_train == 1).sum()

scale_pos_weight = negative / positive

print(f"\nScale pos weight: {scale_pos_weight:.2f}")

# --------------------------------------------------
# XGBoost
# --------------------------------------------------

model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1
)

print("\nTraining XGBoost...")

model.fit(X_train, y_train)

print("Training completed.")

# --------------------------------------------------
# Predictions
# --------------------------------------------------

probabilities = model.predict_proba(X_test)[:, 1]

predictions = (probabilities >= 0.5).astype(int)

# --------------------------------------------------
# Evaluation
# --------------------------------------------------

print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        predictions,
        target_names=["BENIGN", "BOTNET"],
        digits=4
    )
)

print("\n========== CONFUSION MATRIX ==========")

print(confusion_matrix(y_test, predictions))

roc_auc = roc_auc_score(y_test, probabilities)

pr_auc = average_precision_score(y_test, probabilities)

print("\n========== METRICS ==========")

print(f"ROC-AUC : {roc_auc:.4f}")
print(f"PR-AUC  : {pr_auc:.4f}")

# --------------------------------------------------
# Feature importance
# --------------------------------------------------

importance = pd.Series(
    model.feature_importances_,
    index=FEATURES
).sort_values(ascending=False)

print("\n========== FEATURE IMPORTANCE ==========")

print(importance)

# --------------------------------------------------
# Save model
# --------------------------------------------------

MODEL_PATH = MODEL_DIR / "ctu13_botnet_xgboost.json"

model.save_model(MODEL_PATH)

print(f"\nModel saved to: {MODEL_PATH}")