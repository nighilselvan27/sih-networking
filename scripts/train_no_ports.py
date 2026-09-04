import pandas as pd
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)

INPUT = Path("data/scenario1/ctu13_features.csv")

print("Loading dataset...")
df = pd.read_csv(INPUT)

df["StartTime"] = pd.to_datetime(df["StartTime"], errors="coerce")
df = df.sort_values("StartTime").reset_index(drop=True)

# IMPORTANT:
# Sport and Dport deliberately removed
FEATURES = [
    "Dur",
    "ProtocolCode",
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

# Same chronological split as before
split = int(len(df) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

negative = (y_train == 0).sum()
positive = (y_train == 1).sum()

scale_pos_weight = negative / positive

print("\n========== DATA ==========")
print(f"Training samples: {len(X_train):,}")
print(f"Testing samples:  {len(X_test):,}")
print(f"Scale pos weight: {scale_pos_weight:.2f}")

print("\n========== FEATURES ==========")
print(FEATURES)

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

print("\nTraining XGBoost WITHOUT PORT FEATURES...")

model.fit(X_train, y_train)

print("Training completed.")

probabilities = model.predict_proba(X_test)[:, 1]
predictions = (probabilities >= 0.5).astype(int)

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

print("\n========== FEATURE IMPORTANCE ==========")

importance = (
    pd.Series(
        model.feature_importances_,
        index=FEATURES
    )
    .sort_values(ascending=False)
)

print(importance)

model.save_model("models/ctu13_botnet_xgboost_no_ports.json")

print("\nModel saved:")
print("models/ctu13_botnet_xgboost_no_ports.json")