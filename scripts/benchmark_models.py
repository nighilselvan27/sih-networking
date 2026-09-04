import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
from xgboost import XGBClassifier
import time

INPUT = Path("data/scenario1/ctu13_features.csv")

FEATURES = [
    "Dur", "ProtocolCode", "Sport", "Dport", "StateCode",
    "TotPkts", "TotBytes", "SrcBytes", "DstBytes",
    "PacketsPerSecond", "BytesPerSecond", "AvgPacketSize",
    "SrcByteRatio", "DstByteRatio",
]

print("Loading dataset...")
df = pd.read_csv(INPUT)

df["StartTime"] = pd.to_datetime(df["StartTime"], errors="coerce")
df = df.sort_values("StartTime").reset_index(drop=True)

X = df[FEATURES]
y = df["Target"]

# Same chronological 80/20 split for every model
split = int(len(df) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]
y_train = y.iloc[:split]
y_test = y.iloc[split:]

negative = (y_train == 0).sum()
positive = (y_train == 1).sum()
scale_pos_weight = negative / positive

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        n_jobs=-1
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    ),

    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    ),

    "HistGradientBoosting": HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.05,
        max_leaf_nodes=31,
        random_state=42
    ),
}

results = []

for name, model in models.items():

    print(f"\n{'=' * 50}")
    print(f"Training: {name}")
    print(f"{'=' * 50}")

    start = time.time()

    model.fit(X_train, y_train)

    train_time = time.time() - start

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions),
        "Recall": recall_score(y_test, predictions),
        "F1": f1_score(y_test, predictions),
        "ROC-AUC": roc_auc_score(y_test, probabilities),
        "PR-AUC": average_precision_score(y_test, probabilities),
        "Train Time (sec)": train_time,
    })

    print(f"Accuracy : {results[-1]['Accuracy']:.4f}")
    print(f"Precision: {results[-1]['Precision']:.4f}")
    print(f"Recall   : {results[-1]['Recall']:.4f}")
    print(f"F1       : {results[-1]['F1']:.4f}")
    print(f"ROC-AUC  : {results[-1]['ROC-AUC']:.4f}")
    print(f"PR-AUC   : {results[-1]['PR-AUC']:.4f}")
    print(f"Time     : {train_time:.2f}s")

# --------------------------------------------------
# Final comparison
# --------------------------------------------------

results_df = pd.DataFrame(results)

print("\n\n")
print("=" * 80)
print("MODEL COMPARISON")
print("=" * 80)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

results_df.to_csv(
    "models/model_comparison.csv",
    index=False
)

print("\nSaved: models/model_comparison.csv")