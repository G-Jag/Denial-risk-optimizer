import json
import sqlite3
from pathlib import Path

import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, classification_report
from xgboost import XGBClassifier

DB_PATH = Path("data/claims.db")
MODEL_PATH = Path("models/xgb_denial_model.joblib")
METRICS_PATH = Path("reports/model_metrics.json")

def main():
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql("SELECT * FROM claims_model", conn)

    y = df["target_denied"].astype(int)
    X = df.drop(columns=["target_denied"])

    cat_cols = ["Insurance Type", "Procedure Code", "Diagnosis Code"]
    num_cols = ["Billed Amount", "dos_month", "dos_dayofweek", "followup_flag"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ],
        remainder="drop",
    )

    model = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=42,
    )

    pipe = Pipeline([("pre", pre), ("model", model)])
    pipe.fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "f1": float(f1_score(y_test, preds)),
        "threshold": 0.5,
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
        "classification_report": classification_report(y_test, preds, output_dict=True),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "features": {"categorical": cat_cols, "numeric": num_cols},
        "target_definition": "Denied=1, Paid/Partially Paid=0",
        "leakage_excluded": [
            "Allowed Amount", "Paid Amount", "Claim Status", "Reason Code", "AR Status",
            "Claim ID", "Patient ID", "Provider ID"
        ],
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    dump(pipe, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("Saved model:", MODEL_PATH)
    print("Saved metrics:", METRICS_PATH)
    print("ROC AUC:", metrics["roc_auc"], "F1:", metrics["f1"])

if __name__ == "__main__":
    main()
