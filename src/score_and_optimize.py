import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load
import pulp

DB_PATH = Path("data/claims.db")
MODEL_PATH = Path("models/xgb_denial_model.joblib")
OUT_PATH = Path("reports/prioritized_claims.csv")

def estimate_review_minutes(df: pd.DataFrame) -> pd.Series:
    base = np.full(len(df), 10, dtype=int)
    base += df["Insurance Type"].astype(str).str.contains("Commercial", case=False, na=False).astype(int) * 3
    base += df["followup_flag"].astype(int) * 2
    return pd.Series(np.clip(base, 6, 20), index=df.index)

def estimate_recovery(df: pd.DataFrame) -> pd.Series:
    allowed = pd.to_numeric(df["Allowed Amount"], errors="coerce").fillna(0)
    paid = pd.to_numeric(df["Paid Amount"], errors="coerce").fillna(0)
    return (allowed - paid).clip(lower=0)

def optimize_knapsack(expected_value: np.ndarray, minutes: np.ndarray, capacity: int) -> np.ndarray:
    n = len(expected_value)
    x = pulp.LpVariable.dicts("x", range(n), cat="Binary")

    prob = pulp.LpProblem("ClaimPrioritization", pulp.LpMaximize)
    prob += pulp.lpSum(expected_value[i] * x[i] for i in range(n))
    prob += pulp.lpSum(minutes[i] * x[i] for i in range(n)) <= int(capacity)

    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    return np.array([pulp.value(x[i]) for i in range(n)]) > 0.5

def main(total_minutes: int = 240):
    with sqlite3.connect(DB_PATH) as conn:
        df_clean = pd.read_sql("SELECT * FROM claims_clean", conn)
        df_model = pd.read_sql("SELECT * FROM claims_model", conn)

    pipe = load(MODEL_PATH)

    proba = pipe.predict_proba(df_model.drop(columns=["target_denied"]))[:, 1]
    df_clean["denial_prob"] = proba

    df_clean["review_minutes"] = estimate_review_minutes(df_clean)
    df_clean["estimated_recovery"] = estimate_recovery(df_clean)
    df_clean["expected_value"] = df_clean["denial_prob"] * df_clean["estimated_recovery"]
    df_clean["value_per_minute"] = df_clean["expected_value"] / df_clean["review_minutes"].replace(0, 1)

    df_clean["selected_for_review"] = optimize_knapsack(
        expected_value=df_clean["expected_value"].to_numpy(),
        minutes=df_clean["review_minutes"].to_numpy(),
        capacity=total_minutes,
    )

    df_out = df_clean.sort_values(
        ["selected_for_review", "expected_value"],
        ascending=[False, False],
    ).reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUT_PATH, index=False)

    print("Saved:", OUT_PATH)
    print("Selected:", int(df_out["selected_for_review"].sum()), "out of", len(df_out))
    print("Total expected value (selected):", float(df_out.loc[df_out["selected_for_review"], "expected_value"].sum()))
    print("Total minutes (selected):", int(df_out.loc[df_out["selected_for_review"], "review_minutes"].sum()))

if __name__ == "__main__":
    main(total_minutes=240)
