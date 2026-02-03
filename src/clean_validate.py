import pandas as pd
import sqlite3
from pathlib import Path

DB_PATH = Path("data/claims.db")

def main():
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql("SELECT * FROM claims_raw", conn)

    str_cols = [
        "Claim ID", "Provider ID", "Patient ID", "Procedure Code", "Diagnosis Code",
        "Insurance Type", "Claim Status", "Reason Code", "Follow-up Required",
        "AR Status", "Outcome"
    ]
    for c in str_cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    df["Date of Service"] = pd.to_datetime(df["Date of Service"], errors="coerce")
    df["dos_month"] = df["Date of Service"].dt.month.fillna(0).astype(int)
    df["dos_dayofweek"] = df["Date of Service"].dt.dayofweek.fillna(0).astype(int)

    for c in ["Billed Amount", "Allowed Amount", "Paid Amount"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").clip(lower=0)

    df["followup_flag"] = (
        df["Follow-up Required"]
        .astype(str).str.strip().str.lower()
        .isin(["yes", "y", "true", "1"])
        .astype(int)
    )

    df["Outcome"] = df["Outcome"].astype(str).str.strip().str.lower()
    df = df[df["Outcome"].isin(["denied", "paid", "partially paid"])].copy()

    df["target_denied"] = (df["Outcome"] == "denied").astype(int)

    df = df.dropna(subset=["Insurance Type", "Procedure Code", "Diagnosis Code", "Billed Amount"])

    feature_cols = [
        "Insurance Type",
        "Procedure Code",
        "Diagnosis Code",
        "Billed Amount",
        "dos_month",
        "dos_dayofweek",
        "followup_flag",
    ]
    df_model = df[feature_cols + ["target_denied"]].copy()

    with sqlite3.connect(DB_PATH) as conn:
        df.to_sql("claims_clean", conn, if_exists="replace", index=False)
        df_model.to_sql("claims_model", conn, if_exists="replace", index=False)

    print("Cleaned rows:", len(df))
    print("Saved tables: claims_clean, claims_model")

if __name__ == "__main__":
    main()
