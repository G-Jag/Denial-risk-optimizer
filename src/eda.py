import pandas as pd
import numpy as np

CSV_PATH = "data/claim_data.csv"

def main():
    df = pd.read_csv(CSV_PATH)

    print("\n=== SHAPE ===")
    print(df.shape)

    print("\n=== COLUMNS ===")
    print(df.columns.tolist())

    print("\n=== DTYPES ===")
    print(df.dtypes)

    print("\n=== MISSING % ===")
    print((df.isna().mean() * 100).round(2).sort_values(ascending=False))

    print("\n=== UNIQUE COUNTS (top) ===")
    nunique = df.nunique().sort_values(ascending=False)
    print(nunique)

    print("\n=== OUTCOME DISTRIBUTION ===")
    print(df["Outcome"].value_counts(dropna=False))

    print("\n=== CLAIM STATUS DISTRIBUTION ===")
    print(df["Claim Status"].value_counts(dropna=False))

    print("\n=== INSURANCE TYPE DISTRIBUTION ===")
    print(df["Insurance Type"].value_counts(dropna=False))

    print("\n=== REASON CODE DISTRIBUTION ===")
    print(df["Reason Code"].value_counts(dropna=False))

    print("\n=== DATE CHECK ===")
    dos = pd.to_datetime(df["Date of Service"], errors="coerce")
    print("Date parse NA %:", (dos.isna().mean() * 100).round(2))
    print("Min date:", dos.min())
    print("Max date:", dos.max())

    print("\n=== NUMERIC STATS ===")
    for col in ["Billed Amount", "Allowed Amount", "Paid Amount"]:
        s = pd.to_numeric(df[col], errors="coerce")
        print(col, "min:", s.min(), "p50:", s.median(), "p95:", s.quantile(0.95), "max:", s.max())

    print("\n=== SANITY: Paid > Allowed ===")
    allowed = pd.to_numeric(df["Allowed Amount"], errors="coerce")
    paid = pd.to_numeric(df["Paid Amount"], errors="coerce")
    print("Percent paid > allowed:", ((paid > allowed).mean() * 100).round(2), "%")

    print("\n=== DUPLICATES ===")
    print("Duplicate rows:", df.duplicated().sum())
    print("Duplicate Claim ID:", df["Claim ID"].duplicated().sum())

    print("\n=== OUTCOME vs INSURANCE ===")
    print(pd.crosstab(df["Insurance Type"], df["Outcome"], normalize="index").round(3))

    print("\n=== OUTCOME vs PROCEDURE CODE ===")
    print(pd.crosstab(df["Procedure Code"], df["Outcome"], normalize="index").round(3))

if __name__ == "__main__":
    main()
