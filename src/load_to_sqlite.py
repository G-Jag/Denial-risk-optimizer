import pandas as pd
import sqlite3
from pathlib import Path

CSV_PATH = Path("data/claim_data.csv")
DB_PATH = Path("data/claims.db")

def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found at {CSV_PATH}. Put claim_data.csv inside the data/ folder.")

    df = pd.read_csv(CSV_PATH)

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB_PATH) as conn:
        df.to_sql("claims_raw", conn, if_exists="replace", index=False)

        for col in ["Claim ID", "Provider ID", "Patient ID", "Date of Service"]:
            if col in df.columns:
                try:
                    conn.execute(
                        f'CREATE INDEX IF NOT EXISTS idx_{col.replace(" ", "_")} '
                        f'ON claims_raw("{col}");'
                    )
                except Exception:
                    pass

    print("Loaded rows:", len(df))
    print("Saved DB:", DB_PATH)

if __name__ == "__main__":
    main()
