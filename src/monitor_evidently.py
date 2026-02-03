# src/monitor_evidently.py
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd

DB_PATH = Path("data/claims.db")
OUT_HTML = Path("reports/evidently_drift_report.html")

COLS = [
    "Billed Amount",
    "dos_month",
    "dos_dayofweek",
    "followup_flag",
    "Insurance Type",
    "Procedure Code",
    "Diagnosis Code",
]

def psi_numeric(ref: pd.Series, cur: pd.Series, bins: int = 10) -> float:
    ref = pd.to_numeric(ref, errors="coerce").dropna()
    cur = pd.to_numeric(cur, errors="coerce").dropna()
    if len(ref) == 0 or len(cur) == 0:
        return float("nan")

    # bin edges from reference quantiles
    qs = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(ref, qs))
    if len(edges) < 3:
        return 0.0

    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)

    ref_pct = ref_counts / max(ref_counts.sum(), 1)
    cur_pct = cur_counts / max(cur_counts.sum(), 1)

    eps = 1e-6
    ref_pct = np.clip(ref_pct, eps, None)
    cur_pct = np.clip(cur_pct, eps, None)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))

def psi_categorical(ref: pd.Series, cur: pd.Series) -> float:
    ref = ref.astype(str).fillna("NA")
    cur = cur.astype(str).fillna("NA")

    ref_dist = ref.value_counts(normalize=True)
    cur_dist = cur.value_counts(normalize=True)

    cats = sorted(set(ref_dist.index).union(set(cur_dist.index)))
    eps = 1e-6

    psi = 0.0
    for c in cats:
        r = float(ref_dist.get(c, 0.0))
        u = float(cur_dist.get(c, 0.0))
        r = max(r, eps)
        u = max(u, eps)
        psi += (u - r) * np.log(u / r)
    return float(psi)

def psi_level(v: float) -> str:
    if np.isnan(v):
        return "n/a"
    if v < 0.10:
        return "low"
    if v < 0.25:
        return "moderate"
    return "high"

def main():
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql("SELECT * FROM claims_clean", conn)

    baseline = df.sample(frac=0.5, random_state=42).copy()
    current = df.drop(baseline.index).copy()

    rows = []
    for col in COLS:
        if col not in df.columns:
            continue

        if pd.api.types.is_numeric_dtype(df[col]) or col in ["Billed Amount", "dos_month", "dos_dayofweek", "followup_flag"]:
            v = psi_numeric(baseline[col], current[col], bins=10)
            col_type = "numeric"
        else:
            v = psi_categorical(baseline[col], current[col])
            col_type = "categorical"

        rows.append(
            {
                "feature": col,
                "type": col_type,
                "psi": round(v, 6) if not np.isnan(v) else "n/a",
                "drift_level": psi_level(v),
            }
        )

    summary = pd.DataFrame(rows).sort_values(
        by="psi",
        ascending=False,
        key=lambda s: pd.to_numeric(s, errors="coerce").fillna(-1),
    )

    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    html = f"""
    <html>
    <head>
      <meta charset="utf-8"/>
      <title>Drift Report (PSI)</title>
      <style>
        body {{ font-family: Arial, sans-serif; margin: 24px; }}
        h1 {{ margin-bottom: 6px; }}
        .note {{ color: #444; margin-bottom: 16px; }}
        table {{ border-collapse: collapse; width: 900px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; }}
        th {{ background: #f5f5f5; text-align: left; }}
      </style>
    </head>
    <body>
      <h1>Data Drift Report (PSI)</h1>
      <div class="note">
        PSI interpretation: &lt;0.10 low, 0.10–0.25 moderate, &gt;0.25 high drift.
        Baseline = random 50% split, Current = remaining 50% (demo).
      </div>
      {summary.to_html(index=False)}
    </body>
    </html>
    """
    OUT_HTML.write_text(html, encoding="utf-8")
    print("Saved drift report:", OUT_HTML)

if __name__ == "__main__":
    main()
