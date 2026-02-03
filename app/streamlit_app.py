# app/streamlit_app.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import sqlite3
import pandas as pd
import numpy as np
import streamlit as st
from joblib import load


ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "claims.db"
MODEL_PATH = ROOT / "models" / "xgb_denial_model.joblib"

import subprocess

def run(cmd: str):
    subprocess.check_call(cmd, shell=True)

# Build artifacts on first run (Streamlit Cloud is a fresh container)
if not DB_PATH.exists():
    run("python src/load_to_sqlite.py")
    run("python src/clean_validate.py")

if not MODEL_PATH.exists():
    run("python src/train_xgboost.py")


# -----------------------------
# Helpers
# -----------------------------
def estimate_review_minutes(df: pd.DataFrame) -> pd.Series:
    base = np.full(len(df), 10, dtype=int)
    base += df["Insurance Type"].astype(str).str.contains("Commercial", case=False, na=False).astype(int) * 3
    base += df["followup_flag"].astype(int) * 2
    return pd.Series(np.clip(base, 6, 20), index=df.index)


def estimate_recovery(df: pd.DataFrame) -> pd.Series:
    allowed = pd.to_numeric(df["Allowed Amount"], errors="coerce").fillna(0)
    paid = pd.to_numeric(df["Paid Amount"], errors="coerce").fillna(0)
    return (allowed - paid).clip(lower=0)


def fmt_pct(x):
    return f"{int(round(x * 100))}%"


def risk_badge(p):
    if p >= 0.75:
        return "🔴 High"
    if p >= 0.5:
        return "🟠 Medium"
    return "🟢 Low"


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Denial Risk Review Assistant", layout="wide")
st.title("Denial Risk Review Assistant")
st.write("Clean, simple view for business users. Dates shown without timestamps.")

# -----------------------------
# Load data
# -----------------------------
with sqlite3.connect(DB_PATH) as conn:
    df_clean = pd.read_sql("SELECT * FROM claims_clean", conn)
    df_model = pd.read_sql("SELECT * FROM claims_model", conn)


df_clean["Date of Service"] = pd.to_datetime(
    df_clean["Date of Service"], errors="coerce"
).dt.date


# -----------------------------
# Top filters
# -----------------------------
c1, c2, c3, c4 = st.columns(4)

payer_values = ["All"] + sorted(df_clean["Insurance Type"].astype(str).unique())

with c1:
    payer = st.selectbox("Payer", payer_values)

with c2:
    dos_range = st.date_input(
        "Date of Service",
        [df_clean["Date of Service"].min(), df_clean["Date of Service"].max()]
    )

with c3:
    minutes_budget = st.selectbox("Reviewer Minutes", [120, 180, 240, 300, 360, 480], index=2)

with c4:
    show_mode = st.selectbox("Show", ["Recommended only", "All"])


# -----------------------------
# Apply filters
# -----------------------------
mask = pd.Series(True, index=df_clean.index)

if payer != "All":
    mask &= df_clean["Insurance Type"] == payer

mask &= df_clean["Date of Service"].between(dos_range[0], dos_range[1])

df_f = df_clean.loc[mask].copy()
dfm_f = df_model.loc[mask].copy()

if df_f.empty:
    st.warning("No claims match the selected filters.")
    st.stop()


# -----------------------------
# Score + prioritize
# -----------------------------
pipe = load(MODEL_PATH)
df_f["denial_prob"] = pipe.predict_proba(
    dfm_f.drop(columns=["target_denied"])
)[:, 1]

df_f["review_minutes"] = estimate_review_minutes(df_f)
df_f["estimated_recovery"] = estimate_recovery(df_f)
df_f["expected_value"] = df_f["denial_prob"] * df_f["estimated_recovery"]

# Greedy selection: sort by value per minute and pick until budget is used
df_f = df_f.sort_values("expected_value", ascending=False).copy()

used = 0
selected_flags = []
for m in df_f["review_minutes"].astype(int).tolist():
    if used + m <= int(minutes_budget):
        selected_flags.append(True)
        used += m
    else:
        selected_flags.append(False)

df_f["selected_for_review"] = selected_flags

)

df_out = df_f.sort_values(
    ["selected_for_review", "expected_value"],
    ascending=[False, False]
).reset_index(drop=True)


# -----------------------------
# Summary
# -----------------------------
st.markdown(
    f"""
**Claims in scope:** {len(df_out)}  
**Recommended for review:** {df_out["selected_for_review"].sum()}
"""
)

st.divider()


# -----------------------------
# Display table (DATE ONLY)
# -----------------------------
df_view = df_out.copy()
df_view["Denial Risk"] = df_view["denial_prob"].apply(fmt_pct)
df_view["Risk Level"] = df_view["denial_prob"].apply(risk_badge)
df_view["Recommended"] = df_view["selected_for_review"].map({True: "✅ Yes", False: ""})

table_cols = [
    "Recommended",
    "Risk Level",
    "Denial Risk",
    "Claim ID",
    "Date of Service",   
    "Insurance Type",
    "Procedure Code",
    "Diagnosis Code",
    "Billed Amount",
    "Follow-up Required",
]

if show_mode == "Recommended only":
    df_view = df_view[df_view["selected_for_review"]]

st.subheader("Review List")
st.dataframe(df_view[table_cols].head(50), use_container_width=True)

st.download_button(
    "Download CSV",
    df_view[table_cols].to_csv(index=False),
    file_name="review_list.csv"
)
