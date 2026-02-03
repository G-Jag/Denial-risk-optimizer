# app/streamlit_app.py
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data" / "claim_data.csv"

# -----------------------------
# Helpers
# -----------------------------
def fmt_pct(x):
    return f"{int(round(float(x) * 100))}%"

def risk_badge(p):
    p = float(p)
    if p >= 0.75:
        return "🔴 High"
    if p >= 0.50:
        return "🟠 Medium"
    return "🟢 Low"

def estimate_review_minutes(df: pd.DataFrame) -> pd.Series:
    base = np.full(len(df), 10, dtype=int)
    base += df["Insurance Type"].astype(str).str.contains("Commercial", case=False, na=False).astype(int) * 3
    base += df["Follow-up Required"].astype(str).str.lower().eq("yes").astype(int) * 2
    return pd.Series(np.clip(base, 6, 20), index=df.index)

def greedy_select(df_sorted: pd.DataFrame, minutes_budget: int) -> pd.Series:
    used = 0
    flags = []
    for m in df_sorted["review_minutes"].astype(int).tolist():
        if used + m <= int(minutes_budget):
            flags.append(True)
            used += m
        else:
            flags.append(False)
    return pd.Series(flags, index=df_sorted.index)

@st.cache_data
def load_data():
    if not CSV_PATH.exists():
        raise FileNotFoundError("data/claim_data.csv not found in the repo.")
    df = pd.read_csv(CSV_PATH)
    # Date only
    df["Date of Service"] = pd.to_datetime(df["Date of Service"], errors="coerce").dt.date
    return df

@st.cache_resource
def train_model(df: pd.DataFrame):
    # Target: denied vs not denied
    y = df["Outcome"].astype(str).str.lower().str.strip().eq("denied").astype(int)

    # Model-safe, pre-decision features only
    X = df[[
        "Insurance Type",
        "Procedure Code",
        "Diagnosis Code",
        "Billed Amount",
        "Date of Service",
        "Follow-up Required",
    ]].copy()

    # Convert date to simple numeric signals (still pre-decision)
    dos = pd.to_datetime(X["Date of Service"], errors="coerce")
    X["dos_month"] = dos.dt.month.fillna(0).astype(int)
    X["dos_dayofweek"] = dos.dt.dayofweek.fillna(0).astype(int)
    X = X.drop(columns=["Date of Service"])

    cat_cols = ["Insurance Type", "Diagnosis Code", "Follow-up Required"]
    num_cols = ["Procedure Code", "Billed Amount", "dos_month", "dos_dayofweek"]

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
        eval_metric="logloss",
        n_jobs=2,
    )

    pipe = Pipeline([("pre", pre), ("model", model)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    pipe.fit(X_train, y_train)
    p = pipe.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, p)

    return pipe, float(auc)


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Denial Risk Review Assistant", layout="wide")
st.title("Denial Risk Review Assistant")
st.caption("Clean, simple view for business users. Dates shown without timestamps.")

df = load_data()
pipe, auc = train_model(df)

# Top filters (minimal)
c1, c2, c3, c4 = st.columns([2, 2, 2, 2])

payer_values = ["All"] + sorted(df["Insurance Type"].astype(str).unique().tolist())
min_dos = df["Date of Service"].min()
max_dos = df["Date of Service"].max()

with c1:
    payer = st.selectbox("Payer", payer_values, index=0)

with c2:
    dos_range = st.date_input("Date of Service", value=[min_dos, max_dos])

with c3:
    minutes_budget = st.selectbox("Reviewer Minutes", [120, 180, 240, 300, 360, 480, 600], index=2)

with c4:
    show_mode = st.selectbox("Show", ["Recommended only", "All"], index=0)

# Apply filters
mask = pd.Series(True, index=df.index)

if payer != "All":
    mask &= df["Insurance Type"].astype(str).eq(payer)

mask &= df["Date of Service"].between(dos_range[0], dos_range[1])

df_f = df.loc[mask].copy()
if df_f.empty:
    st.warning("No claims match the selected filters.")
    st.stop()

# Build same features for scoring
Xf = df_f[[
    "Insurance Type",
    "Procedure Code",
    "Diagnosis Code",
    "Billed Amount",
    "Date of Service",
    "Follow-up Required",
]].copy()

dos = pd.to_datetime(Xf["Date of Service"], errors="coerce")
Xf["dos_month"] = dos.dt.month.fillna(0).astype(int)
Xf["dos_dayofweek"] = dos.dt.dayofweek.fillna(0).astype(int)
Xf = Xf.drop(columns=["Date of Service"])

df_f["denial_prob"] = pipe.predict_proba(Xf)[:, 1]
df_f["review_minutes"] = estimate_review_minutes(df_f)

# Simple prioritization signal (no paid/allowed shown; no leakage)
df_f["priority_score"] = df_f["denial_prob"] / df_f["review_minutes"].replace(0, 1)

df_f = df_f.sort_values("priority_score", ascending=False).copy()
df_f["selected_for_review"] = greedy_select(df_f, int(minutes_budget))

# Summary (minimal)
st.write(f"Claims in scope: **{len(df_f)}**")
st.write(f"Recommended for review: **{int(df_f['selected_for_review'].sum())}**")

# Table (only model-used business fields + outputs)
df_view = df_f.copy()
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
    data=df_view[table_cols].to_csv(index=False).encode("utf-8"),
    file_name="review_list.csv",
    mime="text/csv",
)

with st.expander("Model info (for reference)"):
    st.write(f"Validation ROC AUC (holdout split): {auc:.3f}")
