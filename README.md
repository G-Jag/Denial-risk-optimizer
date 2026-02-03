# Denial Risk Prediction & Reviewer Prioritization System

This project demonstrates an **end-to-end data science and optimization pipeline** for healthcare revenue cycle teams.  
The goal is to **predict claim denial risk early** and **prioritize which claims to review first**, given limited human reviewer time.

The system combines:
- Machine learning (denial risk prediction)
- Business rules (estimated recovery)
- Optimization (reviewer time constraints)
- A simple, stakeholder-friendly Streamlit dashboard

---

## 🎯 Business Problem

Revenue cycle teams face two constraints:
1. **Many claims are at risk of denial**
2. **Reviewer time is limited**

Reviewing all claims is not feasible.  
This system answers:

> **Which claims should we review first to maximize impact, given limited reviewer time?**

---

## 🧠 Solution Overview

1. **Predict denial risk** using a machine learning model trained on pre-decision claim attributes  
2. **Estimate potential recovery value** for each claim  
3. **Optimize claim selection** using a reviewer time budget  
4. **Present results** in a clean, non-technical dashboard for business users  

---

## 🏗️ Architecture Diagram

               ┌────────────────────────┐
               │  Raw Claims Data (CSV) │
               │  claim_data.csv        │
               └────────────┬───────────┘
                            │
                            ▼
             ┌─────────────────────────────┐
             │ Data Cleaning & Validation   │
             │ (schema, types, rules)       │
             └────────────┬────────────────┘
                          │
                          ▼
             ┌─────────────────────────────┐
             │ Feature Engineering          │
             │ (model-safe features only)  │
             └────────────┬────────────────┘
                          │
                          ▼
             ┌─────────────────────────────┐
             │ XGBoost Denial Risk Model    │
             │ (Denied vs Not Denied)       │
             └────────────┬────────────────┘
                          │
                          ▼
    ┌───────────────────────────────┐
    │ Denial Probability (per claim)│
    └────────────┬──────────────────┘
                 │
                 ▼
    ┌───────────────────────────────┐
    │ Estimated Recovery Value       │
    │ (business rule based)          │
    └────────────┬──────────────────┘
                 │
                 ▼
    ┌───────────────────────────────┐
    │ Optimization (Knapsack)        │
    │ Maximize value under time cap  │
    └────────────┬──────────────────┘
                 │
                 ▼
    ┌───────────────────────────────┐
    │ Streamlit Dashboard            │
    │ - Risk %                       │
    │ - Recommended claims           │
    │ - Simple explanations          │
    └───────────────────────────────┘

---

## 📊 Model Details

**Model**
- Algorithm: XGBoost Classifier
- Target: `Denied` vs `Not Denied`
- Output: Probability of denial (shown as %)

**Features used (pre-decision only)**
- Insurance Type
- Procedure Code
- Diagnosis Code
- Billed Amount
- Date of Service
- Follow-up Required flag

> ⚠️ Post-adjudication fields (Reason Code, Claim Status, Paid Amount) are **not used** for prediction to avoid data leakage.

---

## ⏱️ Reviewer Minutes (Optimization Constraint)

Reviewer Minutes represent **human review capacity**.

- Each claim takes time to review (estimated via simple business rules)
- Reviewers have a fixed daily time budget
- The system selects claims that **maximize expected recovery per minute**

This ensures:
- High-risk alone ≠ automatically reviewed
- Claims are selected based on **risk × value × time**

---

## 🖥️ Dashboard Highlights

The Streamlit dashboard is designed for **non-technical stakeholders**:

- Denial risk shown as **percent**
- Clear **High / Medium / Low** risk labels
- Simple filters:
  - Payer
  - Date of Service
  - Reviewer time budget
- Clear **Recommended / Not Recommended** decision
- Plain-language explanations:
  - “Why risk is high”
  - “Why risk is low”

---

## 📁 Project Structure


---

## ▶️ How to Run Locally

```bash
# create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# install dependencies
pip install -r requirements.txt

# run pipeline
python src/load_to_sqlite.py
python src/clean_validate.py
python src/train_xgboost.py
python src/score_and_optimize.py

# launch dashboard
streamlit run app/streamlit_app.py
