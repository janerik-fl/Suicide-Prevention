# Visuals from the provided datasets using pandas, NumPy, scikit-learn, and matplotlib.
# Requirements from the environment: use matplotlib (not seaborn), single chart per figure, and do not set colors.
import os
import math
import json
from typing import Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Try to import scikit-learn for metrics; handle if missing gracefully
try:
    from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
    from sklearn.calibration import calibration_curve
    SKLEARN_OK = True
except Exception as e:
    SKLEARN_OK = False

# Paths (as provided by you earlier)
data_path = "data/synthetic_uruguay_attempts.csv"
preds_path = "data/preds_logreg.csv"
outdir = "data/figs"
os.makedirs(outdir, exist_ok=True)

# ---------- Helper ----------
def save_show(fig, name):
    path = os.path.join(outdir, name)
    fig.savefig(path, bbox_inches="tight", dpi=144)
    plt.close(fig)
    return path

# ---------- Load main synthetic dataset ----------
df = pd.read_csv(data_path, low_memory=False)

# Parse dates and derive temporal fields
for c in ["Suicide_Attempt_Date", "Second_Attempt_Date_Same_Year", "Date_of_Birth", "Date_of_Registration"]:
    if c in df.columns:
        df[c] = pd.to_datetime(df[c], errors="coerce")

df["Repeated_Same_Year"] = df["Second_Attempt_Date_Same_Year"].notna().astype(int)
df["Attempt_Date"] = df["Suicide_Attempt_Date"]

# Derive month and weekday
df["Month"] = df["Attempt_Date"].dt.month
weekday_map = {0:"Monday",1:"Tuesday",2:"Wednesday",3:"Thursday",4:"Friday",5:"Saturday",6:"Sunday"}
df["Weekday"] = df["Attempt_Date"].dt.weekday.map(weekday_map)

# ---------- 1) Attempts by Sex ----------
sex_counts = df["Sex"].value_counts().sort_index()
fig = plt.figure()
sex_counts.plot(kind="bar")
plt.title("Attempt Presentations by Sex")
plt.xlabel("Sex")
plt.ylabel("Count")
sex_bar_path = save_show(fig, "attempts_by_sex.png")

# ---------- 2) Method distribution (overall) ----------
if "Method_Used" in df.columns:
    method_counts = df["Method_Used"].value_counts().sort_values(ascending=False)
    fig = plt.figure(figsize=(8, 5))
    method_counts.plot(kind="bar")
    plt.title("Method of Attempt (Overall)")
    plt.xlabel("Method")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    method_overall_path = save_show(fig, "method_overall.png")
else:
    method_overall_path = None

# ---------- 3) Method distribution by Sex (grouped bars) ----------
if "Method_Used" in df.columns and "Sex" in df.columns:
    pivot_method_sex = df.pivot_table(index="Method_Used", columns="Sex", values="ID_Number", aggfunc="count").fillna(0)
    pivot_method_sex = pivot_method_sex.loc[pivot_method_sex.sum(axis=1).sort_values(ascending=False).index]
    fig = plt.figure(figsize=(8, 5))
    pivot_method_sex.plot(kind="bar", ax=plt.gca())
    plt.title("Method of Attempt by Sex")
    plt.xlabel("Method")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    method_by_sex_path = save_show(fig, "method_by_sex.png")
else:
    method_by_sex_path = None

# ---------- 4) Age distribution ----------
if "Age_at_Attempt" in df.columns:
    fig = plt.figure()
    plt.hist(df["Age_at_Attempt"].dropna(), bins=25)
    plt.title("Age at Attempt (Histogram)")
    plt.xlabel("Age (years)")
    plt.ylabel("Frequency")
    age_hist_path = save_show(fig, "age_hist.png")
else:
    age_hist_path = None

# ---------- 5) Attempts by Month (across all years) ----------
month_counts = df["Month"].value_counts().sort_index()
fig = plt.figure()
month_counts.plot(kind="bar")
plt.title("Attempt Presentations by Month (All Years)")
plt.xlabel("Month")
plt.ylabel("Count")
attempts_by_month_path = save_show(fig, "attempts_by_month.png")

# ---------- 6) Attempts by Weekday ----------
weekday_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
weekday_counts = df["Weekday"].value_counts().reindex(weekday_order)
fig = plt.figure()
weekday_counts.plot(kind="bar")
plt.title("Attempt Presentations by Day of Week")
plt.xlabel("Day of Week")
plt.ylabel("Count")
attempts_by_weekday_path = save_show(fig, "attempts_by_weekday.png")

# ---------- 7) Repetition rate overall and by sex ----------
repeat_rate = df["Repeated_Same_Year"].mean()
rep_counts = df["Repeated_Same_Year"].value_counts().reindex([0,1]).fillna(0).astype(int)
fig = plt.figure()
rep_counts.index = ["No Repeat (same year)","Repeat (same year)"]
rep_counts.plot(kind="bar")
plt.title(f"Same-Year Repetition: Overall Rate = {repeat_rate:.1%}")
plt.xlabel("Outcome")
plt.ylabel("Count")
repeat_overall_path = save_show(fig, "repetition_overall.png")

if "Sex" in df.columns:
    rep_by_sex = df.groupby("Sex")["Repeated_Same_Year"].mean().sort_index()
    fig = plt.figure()
    rep_by_sex.plot(kind="bar")
    plt.title("Same-Year Repetition Rate by Sex")
    plt.xlabel("Sex")
    plt.ylabel("Rate")
    repetition_by_sex_path = save_show(fig, "repetition_by_sex.png")
else:
    repetition_by_sex_path = None

# ---------- 8) Metrics and curves from preds_logreg.csv (if available) ----------
metrics_summary = {}
roc_path = pr_path = calib_path = None

if os.path.exists(preds_path) and SKLEARN_OK:
    preds = pd.read_csv(preds_path)
    # Try to infer label and probability columns
    label_candidates = [c for c in preds.columns if preds[c].dropna().isin([0,1]).all()]
    proba_candidates = [c for c in preds.columns if np.issubdtype(preds[c].dropna().dtype, np.number) 
                        and preds[c].dropna().between(0,1).mean() > 0.95 and preds[c].nunique() > 10]
    # fallbacks
    y_col = None
    p_col = None
    for c in ["y_true","label","is_repeat","repeat_label","Repeated_Same_Year"]:
        if c in preds.columns and preds[c].dropna().isin([0,1]).all():
            y_col = c; break
    for c in ["y_pred","p_repeat","prob","proba","pred_prob","pred_proba"]:
        if c in preds.columns and np.issubdtype(preds[c].dropna().dtype, np.number):
            p_col = c; break
    if y_col is None and label_candidates:
        y_col = label_candidates[0]
    if p_col is None and proba_candidates:
        p_col = proba_candidates[0]
    if y_col is not None and p_col is not None:
        y_true = preds[y_col].astype(int).values
        y_score = preds[p_col].astype(float).values

        # ROC
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        fig = plt.figure()
        plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.3f}")
        plt.plot([0,1],[0,1], linestyle="--")
        plt.title("ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        roc_path = save_show(fig, "roc_curve.png")

        # PR
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        fig = plt.figure()
        plt.plot(recall, precision, label=f"AP = {ap:.3f}")
        base = (y_true==1).mean()
        plt.hlines(base, 0, 1, linestyles="--", label=f"Baseline = {base:.3f}")
        plt.title("Precision–Recall Curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend()
        pr_path = save_show(fig, "pr_curve.png")

        # Calibration curve
        prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=10, strategy="quantile")
        fig = plt.figure()
        plt.plot(prob_pred, prob_true, marker="o", label="Binned")
        plt.plot([0,1],[0,1], linestyle="--", label="Perfect")
        plt.title("Calibration (Reliability) Curve")
        plt.xlabel("Predicted probability")
        plt.ylabel("Observed frequency")
        plt.legend()
        calib_path = save_show(fig, "calibration_curve.png")

        metrics_summary = {
            "roc_auc": float(roc_auc),
            "average_precision": float(ap),
            "prevalence": float(base)
        }
    else:
        metrics_summary = {"warning": "Could not infer label/probability columns in preds_logreg.csv"}
else:
    if not os.path.exists(preds_path):
        metrics_summary = {"info": "preds_logreg.csv not found—skipping ROC/PR/Calibration visuals"}
    elif not SKLEARN_OK:
        metrics_summary = {"info": "scikit-learn not available—skipping ROC/PR/Calibration visuals"}

# ---------- Save a small README with figure paths ----------
index_info = {
    "figures": {
        "attempts_by_sex": sex_bar_path,
        "method_overall": method_overall_path,
        "method_by_sex": method_by_sex_path,
        "age_hist": age_hist_path,
        "attempts_by_month": attempts_by_month_path,
        "attempts_by_weekday": attempts_by_weekday_path,
        "repetition_overall": repeat_overall_path,
        "repetition_by_sex": repetition_by_sex_path,
        "roc_curve": roc_path,
        "pr_curve": pr_path,
        "calibration_curve": calib_path
    },
    "metrics_summary": metrics_summary
}
with open(os.path.join(outdir, "index.json"), "w") as f:
    json.dump(index_info, f, indent=2)

index_info
# This script generates various visualizations from the synthetic dataset and prediction results.
# The figures are saved in the specified output directory, and a summary of metrics is also provided