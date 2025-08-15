import argparse
import os
import json
import pandas as pd
import numpy as np
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, average_precision_score

def load_df(p: str) -> pd.DataFrame:
    """Loads a CSV file and converts date columns to datetime objects."""
    df = pd.read_csv(p, low_memory=False)
    df['Suicide_Attempt_Date'] = pd.to_datetime(df['Suicide_Attempt_Date'])
    df['Second_Attempt_Date_Same_Year'] = pd.to_datetime(df['Second_Attempt_Date_Same_Year'], errors='coerce')
    return df

def engineer(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, list, list]:
    """
    Performs feature engineering, including time-based features and
    data type conversions for modeling.
    """
    df = df.copy()
    df['year'] = df['Suicide_Attempt_Date'].dt.year
    df['month'] = df['Suicide_Attempt_Date'].dt.month
    df['dow'] = df['Suicide_Attempt_Date'].dt.day_name()
    
    # Convert features to float to allow for NaN values, which the imputer will handle.
    for c in ['Previous_Suicide_Attempts', 'Undergoing_Mental_Health_Treatment']:
        df[c] = df[c].astype(float)
    
    # Create the binary target variable (y) for second attempts.
    df['y'] = df['Second_Attempt_Date_Same_Year'].notna().astype(int)
    
    feat_cat = ['Sex', 'Method_Used', 'Health_Care_Institution', 'Country_of_Origin', 'ED_Where_Recorded', 'dow']
    feat_num = ['Age_at_Attempt', 'month', 'Previous_Suicide_Attempts', 'Undergoing_Mental_Health_Treatment']
    
    X = df[feat_cat + feat_num]
    y = df['y'].values
    
    return X, y, df, feat_cat, feat_num

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="./data/synthetic_uruguay_attempts.csv")
    ap.add_argument("--outdir", default="./models")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    df = load_df(a.input)
    X, y, df, feat_cat, feat_num = engineer(df)

    # Split data based on time to simulate a real-world scenario.
    last_year = int(df['year'].max())
    tr = df['year'] < last_year
    te = df['year'] == last_year

    # Preprocessing Pipeline: Imputation, One-Hot Encoding, and Scaling.
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    preprocessing = ColumnTransformer([
        ('cat', cat_pipe, feat_cat),
        ('num', num_pipe, feat_num)
    ])

    # --- Logistic Regression Model ---
    lr_model = LogisticRegression(max_iter=2000, class_weight='balanced')
    lr_pipeline = Pipeline([('preprocessing', preprocessing), ('classifier', lr_model)]).fit(X.loc[tr], y[tr])
    
    lr_predictions = lr_pipeline.predict_proba(X.loc[te])[:, 1]
    
    lr_metrics = dict(
        roc_auc=float(roc_auc_score(y[te], lr_predictions)),
        pr_auc=float(average_precision_score(y[te], lr_predictions))
    )
    joblib.dump(lr_pipeline, os.path.join(a.outdir, "logreg_pipeline.joblib"))

    # --- HGB Classifier with Isotonic Calibration ---
    hgb_model = HistGradientBoostingClassifier(learning_rate=0.05, max_bins=255, random_state=a.seed)
    
    # The pipeline is nested inside CalibratedClassifierCV, which handles fitting.
    calibrated_hgb = CalibratedClassifierCV(
        Pipeline([('preprocessing', preprocessing), ('classifier', hgb_model)]),
        method='isotonic',
        cv=3
    ).fit(X.loc[tr], y[tr])
    
    hgb_predictions = calibrated_hgb.predict_proba(X.loc[te])[:, 1]
    
    hgb_metrics = dict(
        roc_auc=float(roc_auc_score(y[te], hgb_predictions)),
        pr_auc=float(average_precision_score(y[te], hgb_predictions))
    )
    joblib.dump(calibrated_hgb, os.path.join(a.outdir, "hgb_calibrated.joblib"))

    # Save model metadata and evaluation metrics
    spec = {
        "feat_cols_cat": feat_cat,
        "feat_cols_num": feat_num,
        "label": "y_repeat_same_year",
        "train_years": sorted(map(int, set(df.loc[tr, 'year']))),
        "test_year": int(last_year)
    }
    json.dump(spec, open(os.path.join(a.outdir, "model_spec.json"), "w"), indent=2)
    
    metrics = {"logreg": lr_metrics, "hgb_calibrated": hgb_metrics}
    json.dump(metrics, open(os.path.join(a.outdir, "metrics.json"), "w"), indent=2)

    print("Saved models to", a.outdir, "| Metrics:", metrics)