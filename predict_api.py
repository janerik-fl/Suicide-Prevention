from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd, numpy as np, joblib, json, os

app=FastAPI(title="Synthetic Uruguay Repeat Risk API",version="1.0.0")
MODEL_PATH=os.environ.get("MODEL_PATH","./models/logreg_pipeline.joblib")
SPEC_PATH=os.environ.get("SPEC_PATH","./models/model_spec.json")
model=None; spec=None

class Record(BaseModel):
    ID_Number: Optional[str]=None
    Sex: Optional[str]=None
    Method_Used: Optional[str]=None
    Health_Care_Institution: Optional[str]=None
    Country_of_Origin: Optional[str]=None
    ED_Where_Recorded: Optional[str]=None
    Suicide_Attempt_Date: str
    Age_at_Attempt: Optional[float]=None
    Previous_Suicide_Attempts: Optional[float]=None
    Undergoing_Mental_Health_Treatment: Optional[float]=None

class PredictRequest(BaseModel):
    records: List[Record]
    top_k: Optional[int]=None
    min_proba: Optional[float]=None

def prep(df,feat_cat,feat_num):
    df=df.copy()
    df['Suicide_Attempt_Date']=pd.to_datetime(df['Suicide_Attempt_Date'])
    df['month']=df['Suicide_Attempt_Date'].dt.month
    df['dow']=df['Suicide_Attempt_Date'].dt.day_name()
    for c in ['Previous_Suicide_Attempts','Undergoing_Mental_Health_Treatment']:
        if c in df.columns: df[c]=df[c].astype(float)
    for col in feat_cat+feat_num:
        if col not in df.columns: df[col]=np.nan
    return df[feat_cat+feat_num]

@app.on_event("startup")
def load_artifacts():
    global model,spec
    try:
        model=joblib.load(MODEL_PATH)
        spec=json.load(open(SPEC_PATH))
    except Exception as e:
        raise RuntimeError(f"Failed to load artifacts: {e}")

@app.get("/health")
def health():
    return {"status":"ok" if (model and spec) else "error","model":os.path.basename(MODEL_PATH)}

@app.post("/predict")
def predict(req:PredictRequest):
    if model is None or spec is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    feat_cat,feat_num=spec["feat_cols_cat"],spec["feat_cols_num"]
    df_in=pd.DataFrame([r.dict() for r in req.records])
    X=prep(df_in,feat_cat,feat_num)

    if hasattr(model,"predict_proba"): proba=model.predict_proba(X)[:,1]
    else:
        from scipy.special import expit
        proba=expit(model.decision_function(X))

    out=df_in[['ID_Number']].copy() if 'ID_Number' in df_in.columns else pd.DataFrame({"ID_Number":np.arange(len(df_in))})
    out["pred_proba"]=proba

    if req.min_proba is not None:
        out=out[out["pred_proba"]>=float(req.min_proba)]
    if req.top_k is not None:
        out=out.sort_values("pred_proba",ascending=False).head(req.top_k)

    return {"n":int(len(out)),"predictions":out.to_dict(orient="records")}
