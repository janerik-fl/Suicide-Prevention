import argparse, os, json, pandas as pd, numpy as np, joblib

def prep(df,feat_cat,feat_num):
    df=df.copy(); df['Suicide_Attempt_Date']=pd.to_datetime(df['Suicide_Attempt_Date'])
    df['month']=df['Suicide_Attempt_Date'].dt.month; df['dow']=df['Suicide_Attempt_Date'].dt.day_name()
    for c in ['Previous_Suicide_Attempts','Undergoing_Mental_Health_Treatment']:
        if c in df.columns: df[c]=df[c].astype(float)
    for col in feat_cat+feat_num:
        if col not in df.columns: df[col]=np.nan
    return df[feat_cat+feat_num]

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("input_csv")
    ap.add_argument("--model",default="./models/logreg_pipeline.joblib")
    ap.add_argument("--spec",default="./models/model_spec.json")
    ap.add_argument("--id-col",default="ID_Number")
    ap.add_argument("--k",type=int,default=100)
    ap.add_argument("--min-proba",type=float,default=None)
    ap.add_argument("--output",default=None)
    a=ap.parse_args()

    model=joblib.load(a.model); spec=json.load(open(a.spec))
    df=pd.read_csv(a.input_csv,low_memory=False); X=prep(df,spec["feat_cols_cat"],spec["feat_cols_num"])

    if hasattr(model,"predict_proba"): proba=model.predict_proba(X)[:,1]
    else:
        from scipy.special import expit
        proba=expit(model.decision_function(X))

    out=df[[a.id_col]].copy() if a.id_col in df.columns else pd.DataFrame({a.id_col:np.arange(len(df))})
    out["pred_proba"]=proba
    if a.min_proba is not None: out=out[out["pred_proba"]>=float(a.min_proba)]
    out=out.sort_values("pred_proba",ascending=False).head(a.k)

    outpath=a.output or (os.path.splitext(a.input_csv)[0]+"_topk.csv")
    out.to_csv(outpath,index=False)
    print("Wrote top-{} to: {}".format(a.k, outpath))
    print(out.head(10).to_string(index=False))
