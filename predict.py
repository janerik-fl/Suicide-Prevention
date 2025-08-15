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
    ap.add_argument("--output",default=None)
    ap.add_argument("--proba-col",default="pred_proba")
    ap.add_argument("--flag-topq",type=float,default=None)
    a=ap.parse_args()

    model=joblib.load(a.model); spec=json.load(open(a.spec))
    df=pd.read_csv(a.input_csv,low_memory=False); X=prep(df,spec["feat_cols_cat"],spec["feat_cols_num"])

    if hasattr(model,"predict_proba"): proba=model.predict_proba(X)[:,1]
    else:
        from scipy.special import expit
        proba=expit(model.decision_function(X))

    out=pd.DataFrame({a.id_col:(df[a.id_col] if a.id_col in df.columns else range(len(df))), a.proba_col:proba})
    if a.flag_topq is not None:
        thr=float(np.quantile(proba,a.flag_topq))
        out["flag_high_risk"]=(proba>=thr).astype(int)
        out["threshold_used"]=thr

    outpath=a.output or (os.path.splitext(a.input_csv)[0]+"_preds.csv")
    out.to_csv(outpath,index=False)
    print("Wrote", outpath)
