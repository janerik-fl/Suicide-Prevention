import argparse, os, json, numpy as np, pandas as pd, joblib
from sklearn.metrics import precision_recall_curve, roc_auc_score, average_precision_score

def prep_holdout(df,feat_cat,feat_num):
    df=df.copy()
    df['Suicide_Attempt_Date']=pd.to_datetime(df['Suicide_Attempt_Date'])
    df['Second_Attempt_Date_Same_Year']=pd.to_datetime(df['Second_Attempt_Date_Same_Year'],errors='coerce')
    years=df['Suicide_Attempt_Date'].dt.year
    last=years.max()
    df=df[years==last].copy()
    df['month']=df['Suicide_Attempt_Date'].dt.month
    df['dow']=df['Suicide_Attempt_Date'].dt.day_name()
    y=df['Second_Attempt_Date_Same_Year'].notna().astype(int).values
    for c in ['Previous_Suicide_Attempts','Undergoing_Mental_Health_Treatment']:
        df[c]=df[c].astype(float)
    X=df[feat_cat+feat_num]
    return X,y,int(last)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",default="./data/synthetic_uruguay_attempts.csv")
    ap.add_argument("--model",default="./models/logreg_pipeline.joblib")
    ap.add_argument("--spec",default="./models/model_spec.json")
    ap.add_argument("--out",default="./models/thresholds.json")
    ap.add_argument("--target-precision",type=float,default=0.6)
    ap.add_argument("--target-recall",type=float,default=0.5)
    a=ap.parse_args()

    spec=json.load(open(a.spec))
    df=pd.read_csv(a.data,low_memory=False)
    X,y,last=prep_holdout(df,spec["feat_cols_cat"],spec["feat_cols_num"])

    model=joblib.load(a.model)
    if hasattr(model,"predict_proba"): proba=model.predict_proba(X)[:,1]
    else:
        from scipy.special import expit
        proba=expit(model.decision_function(X))

    roc=float(roc_auc_score(y,proba))
    pr=float(average_precision_score(y,proba))
    prec,rec,thr=precision_recall_curve(y,proba)

    th_prec=next((t for p,t in zip(prec,np.append(thr,1.0)) if p>=a.target_precision), float(np.quantile(proba,0.95)))
    th_reca=next((t for r,t in zip(rec,np.append(thr,1.0)) if r>=a.target_recall), float(np.quantile(proba,0.50)))

    out={"model":os.path.basename(a.model),"holdout_year":last,"metrics":{"roc_auc":roc,"pr_auc":pr},
         "thresholds":{"for_precision":{"target":a.target_precision,"threshold":float(th_prec)},
                       "for_recall":{"target":a.target_recall,"threshold":float(th_reca)}}}
    os.makedirs(os.path.dirname(a.out),exist_ok=True)
    json.dump(out,open(a.out,"w"),indent=2)
    print(json.dumps(out,indent=2))
