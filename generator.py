
# generator.py

import pandas as pd, numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import calendar

WEEKDAY_TO_INT={'Monday':0,'Tuesday':1,'Wednesday':2,'Thursday':3,'Friday':4,'Saturday':5,'Sunday':6}

@dataclass
class UruguayConfig:
    seed:int=123; start_year:int=2023; num_years:int=2; records_per_year:int=1000
    sex=dict(Female=0.716, Male=0.284)
    country_of_origin=dict(Uruguay=0.998, Foreign=0.002)
    institution=dict(Private=0.611, Public=0.389)
    methods_by_sex: dict = field(default_factory=lambda:{
        "Female":{"Self-poisoning by drugs/medicines":0.565,"Self-cutting":0.085,"Handgun discharge":0.001,"Hanging/Suffocation":0.0,"Other":0.349},
        "Male":{"Self-poisoning by drugs/medicines":0.780,"Hanging/Suffocation":0.184,"Handgun discharge":0.010,"Self-cutting":0.0,"Other":0.026}
    })
    month_weights=dict([(m,1.0) for m in range(1,13)]); month_weights={**month_weights,6:0.9,7:0.9,10:1.2}
    dow_weights={"Monday":796,"Tuesday":689.5,"Wednesday":689.5,"Thursday":689.5,"Friday":583,"Saturday":658.5,"Sunday":734}
    youth_15_29:float=0.473; child_5_14:float=0.075; normal_mean:float=32.0; normal_sd:float=16.5; min_age:int=5; max_age:int=93
    yearly_repeat_rate:float=0.0817; lognormal_median_days:float=54; lognormal_q1_days:float=16; lognormal_q3_days:float=127
    undergoing_rate:float=0.69; referral_if_treated:float=0.9
    missingness=dict(Date_of_Birth=0.015, Method_Used=0.008, Undergoing_Mental_Health_Treatment=0.025)

def lognormal_params_from_median_iqr(median,q1,q3):
    mu=np.log(median); sigma=(np.log(q3)-np.log(q1))/(2*0.67448975); return mu,sigma

def compute_model_score_repeat_risk(row)->float:
    age_dec=row["Age_at_Attempt"]/10.0; sex_male=1.0 if row["Sex"]=="Male" else 0.0
    prev=1.0 if row["Previous_Suicide_Attempts"] else 0.0
    hanging=1.0 if row["Method_Used"]=="Hanging/Suffocation" else 0.0
    treated=1.0 if row["Undergoing_Mental_Health_Treatment"] else 0.0
    b0,b_age,b_male,b_prev,b_hang,b_treat=-2.0,0.05,-0.15,0.9,0.35,-0.1
    logit=b0+b_age*age_dec+b_male*sex_male+b_prev*prev+b_hang*hanging+b_treat*treated
    return float(1/(1+np.exp(-logit)))

class UruguaySyntheticGenerator:
    def __init__(self,cfg:UruguayConfig):
        self.cfg=cfg; self.rng=np.random.default_rng(cfg.seed); self._prep()
    def _prep(self):
        self.dow_keys=list(self.cfg.dow_weights.keys()); self.dow_probs=np.array([self.cfg.dow_weights[d] for d in self.dow_keys],dtype=float); self.dow_probs/=self.dow_probs.sum()
        self.month_keys=sorted(self.cfg.month_weights.keys()); self.month_probs=np.array([self.cfg.month_weights[m] for m in self.month_keys],dtype=float); self.month_probs/=self.month_probs.sum()
        self.method_names=list(self.cfg.methods_by_sex["Female"].keys())
        self.mu_rep,self.sigma_rep=lognormal_params_from_median_iqr(self.cfg.lognormal_median_days,self.cfg.lognormal_q1_days,self.cfg.lognormal_q3_days)
    def _draw_age(self)->int:
        u=self.rng.random()
        if u<self.cfg.youth_15_29: return int(self.rng.integers(15,30))
        elif u<self.cfg.youth_15_29+self.cfg.child_5_14: return int(self.rng.integers(5,15))
        else: return int(np.clip(self.rng.normal(self.cfg.normal_mean,self.cfg.normal_sd),self.cfg.min_age,self.cfg.max_age))
    def _attempt_date(self,year:int)->datetime:
        month=int(self.rng.choice(self.month_keys,p=self.month_probs)); last_day=calendar.monthrange(year,month)[1]
        day=int(self.rng.integers(1,last_day+1)); base=datetime(year,month,day)
        target=self.rng.choice(self.dow_keys,p=self.dow_probs); dd=(WEEKDAY_TO_INT[target]-base.weekday()+7)%7
        dt=base+timedelta(days=int(dd))
        if dt.year!=year: dt=datetime(year,month,int(self.rng.integers(1,last_day+1)))
        return dt
    def _maybe_missing(self,col,val):
        p=self.cfg.missingness.get(col,0.0); 
        return (None if self.rng.random()<p else val)
    def generate(self)->pd.DataFrame:
        rows=[]; rec_id=0
        for y in range(self.cfg.num_years):
            year=self.cfg.start_year+y
            for _ in range(self.cfg.records_per_year):
                rec_id+=1; row={}
                country=self.rng.choice(list(self.cfg.country_of_origin.keys()),p=list(self.cfg.country_of_origin.values()))
                sex=self.rng.choice(list(self.cfg.sex.keys()),p=list(self.cfg.sex.values()))
                age=self._draw_age()
                m_probs=np.array([self.cfg.methods_by_sex[sex][m] for m in self.method_names]); method=self.rng.choice(self.method_names,p=m_probs)
                attempt=self._attempt_date(year); reg_hours=int(self.rng.integers(1,24))
                institution=self.rng.choice(list(self.cfg.institution.keys()),p=list(self.cfg.institution.values()))
                ed_code=f"ED_{int(self.rng.integers(1,98)):03d}"
                in_treat=bool(self.rng.random()<self.cfg.undergoing_rate)
                referred=(not in_treat) or (self.rng.random()<self.cfg.referral_if_treated)
                prev=bool(self.rng.random()<0.506)
                if self.rng.random()<self.cfg.yearly_repeat_rate:
                    mu=np.log(self.cfg.lognormal_median_days); sigma=(np.log(self.cfg.lognormal_q3_days)-np.log(self.cfg.lognormal_q1_days))/(2*0.67448975)
                    days=float(np.exp(self.rng.normal(mu,sigma))); days=float(np.clip(days,self.cfg.lognormal_q1_days,self.cfg.lognormal_q3_days))
                    second=attempt+timedelta(days=int(round(days))); second_same=second if second.year==year else None
                else: second_same=None
                approx_mid=datetime(year,7,1); dob_year=approx_mid.year-int(age); dob=datetime(dob_year,int(self.rng.integers(1,13)),int(self.rng.integers(1,29)))
                row.update({
                    "ID_Number":str(rec_id),"Country_of_Origin":country,"Sex":sex,"Age_at_Attempt":int(age),
                    "Date_of_Birth":self._maybe_missing("Date_of_Birth",dob.strftime("%Y-%m-%d")),
                    "Method_Used":self._maybe_missing("Method_Used",method),
                    "Suicide_Attempt_Date":attempt.strftime("%Y-%m-%d"),
                    "Previous_Suicide_Attempts":prev,
                    "Second_Attempt_Date_Same_Year":(second_same.strftime("%Y-%m-%d") if second_same else None),
                    "Undergoing_Mental_Health_Treatment":self._maybe_missing("Undergoing_Mental_Health_Treatment",in_treat),
                    "Referred_to_Mental_Health_Care":referred,
                    "Health_Care_Institution":institution,"ED_Where_Recorded":ed_code,
                    "Date_of_Registration":(attempt+timedelta(hours=reg_hours)).strftime("%Y-%m-%d %H:%M:%S")
                })
                row["model_score_repeat_risk"]=compute_model_score_repeat_risk(row)
                rows.append(row)
        df=pd.DataFrame(rows)
        for c in ["Country_of_Origin","Sex","Method_Used","Health_Care_Institution","ED_Where_Recorded"]:
            df[c]=df[c].astype("category")
        return df

if __name__=="__main__":
    import os
    os.makedirs("data", exist_ok=True)
    df=UruguaySyntheticGenerator(UruguayConfig()).generate()
    df.to_csv("data/synthetic_uruguay_attempts.csv",index=False)
    print("Wrote data/synthetic_uruguay_attempts.csv", len(df))
