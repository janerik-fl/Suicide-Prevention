import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import calendar
from typing import Dict, Optional, List

# -------------------- Config dataclass --------------------

@dataclass
class UruguayConfig:
    seed: int = 123
    start_year: int = 2023
    num_years: int = 5
    records_per_year: int = 5000

    sex: Dict[str, float] = field(default_factory=lambda: {"Female": 0.716, "Male": 0.284})
    country_of_origin: Dict[str, float] = field(default_factory=lambda: {"Uruguay": 0.998, "Foreign": 0.002})
    institution: Dict[str, float] = field(default_factory=lambda: {"Private": 0.611, "Public": 0.389})

    methods_by_sex: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "Female": {
            "Self-poisoning by drugs/medicines": 0.565,
            "Self-cutting": 0.085,
            "Handgun discharge": 0.001,
            "Hanging/Suffocation": 0.0,
            "Other": 0.349,
        },
        "Male": {
            "Self-poisoning by drugs/medicines": 0.780,
            "Hanging/Suffocation": 0.184,
            "Handgun discharge": 0.010,
            "Self-cutting": 0.0,
            "Other": 0.026,
        },
    })

    month_weights: Dict[int, float] = field(default_factory=lambda: {
        1:1, 2:1, 3:1, 4:1, 5:1, 6:0.9, 7:0.9, 8:1, 9:1, 10:1.2, 11:1, 12:1
    })
    dow_weights: Dict[str, float] = field(default_factory=lambda: {
        "Monday":796, "Tuesday":689.5, "Wednesday":689.5, "Thursday":689.5,
        "Friday":583, "Saturday":658.5, "Sunday":734
    })

    # Age mixture
    youth_15_29: float = 0.473
    child_5_14: float = 0.075
    normal_mean: float = 32.0
    normal_sd: float = 16.5
    min_age: int = 5
    max_age: int = 93

    # Repeat timing
    yearly_repeat_rate: float = 0.0817
    lognormal_median_days: float = 54
    lognormal_q1_days: float = 16
    lognormal_q3_days: float = 127

    # Treatment/referral
    undergoing_rate: float = 0.69
    referral_if_treated: float = 0.9

    # Missingness (MCAR + simple MAR hook)
    missingness: Dict[str, float] = field(default_factory=lambda: {
        "Date_of_Birth": 0.015,
        "Method_Used": 0.008,
        "Undergoing_Mental_Health_Treatment": 0.025
    })

# -------------------- Utility helpers --------------------

def lognormal_params_from_median_iqr(median: float, q1: float, q3: float):
    # For X ~ LogNormal(mu, sigma): median = exp(mu), IQR ratio = exp(2*0.67449*sigma)
    mu = np.log(median)
    sigma = (np.log(q3) - np.log(q1)) / (2 * 0.67448975)
    return mu, sigma

WEEKDAY_TO_INT = {'Monday':0,'Tuesday':1,'Wednesday':2,'Thursday':3,'Friday':4,'Saturday':5,'Sunday':6}

# -------------------- AI feature (synthetic risk score) --------------------

def compute_model_score_repeat_risk(row) -> float:
    """
    Logistic model producing a synthetic risk-of-repeat score in [0,1].
    Coefficients are illustrative; adjust in config if you externalize.
    """
    # Features
    age_dec = row["Age_at_Attempt"] / 10.0
    sex_male = 1.0 if row["Sex"] == "Male" else 0.0
    prev = 1.0 if row["Previous_Suicide_Attempts"] else 0.0
    method = row["Method_Used"]
    hanging = 1.0 if method == "Hanging/Suffocation" else 0.0
    treated = 1.0 if row["Undergoing_Mental_Health_Treatment"] else 0.0

    # Coefs (synthetic, conservative)
    b0 = -2.0
    b_age = 0.05
    b_male = -0.15
    b_prev = 0.9
    b_hanging = 0.35
    b_treat = -0.1

    logit = b0 + b_age*age_dec + b_male*sex_male + b_prev*prev + b_hanging*hanging + b_treat*treated
    return float(1.0 / (1.0 + np.exp(-logit)))

# -------------------- Core generator --------------------

class UruguaySyntheticGenerator:
    def __init__(self, cfg: UruguayConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self._prep()

    def _prep(self):
        # Normalize DOW and month probabilities
        self.dow_keys = list(self.cfg.dow_weights.keys())
        self.dow_probs = np.array([self.cfg.dow_weights[d] for d in self.dow_keys], dtype=float)
        self.dow_probs /= self.dow_probs.sum()

        months = sorted(self.cfg.month_weights.keys())
        self.month_keys = months
        self.month_probs = np.array([self.cfg.month_weights[m] for m in months], dtype=float)
        self.month_probs /= self.month_probs.sum()

        # Methods
        self.method_names = list(next(iter(self.cfg.methods_by_sex.values())).keys())

        # Lognormal params for repeat timing
        self.mu_rep, self.sigma_rep = lognormal_params_from_median_iqr(
            self.cfg.lognormal_median_days,
            self.cfg.lognormal_q1_days,
            self.cfg.lognormal_q3_days
        )

    def _draw_age(self) -> int:
        u = self.rng.random()
        if u < self.cfg.youth_15_29:
            return int(self.rng.integers(15, 30))
        elif u < self.cfg.youth_15_29 + self.cfg.child_5_14:
            return int(self.rng.integers(5, 15))
        else:
            return int(np.clip(self.rng.normal(self.cfg.normal_mean, self.cfg.normal_sd), self.cfg.min_age, self.cfg.max_age))

    def _attempt_date(self, year: int) -> datetime:
        month = int(self.rng.choice(self.month_keys, p=self.month_probs))
        last_day = calendar.monthrange(year, month)[1]
        day = int(self.rng.integers(1, last_day + 1))
        base = datetime(year, month, day)
        target_dow = self.rng.choice(self.dow_keys, p=self.dow_probs)
        days_diff = (WEEKDAY_TO_INT[target_dow] - base.weekday() + 7) % 7
        dt = base + timedelta(days=int(days_diff))
        if dt.year != year:  # safety
            dt = datetime(year, month, int(self.rng.integers(1, last_day + 1)))
        return dt

    def _maybe_missing(self, col: str, value):
        p = self.cfg.missingness.get(col, 0.0)
        if p <= 0: 
            return value
        # Simple MAR: slightly more missingness in public provider for some fields
        if col in ("Undergoing_Mental_Health_Treatment", "Date_of_Birth"):
            boost = 0.01
            p = min(1.0, p + boost) if getattr(self, "_is_public", False) else p
        return None if (self.rng.random() < p) else value

    def generate(self) -> pd.DataFrame:
        rows: List[Dict] = []
        rec_id = 0

        for y in range(self.cfg.num_years):
            year = self.cfg.start_year + y
            for _ in range(self.cfg.records_per_year):
                rec_id += 1
                row = {}

                # Core draws
                country = self.rng.choice(list(self.cfg.country_of_origin.keys()),
                                          p=list(self.cfg.country_of_origin.values()))
                sex = self.rng.choice(list(self.cfg.sex.keys()), p=list(self.cfg.sex.values()))
                age = self._draw_age()

                # Methods by sex
                m_probs = np.array([self.cfg.methods_by_sex[sex][m] for m in self.method_names], dtype=float)
                method = self.rng.choice(self.method_names, p=m_probs)

                # Attempt date and registration
                attempt_date = self._attempt_date(year)
                reg_hours = int(self.rng.integers(1, 24))
                date_of_registration = attempt_date + timedelta(hours=reg_hours)

                # Institution & ED
                institution = self.rng.choice(list(self.cfg.institution.keys()), p=list(self.cfg.institution.values()))
                self._is_public = (institution == "Public")
                ed_code = f"ED_{int(self.rng.integers(1, 98)):03d}"

                # Treatment/referral
                in_treatment = bool(self.rng.random() < self.cfg.undergoing_rate)
                referred = (not in_treatment) or (self.rng.random() < self.cfg.referral_if_treated)

                # Previous & repeat
                prev_attempts = bool(self.rng.random() < 0.506)
                if self.rng.random() < self.cfg.yearly_repeat_rate:
                    # lognormal in days, clamp to 16–127 like IQR coverage
                    days = np.exp(self.rng.normal(self.mu_rep, self.sigma_rep))
                    days = float(np.clip(days, self.cfg.lognormal_q1_days, self.cfg.lognormal_q3_days))
                    second_date = attempt_date + timedelta(days=int(round(days)))
                    second_same_year = second_date if second_date.year == year else None
                else:
                    second_same_year = None

                # DOB (approx mid-year anchor)
                approx_mid = datetime(year, 7, 1)
                dob_year = approx_mid.year - int(age)
                dob_month = int(self.rng.integers(1, 13))
                dob_day = int(self.rng.integers(1, 29))
                dob = datetime(dob_year, dob_month, dob_day)

                # Assemble
                row["ID_Number"] = str(rec_id)
                row["Country_of_Origin"] = country
                row["Sex"] = sex
                row["Age_at_Attempt"] = int(age)
                row["Date_of_Birth"] = self._maybe_missing("Date_of_Birth", dob.strftime("%Y-%m-%d"))
                row["Method_Used"] = self._maybe_missing("Method_Used", method)
                row["Suicide_Attempt_Date"] = attempt_date.strftime("%Y-%m-%d")
                row["Previous_Suicide_Attempts"] = prev_attempts
                row["Second_Attempt_Date_Same_Year"] = (
                    second_same_year.strftime("%Y-%m-%d") if second_same_year else None
                )
                row["Undergoing_Mental_Health_Treatment"] = self._maybe_missing(
                    "Undergoing_Mental_Health_Treatment", in_treatment
                )
                row["Referred_to_Mental_Health_Care"] = referred
                row["Health_Care_Institution"] = institution
                row["ED_Where_Recorded"] = ed_code
                row["Date_of_Registration"] = date_of_registration.strftime("%Y-%m-%d %H:%M:%S")

                # AI-ready feature
                row["model_score_repeat_risk"] = compute_model_score_repeat_risk(row)

                rows.append(row)

        df = pd.DataFrame(rows)

        # Optimize dtypes
        for col in ["Country_of_Origin", "Sex", "Method_Used", "Health_Care_Institution", "ED_Where_Recorded"]:
            df[col] = df[col].astype("category")

        return df

# -------------------- Validation (quick checks) --------------------

def validate_against_targets(df: pd.DataFrame) -> Dict[str, float]:
    """Return simple deviation metrics from targets (add χ²/KS as needed)."""
    metrics = {}

    # Sex distribution
    sex_props = df["Sex"].value_counts(normalize=True)
    metrics["sex_female"] = float(sex_props.get("Female", 0))
    metrics["sex_male"] = float(sex_props.get("Male", 0))

    # Institution
    inst_props = df["Health_Care_Institution"].value_counts(normalize=True)
    metrics["inst_private"] = float(inst_props.get("Private", 0))
    metrics["inst_public"] = float(inst_props.get("Public", 0))

    # Repeat rate
    rep_rate = df["Second_Attempt_Date_Same_Year"].notna().mean()
    metrics["repeat_rate"] = float(rep_rate)

    # Treatment rate (excluding missing)
    treat = df["Undergoing_Mental_Health_Treatment"].dropna()
    metrics["treatment_rate"] = float(treat.mean()) if len(treat) else np.nan

    # Method distribution (overall)
    m = df["Method_Used"].value_counts(normalize=True, dropna=True).to_dict()
    for k, v in m.items():
        metrics[f"method_{k}"] = float(v)

    return metrics

# -------------------- Demo run --------------------

if __name__ == "__main__":
    cfg = UruguayConfig()
    gen = UruguaySyntheticGenerator(cfg)
    df = gen.generate()

    print("Sample rows:")
    print(df.head(5).to_markdown(index=False))

    print("\nQuick validation metrics:")
    print(validate_against_targets(df))

    # Save artifacts
    df.to_csv("synthetic_uruguay_attempts.csv", index=False)
    print("\nSaved: synthetic_uruguay_attempts.csv")
