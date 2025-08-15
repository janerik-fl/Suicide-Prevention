import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import calendar
from typing import Optional

def generate_synthetic_data(num_years: int = 5, records_per_year: int = 5000, seed: Optional[int] = 42) -> pd.DataFrame:
    """
    Synthetic data for Uruguay's National Real-Time Suicide Attempt Surveillance System.
    Key assumptions mirror the report; randomness is reproducible via `seed`.
    """
    rng = np.random.default_rng(seed)
    data = []
    start_year = 2023

    # Demographics
    gender_distribution = {'Female': 0.716, 'Male': 0.284}

    methods = {
        'Self-poisoning by drugs/medicines': {'Female': 0.565, 'Male': 0.780},
        'Hanging/Suffocation':              {'Female': 0.000, 'Male': 0.184},
        'Self-cutting':                      {'Female': 0.085, 'Male': 0.000},
        'Handgun discharge':                 {'Female': 0.001, 'Male': 0.010},
        'Other':                             {'Female': 0.349, 'Male': 0.026},
    }  # sums to 1.0 per gender

    healthcare_provider_distribution = {'Private': 0.611, 'Public': 0.389}

    # Day-of-week weights (Mon/Sun higher, Fri lowest)
    dow_w = {'Monday':796, 'Tuesday':(583+796)/2, 'Wednesday':(583+796)/2,
             'Thursday':(583+796)/2, 'Friday':583, 'Saturday':(583+734)/2, 'Sunday':734}
    dow_keys = list(dow_w.keys())
    dow_probs = np.array([dow_w[d] for d in dow_keys], dtype=float)
    dow_probs /= dow_probs.sum()
    weekday_to_int = {'Monday':0,'Tuesday':1,'Wednesday':2,'Thursday':3,'Friday':4,'Saturday':5,'Sunday':6}

    # Month weights (Oct ↑, Jun/Jul ↓)
    month_weights = np.ones(12, dtype=float)
    month_weights[9]  = 1.2  # October (index 9)
    month_weights[5]  = 0.9  # June
    month_weights[6]  = 0.9  # July
    month_probs = month_weights / month_weights.sum()

    rec_id = 0
    for y in range(num_years):
        current_year = start_year + y
        for _ in range(records_per_year):
            rec_id += 1
            rec = {}

            rec['ID_Number'] = str(rec_id)
            rec['Country_of_Origin'] = rng.choice(['Uruguay', 'Foreign'], p=[0.998, 0.002])

            sex = rng.choice(['Female','Male'], p=[gender_distribution['Female'], gender_distribution['Male']])
            rec['Sex'] = sex

            # Age: target ~47.3% in 15–29, ~7.5% in 5–14 (remaining from N(32,16.5^2) clipped to [5,93])
            u = rng.random()
            if u < 0.473:
                age = rng.integers(15, 30)  # 15–29 inclusive
            elif u < 0.548:  # 0.473 + 0.075
                age = rng.integers(5, 15)   # 5–14 inclusive
            else:
                age = int(np.clip(rng.normal(32, 16.5), 5, 93))
            rec['Age_at_Attempt'] = int(age)

            # DOB (approx mid-year anchor)
            approx_attempt_date = datetime(current_year, 7, 1)
            dob_year = approx_attempt_date.year - int(age)
            dob_month = int(rng.integers(1, 13))
            dob_day = int(rng.integers(1, 29))  # safe day
            rec['Date_of_Birth'] = datetime(dob_year, dob_month, dob_day).strftime('%Y-%m-%d')

            # Method by sex
            method_names = list(methods.keys())
            method_probs = np.array([methods[m][sex] for m in method_names], dtype=float)
            rec['Method_Used'] = rng.choice(method_names, p=method_probs)

            # Attempt date: month bias + day-of-week bias
            chosen_month = int(rng.choice(np.arange(1, 13), p=month_probs))
            last_day = calendar.monthrange(current_year, chosen_month)[1]
            chosen_day = int(rng.integers(1, last_day + 1))
            base_date = datetime(current_year, chosen_month, chosen_day)

            target_dow = rng.choice(dow_keys, p=dow_probs)
            target_dow_int = weekday_to_int[target_dow]
            days_diff = (target_dow_int - base_date.weekday() + 7) % 7
            attempt_date = base_date + timedelta(days=int(days_diff))
            if attempt_date.year != current_year:
                # Fallback: random day in chosen month within current year
                attempt_date = datetime(current_year, chosen_month, int(rng.integers(1, last_day + 1)))

            rec['Suicide_Attempt_Date'] = attempt_date.strftime('%Y-%m-%d')

            # Previous attempts (50.6%)
            rec['Previous_Suicide_Attempts'] = bool(rng.random() < 0.506)

            # Repeat within same year (8.17%), median ~54 days (clamped 16–127)
            if rng.random() < 0.0817:
                days_to_repeat = int(np.clip(rng.normal(54, 30), 16, 127))
                second_date = attempt_date + timedelta(days=days_to_repeat)
                rec['Second_Attempt_Date_Same_Year'] = (
                    second_date.strftime('%Y-%m-%d') if second_date.year == current_year else None
                )
            else:
                rec['Second_Attempt_Date_Same_Year'] = None

            # Treatment and referral
            under_treatment = bool(rng.random() < 0.69)
            rec['Undergoing_Mental_Health_Treatment'] = under_treatment
            rec['Referred_to_Mental_Health_Care'] = (not under_treatment) or (rng.random() < 0.9)

            # Institution, ED, registration timestamp (within 24h)
            rec['Health_Care_Institution'] = rng.choice(['Private','Public'], p=[0.611, 0.389])
            rec['ED_Where_Recorded'] = f'ED_{int(rng.integers(1, 98)):03d}'
            reg_hours = int(rng.integers(1, 24))
            rec['Date_of_Registration'] = (attempt_date + timedelta(hours=reg_hours)).strftime('%Y-%m-%d %H:%M:%S')

            data.append(rec)

    df = pd.DataFrame(data)

    # Optimize dtypes
    cat_cols = ['Sex','Method_Used','Health_Care_Institution','Country_of_Origin','ED_Where_Recorded']
    for c in cat_cols:
        df[c] = df[c].astype('category')

    return df

if __name__ == "__main__":
    df = generate_synthetic_data(num_years=5, records_per_year=5000, seed=123)
    print("Generated Synthetic Data (first 5 rows):")
    print(df.head().to_markdown(index=False))
    print("\nQuick checks:")
    print(df['Sex'].value_counts(normalize=True))
    print(df['Method_Used'].value_counts(normalize=True))
    print(df['Health_Care_Institution'].value_counts(normalize=True))
    # Instead of printing all CSV, write to file:
    out_path = "synthetic_uruguay_attempts.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved CSV to: {out_path}  (rows={len(df)})")
