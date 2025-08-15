import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_synthetic_data(num_years=5, records_per_year=5000):
    """
    Generates synthetic data for Uruguay's National Real-Time Suicide Attempt Surveillance System.

    Args:
        num_years (int): The number of years for which to generate data.
        records_per_year (int): The approximate number of records per year.

    Returns:
        pandas.DataFrame: A DataFrame containing the synthetic data.
    """

    data = []
    start_year = 2023 # Start year as specified in the report (first year was 2023)

    # Demographic distributions based on the report
    gender_distribution = {'Female': 0.716, 'Male': 0.284}

    # Method distribution based on the report (approximated for gender where specific breakdown isn't 100%)
    # Overall: Self-poisoning (71.6%), Other (9.8%)
    # Men: Self-poisoning (78%), Hanging/Suffocation (18.4%), Firearm (1%) -> other approx 2.6%
    # Women: Self-poisoning (56.5%), Self-cutting (8.5%), Firearm (0.1%) -> other approx 34.9% (to make it add up)

    methods = {
        'Self-poisoning by drugs/medicines': {'Female': 0.565, 'Male': 0.78},
        'Hanging/Suffocation': {'Female': 0.0, 'Male': 0.184}, # Only male specific given
        'Self-cutting': {'Female': 0.085, 'Male': 0.0}, # Only female specific given
        'Handgun discharge': {'Female': 0.001, 'Male': 0.01},
        'Other': {'Female': 0.349, 'Male': 0.026} # Adjusted to sum to 1.0 for each gender
    }

    # Age distribution based on report: over half (54.8%) under 30, 47.3% between 15-29
    # Average age 32 (SD 16.5), range 5-93. This suggests a skewed distribution.
    # We'll use a mix of normal and uniform to simulate this.

    # Healthcare provider distribution - CORRECTED to sum to 1.0
    healthcare_provider_distribution = {'Private': 0.611, 'Public': 0.389} # Corrected from 0.388 to 0.389

    # Day of week preference (Mondays and Sundays slightly higher, Fridays lowest)
    day_of_week_weights = {
        'Monday': 796, 'Tuesday': (583+796)/2, 'Wednesday': (583+796)/2,
        'Thursday': (583+796)/2, 'Friday': 583, 'Saturday': (583+734)/2, 'Sunday': 734
    }
    total_weights = sum(day_of_week_weights.values())
    day_of_week_probabilities = {day: weight / total_weights for day, weight in day_of_week_weights.items()}
    days_of_week = list(day_of_week_probabilities.keys())
    day_probabilities = list(day_of_week_probabilities.values())

    # Month preference (October highest)
    # Average 393.6 attempts/month. October is highest.
    # A simple way to simulate: add a slight bias to October.
    month_weights = {month: 1.0 for month in range(1, 13)}
    month_weights[10] = 1.2 # October (Spring)
    # weeks 25-29 (winter, school holidays) observed subtle decrease - can model this with lower weights
    # Assuming rough midpoint of year for weeks 25-29, around June-July.
    month_weights[6] = 0.9
    month_weights[7] = 0.9

    total_month_weights = sum(month_weights.values())
    month_probabilities = {month: weight / total_month_weights for month, weight in month_weights.items()}


    for year_offset in range(num_years):
        current_year = start_year + year_offset
        start_date = datetime(current_year, 1, 1)
        end_date = datetime(current_year, 12, 31)

        for i in range(records_per_year):
            record = {}

            # ID Number
            record['ID_Number'] = str(i + 1 + year_offset * records_per_year) # Simple sequential for synthetic data

            # Country of Origin
            record['Country_of_Origin'] = np.random.choice(['Uruguay', 'Foreign'], p=[0.998, 0.002])

            # Sex
            sex = np.random.choice(list(gender_distribution.keys()), p=list(gender_distribution.values()))
            record['Sex'] = sex

            # Date of Birth (to determine age)
            # Simulate age distribution: 54.8% under 30, 47.3% between 15-29. Avg 32, SD 16.5
            age = int(np.random.normal(32, 16.5))
            # Adjust for specific age group percentages and general range
            if random.random() < 0.473: # 47.3% between 15-29
                age = random.randint(15, 29)
            elif random.random() < 0.548: # remaining % under 30 (0-14, excluding 15-29 for now, so 0-14)
                 age = random.randint(5, 14)

            # Ensure age is within 5-93
            age = max(5, min(93, age))
            record['Age_at_Attempt'] = age

            # Calculate DOB based on age and current_year (attempt date approx mid-year)
            approx_attempt_date = datetime(current_year, 7, 1)
            dob_year = approx_attempt_date.year - age
            dob_month = random.randint(1, 12)
            dob_day = random.randint(1, 28) # Simplification for valid day
            record['Date_of_Birth'] = datetime(dob_year, dob_month, dob_day).strftime('%Y-%m-%d')


            # Methods Used in Suicide Attempt
            method_probs = [methods[m][sex] for m in methods]
            method_names = list(methods.keys())
            record['Method_Used'] = np.random.choice(method_names, p=method_probs)


            # Suicide Attempt Date (Higher on Mon/Sun, lower on Fri, higher in Oct)
            # Pick a random date within the year, then adjust based on day/month preferences
            chosen_month = np.random.choice(list(month_probabilities.keys()), p=list(month_probabilities.values()))
            # To ensure a valid day for the chosen month
            max_day = (datetime(current_year, chosen_month % 12 + 1, 1) - timedelta(days=1)).day if chosen_month != 12 else 31 # For December
            chosen_day_in_month = random.randint(1, max_day)

            attempt_date_base = datetime(current_year, chosen_month, chosen_day_in_month)

            # Adjust day of week
            # Find a date close to attempt_date_base that matches the preferred day of week
            target_weekday_name = np.random.choice(days_of_week, p=day_probabilities)
            # Convert weekday name to integer (Monday=0, Sunday=6)
            weekday_map = {'Monday':0, 'Tuesday':1, 'Wednesday':2, 'Thursday':3, 'Friday':4, 'Saturday':5, 'Sunday':6}
            target_weekday_int = weekday_map[target_weekday_name]

            days_diff = (target_weekday_int - attempt_date_base.weekday() + 7) % 7
            attempt_date = attempt_date_base + timedelta(days=days_diff)

            # Ensure attempt_date is still within the current year
            if attempt_date.year != current_year:
                attempt_date = datetime(current_year, chosen_month, chosen_day_in_month) # Reset if it drifts to next year
                # Simple fallback: if month is October, just pick a random day in October
                if chosen_month == 10:
                    attempt_date = datetime(current_year, 10, random.randint(1, 31))

            record['Suicide_Attempt_Date'] = attempt_date.strftime('%Y-%m-%d')


            # Previous Suicide Attempts
            # 50.6% have previous attempts
            record['Previous_Suicide_Attempts'] = random.random() < 0.506

            # For 8.17% of individuals, repeat within the same year
            if random.random() < 0.0817:
                # Simulate a second attempt within 54 days median (16-127 IQR)
                days_to_repeat = int(np.random.normal(54, 30)) # Using normal for simplicity, adjust for IQR if needed
                days_to_repeat = max(16, min(127, days_to_repeat)) # Cap within IQR range approximately
                second_attempt_date = attempt_date + timedelta(days=days_to_repeat)
                record['Second_Attempt_Date_Same_Year'] = second_attempt_date.strftime('%Y-%m-%d') if second_attempt_date.year == current_year else None
            else:
                record['Second_Attempt_Date_Same_Year'] = None


            # Mental Health Treatment (69% undergoing treatment)
            record['Undergoing_Mental_Health_Treatment'] = random.random() < 0.69

            # Referral to Mental Health Care (assume high probability if not already in treatment)
            record['Referred_to_Mental_Health_Care'] = True if not record['Undergoing_Mental_Health_Treatment'] else (random.random() < 0.9) # High likelihood of referral


            # Health Care Institution
            record['Health_Care_Institution'] = np.random.choice(
                list(healthcare_provider_distribution.keys()), p=list(healthcare_provider_distribution.values()))

            # ED Where Recorded (simulated as generic ED ID)
            record['ED_Where_Recorded'] = f'ED_{random.randint(1, 97):03d}' # 97 EDs nationwide

            # Date of Registration (within 24 hours of attempt date)
            record['Date_of_Registration'] = (attempt_date + timedelta(hours=random.randint(1, 23))).strftime('%Y-%m-%d %H:%M:%S')

            data.append(record)

    df = pd.DataFrame(data)
    return df

# Generate the data
synthetic_df = generate_synthetic_data(num_years=5, records_per_year=5000)

# Display the first few rows and information
print("Generated Synthetic Data (first 5 rows):")
print(synthetic_df.head().to_markdown(index=False))
print("\nDataFrame Info:")
print(synthetic_df.info())
print("\nValue Counts for key categorical fields:")
print("\nSex:\n", synthetic_df['Sex'].value_counts(normalize=True))
print("\nMethod_Used:\n", synthetic_df['Method_Used'].value_counts(normalize=True))
print("\nPrevious_Suicide_Attempts:\n", synthetic_df['Previous_Suicide_Attempts'].value_counts(normalize=True))
print("\nUndergoing_Mental_Health_Treatment:\n", synthetic_df['Undergoing_Mental_Health_Treatment'].value_counts(normalize=True))
print("\nHealth_Care_Institution:\n", synthetic_df['Health_Care_Institution'].value_counts(normalize=True))

print("\n\n--- Synthetic Data (CSV Format) ---")
# Output the entire DataFrame as CSV for easy copy-pasting
print(synthetic_df.to_csv(index=False))