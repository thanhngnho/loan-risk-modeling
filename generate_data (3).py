# generate_data.py
# only run this if you DON'T have the real Kaggle dataset
# this just creates fake loan data so you can test the code
# when you get the real data from Kaggle, skip this file entirely
# and just rename your downloaded file to loan_data.csv

import numpy as np
import pandas as pd

np.random.seed(42)
N = 270000  # 270k rows to match the real dataset size

# loan grade A to G (A = safest borrower, G = riskiest)
grades = np.random.choice(
    ["A", "B", "C", "D", "E", "F", "G"], N,
    p=[0.20, 0.25, 0.22, 0.15, 0.10, 0.05, 0.03]
)

# FICO score — better grade usually means higher FICO
grade_to_fico = {"A": 750, "B": 720, "C": 690, "D": 670, "E": 650, "F": 630, "G": 610}
fico_scores = np.array([
    np.clip(np.random.normal(grade_to_fico[g], 25), 300, 850) for g in grades
]).astype(int)

# how much they're borrowing
loan_amount = np.random.choice(
    [5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000], N,
    p=[0.10, 0.20, 0.20, 0.20, 0.15, 0.08, 0.05, 0.02]
)

# annual income
annual_income = np.random.lognormal(mean=10.8, sigma=0.6, size=N).clip(15000, 500000)

# DTI = debt to income ratio, higher = more debt relative to income = riskier
dti = np.random.normal(18, 8, N).clip(0, 50).round(2)

# interest rate — riskier grade = higher rate
grade_to_rate = {"A": 7, "B": 11, "C": 14, "D": 17, "E": 20, "F": 23, "G": 26}
int_rate = np.array([
    np.clip(np.random.normal(grade_to_rate[g], 2), 5, 31) for g in grades
]).round(2)

# how long they've been employed
emp_length = np.random.choice(range(11), N)

# do they rent, own, or have a mortgage
home_ownership = np.random.choice(
    ["RENT", "MORTGAGE", "OWN", "OTHER"], N,
    p=[0.45, 0.40, 0.12, 0.03]
)

# what the loan is for
purpose = np.random.choice(
    ["debt_consolidation", "credit_card", "home_improvement", "other", "major_purchase", "medical"], N,
    p=[0.45, 0.20, 0.12, 0.10, 0.08, 0.05]
)

# public records (bankruptcies etc)
pub_rec = np.random.choice([0, 1, 2, 3], N, p=[0.85, 0.10, 0.03, 0.02])

# how much of their credit limit they're using
revol_util = np.random.normal(50, 20, N).clip(0, 100).round(1)


# simulate who actually defaulted
# based on real patterns — grade, fico, dti all affect default probability
grade_base_default = {"A": 0.05, "B": 0.10, "C": 0.16, "D": 0.22, "E": 0.28, "F": 0.34, "G": 0.40}

default_prob = np.array([grade_base_default[g] for g in grades])
default_prob += 0.003 * (dti - 18)              # higher DTI = more defaults
default_prob -= 0.0003 * (fico_scores - 680)    # higher FICO = fewer defaults
default_prob += 0.0002 * (int_rate - 14)        # higher rate = more defaults
default_prob -= 0.001 * (emp_length - 5)        # more job stability = fewer defaults
default_prob += 0.001 * pub_rec                 # past records = more defaults
default_prob += np.random.normal(0, 0.03, N)   # some randomness
default_prob = np.clip(default_prob, 0.01, 0.95)

loan_status = (np.random.rand(N) < default_prob).astype(int)  # 1 = defaulted, 0 = paid off


df = pd.DataFrame({
    "loan_amnt":      loan_amount,
    "int_rate":       int_rate,
    "grade":          grades,
    "fico_range_low": fico_scores,
    "annual_inc":     annual_income.round(0),
    "dti":            dti,
    "emp_length":     emp_length,
    "home_ownership": home_ownership,
    "purpose":        purpose,
    "pub_rec":        pub_rec,
    "revol_util":     revol_util,
    "loan_status":    loan_status,
})

df.to_csv("loan_data.csv", index=False)
print(f"done — saved {len(df):,} rows to loan_data.csv")
print(f"default rate: {df['loan_status'].mean():.1%}")
