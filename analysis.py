# analysis.py
# run this after generate_data.py (or after putting real Kaggle data in loan_data.csv)
# trains a random forest to predict who will default on their loan
 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay
)
import pickle
import warnings
warnings.filterwarnings("ignore")
 
# try to import SMOTE for handling class imbalance
# if you don't have it: pip install imbalanced-learn
try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False
    print("no SMOTE found, using class_weight instead (that's fine)")
 
 
# load data
df = pd.read_csv("loan_data.csv")
 
print("=== quick look at the data ===")
print(f"total loans: {len(df):,}")
print(f"default rate: {df['loan_status'].mean():.1%}")
print(f"paid off rate: {1 - df['loan_status'].mean():.1%}")
# way more people pay off than default — that's the class imbalance problem
 
 
# 6 charts to understand the data before modeling
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Loan Default — Exploratory Data Analysis", fontsize=14, fontweight="bold")
 
 
# does grade predict default? (it really should)
grade_order = ["A", "B", "C", "D", "E", "F", "G"]
grade_default = df.groupby("grade")["loan_status"].mean().reindex(grade_order)
axes[0, 0].bar(grade_order, grade_default.values, color="#E05B5B", edgecolor="white")
axes[0, 0].set_xlabel("loan grade")
axes[0, 0].set_ylabel("default rate")
axes[0, 0].set_title("default rate by loan grade")
for i, v in enumerate(grade_default.values):
    axes[0, 0].text(i, v + 0.005, f"{v:.0%}", ha="center", fontsize=9)
 
 
# FICO score — do lower scores default more?
axes[0, 1].hist(df[df["loan_status"] == 0]["fico_range_low"], bins=40,
                alpha=0.6, color="#5B8DEF", label="paid off", density=True)
axes[0, 1].hist(df[df["loan_status"] == 1]["fico_range_low"], bins=40,
                alpha=0.6, color="#E05B5B", label="defaulted", density=True)
axes[0, 1].set_xlabel("FICO score")
axes[0, 1].set_ylabel("density")
axes[0, 1].set_title("FICO score by loan outcome")
axes[0, 1].legend()
 
 
# DTI — does more debt = more defaults?
dti_bins = pd.cut(df["dti"], bins=[0, 10, 20, 30, 40, 50])
dti_default = df.groupby(dti_bins, observed=True)["loan_status"].mean()
axes[0, 2].bar(range(len(dti_default)), dti_default.values, color="#F4A261", edgecolor="white")
axes[0, 2].set_xticks(range(len(dti_default)))
axes[0, 2].set_xticklabels(["0-10", "10-20", "20-30", "30-40", "40-50"])
axes[0, 2].set_xlabel("DTI ratio")
axes[0, 2].set_ylabel("default rate")
axes[0, 2].set_title("default rate by DTI")
 
 
# interest rate — higher rate borrowers default more?
axes[1, 0].hist(df[df["loan_status"] == 0]["int_rate"], bins=40,
                alpha=0.6, color="#5B8DEF", label="paid off", density=True)
axes[1, 0].hist(df[df["loan_status"] == 1]["int_rate"], bins=40,
                alpha=0.6, color="#E05B5B", label="defaulted", density=True)
axes[1, 0].set_xlabel("interest rate (%)")
axes[1, 0].set_title("interest rate by loan outcome")
axes[1, 0].legend()
 
 
# home ownership
home_default = df.groupby("home_ownership")["loan_status"].mean().sort_values(ascending=False)
axes[1, 1].bar(home_default.index, home_default.values, color="#A78BFA", edgecolor="white")
axes[1, 1].set_xlabel("home ownership")
axes[1, 1].set_ylabel("default rate")
axes[1, 1].set_title("default rate by home ownership")
 
 
# loan amount
axes[1, 2].hist(df[df["loan_status"] == 0]["loan_amnt"], bins=30,
                alpha=0.6, color="#5B8DEF", label="paid off", density=True)
axes[1, 2].hist(df[df["loan_status"] == 1]["loan_amnt"], bins=30,
                alpha=0.6, color="#E05B5B", label="defaulted", density=True)
axes[1, 2].set_xlabel("loan amount ($)")
axes[1, 2].set_title("loan amount by outcome")
axes[1, 2].legend()
 
plt.tight_layout()
plt.savefig("eda_plots.png", bbox_inches="tight")
plt.show()
print("saved eda_plots.png")
 
 
# prep features
df_model = df.copy()
 
# encode text columns as numbers since the model can't read text
df_model["grade_enc"]   = df_model["grade"].map({"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6})
df_model["home_enc"]    = LabelEncoder().fit_transform(df_model["home_ownership"])
df_model["purpose_enc"] = LabelEncoder().fit_transform(df_model["purpose"])
 
features = [
    "loan_amnt", "int_rate", "grade_enc", "fico_range_low",
    "annual_inc", "dti", "emp_length", "home_enc",
    "purpose_enc", "pub_rec", "revol_util"
]
 
X = df_model[features].values
y = df_model["loan_status"].values   # 1 = defaulted, 0 = paid off
 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
print(f"\ntraining on {len(X_train):,} rows, testing on {len(X_test):,} rows")
 
 
# SMOTE — creates fake default examples so the model sees more of them
# without this, the model might just always predict "paid off" since that's most common
if HAS_SMOTE:
    print("applying SMOTE to fix class imbalance...")
    X_train, y_train = SMOTE(random_state=42, sampling_strategy=0.5).fit_resample(X_train, y_train)
    print(f"after SMOTE: {len(X_train):,} training rows")
 
 
# train the random forest
# it builds 100 decision trees and averages them — more reliable than one tree
print("training random forest (this takes a minute)...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight=None if HAS_SMOTE else "balanced",
    random_state=42,
    n_jobs=-1   # uses all CPU cores to go faster
)
model.fit(X_train, y_train)
 
 
# see how well it did
y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
auc     = roc_auc_score(y_test, y_proba)
 
print("\n=== results ===")
print(f"ROC-AUC: {auc:.3f}")
print(classification_report(y_test, y_pred, target_names=["paid off", "defaulted"]))
 
 
# plot results
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Random Forest Results", fontsize=14, fontweight="bold")
 
 
# which features does the model think matter most?
importances = model.feature_importances_
feature_labels = ["Loan amount", "Int rate", "Grade", "FICO",
                  "Income", "DTI", "Emp length", "Home",
                  "Purpose", "Pub records", "Revol util"]
sorted_idx = np.argsort(importances)
axes[0].barh(
    [feature_labels[i] for i in sorted_idx],
    [importances[i] for i in sorted_idx],
    color="#5B8DEF", edgecolor="white"
)
axes[0].set_xlabel("importance score")
axes[0].set_title("what predicts default?")
 
 
# confusion matrix — where did it get right vs wrong
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred),
                       display_labels=["paid off", "defaulted"]).plot(
    ax=axes[1], colorbar=False, cmap="Blues"
)
axes[1].set_title("confusion matrix")
 
 
# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
axes[2].plot(fpr, tpr, color="#5B8DEF", lw=2, label=f"our model (AUC = {auc:.3f})")
axes[2].plot([0, 1], [0, 1], "k--", lw=1, label="random guessing (0.5)")
axes[2].set_xlabel("false positive rate")
axes[2].set_ylabel("true positive rate")
axes[2].set_title("ROC curve")
axes[2].legend()
 
plt.tight_layout()
plt.savefig("model_results.png", bbox_inches="tight")
plt.show()
print("saved model_results.png")
 
 
# save the model so the Flask app can use it
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
print("saved model.pkl")
 
 
print(f"""
=== summary ===
total loans  : {len(df):,}
default rate : {df['loan_status'].mean():.1%}
ROC-AUC      : {auc:.3f}
 
what mattered most:
  FICO score     — higher score = less likely to default
  interest rate  — higher rate = more likely to default
  loan grade     — G borrowers default way more than A borrowers
  DTI ratio      — more debt relative to income = more defaults
""")