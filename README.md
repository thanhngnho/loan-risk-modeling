# Loan Default Risk Model 
 
So I was looking at around in the bank while waiting for my parents and I was like — predicting whether a borrower will default on their loan. It's literally what banks do every time someone applies for a loan, so I figured why not build it myself. Got the data from Kaggle (real Lending Club loan records) and built a Random Forest model that predicts default risk based on things like FICO score, DTI ratio, and loan grade.
 
The cool part is I also deployed it as a Flask web app where you can type in borrower details and get a live risk score back.
 
## Getting Started
 
These instructions will get the project running on your local machine.
 
### Prerequisites
 
What you need before running anything:
 
* Python 3.12
* pip (comes with Python)
Install everything at once:
 
```
pip install -r requirements.txt
```
 
Or manually:
 
```
pip install pandas numpy matplotlib seaborn scikit-learn flask imbalanced-learn
```
 
### Installing
 
Step by step to get it running:
 
Clone the repo:
 
```
git clone https://github.com/thanhngnho/Loan-Default-Risk
cd Loan-Default-Risk
```
 
**Option A — using the real Kaggle dataset (recommended):**
 
Download the dataset from:
```
https://www.kaggle.com/datasets/wordsforthewise/lending-club
```
 
Rename the downloaded file to `loan_data.csv` and drop it in the project folder. Then skip straight to Step 2.
 
**Option B — using fake data to test the code first:**
 
```
python generate_data.py
```
 
This creates a fake `loan_data.csv` with 270,000 simulated loan records so you can run the code without the real data.
 
**Step 2** — train the model:
 
```
python analysis.py
```
 
This trains the Random Forest, handles class imbalance with SMOTE, and saves the model + 2 chart images.
 
**Step 3** — run the web app:
 
```
python app.py
```
 
Then open `http://localhost:5000` in your browser. You'll see a form where you can enter borrower details and get a predicted default risk score.
 
You should see something like this in the terminal after Step 2:
 
```
=== results ===
ROC-AUC: 0.716
 
what mattered most:
  FICO score     — higher score = less likely to default
  interest rate  — higher rate = more likely to default
  loan grade     — G borrowers default way more than A borrowers
  DTI ratio      — more debt relative to income = more defaults
```
 
## Running the Tests
 
### Model performance
 
The model is evaluated on 25% of data it never saw during training:
 
```
python analysis.py
```
 
Prints a full classification report (precision, recall, F1) for both "paid off" and "defaulted" classes.
 
### Class imbalance check
 
Way more loans are paid off than defaulted — without fixing this, the model would just always predict "paid off". SMOTE handles this by generating synthetic default examples during training. The script prints the class distribution before and after SMOTE so you can see it working.
 
## Deployment
 
The Flask app runs locally out of the box. To deploy it publicly you can use:
 
* [Render](https://render.com) — free tier, easy to set up
* [Railway](https://railway.app) — also free, very beginner friendly
* [Heroku](https://heroku.com) — classic option
To switch from fake data to real data: just replace `loan_data.csv` with the Kaggle file and re-run `analysis.py`. No code changes needed.
 
## Built With
 
* [pandas](https://pandas.pydata.org/) - data handling
* [scikit-learn](https://scikit-learn.org/) - Random Forest, model evaluation
* [imbalanced-learn](https://imbalanced-learn.org/) - SMOTE for class imbalance
* [matplotlib](https://matplotlib.org/) / [seaborn](https://seaborn.pydata.org/) - charts
* [Flask](https://flask.palletsprojects.com/) - web app framework
## Authors
 
* **Thanh Ngo** - [thanhngnho](https://github.com/thanhngnho)
## Acknowledgments
 
* Lending Club dataset from Kaggle — [wordsforthewise/lending-club](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
* Kahneman & Tversky (1979) for getting me interested in risk modeling in the first place
* Every finance class that made me want to actually build something instead of just studying it
