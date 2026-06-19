# app.py
# Flask web app — type in borrower info, get a default risk score
# run with: python app.py
# then go to http://localhost:5000 in your browser

from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# load the model we trained in analysis.py
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# these have to match the encoding from analysis.py
GRADE_MAP   = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6}
HOME_MAP    = {"MORTGAGE": 0, "OTHER": 1, "OWN": 2, "RENT": 3}
PURPOSE_MAP = {
    "credit_card": 0, "debt_consolidation": 1,
    "home_improvement": 2, "major_purchase": 3,
    "medical": 4, "other": 5
}


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        # put the inputs in the same order as the training features
        features = np.array([[
            float(data["loan_amnt"]),
            float(data["int_rate"]),
            GRADE_MAP.get(data["grade"], 2),
            float(data["fico_range_low"]),
            float(data["annual_inc"]),
            float(data["dti"]),
            float(data["emp_length"]),
            HOME_MAP.get(data["home_ownership"], 3),
            PURPOSE_MAP.get(data["purpose"], 5),
            float(data["pub_rec"]),
            float(data["revol_util"]),
        ]])

        # probability of defaulting (0 to 1, shown as %)
        default_prob = model.predict_proba(features)[0][1]
        risk_score   = round(default_prob * 100, 1)

        # label based on score
        if risk_score < 15:
            risk_label = "Low Risk"
            color = "#2ECC71"
        elif risk_score < 30:
            risk_label = "Medium Risk"
            color = "#F39C12"
        else:
            risk_label = "High Risk"
            color = "#E74C3C"

        return jsonify({
            "risk_score": risk_score,
            "risk_label": risk_label,
            "color": color
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)
