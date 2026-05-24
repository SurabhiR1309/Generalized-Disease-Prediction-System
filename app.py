from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# =========================================
# Load Models
# =========================================
heart_model = pickle.load(open("models/heart_model.pkl", "rb"))
liver_model = pickle.load(open("models/liver_model.pkl", "rb"))
diabetes_model = pickle.load(open("models/diabetes_model.pkl", "rb"))

# =========================================
# Load Datasets
# =========================================
# Dataset 1: hospital treatment dataset
hospital_df = pd.read_csv("datasets/hospital_patient_treatment_dataset.csv")

# Dataset 2: Practo/Doctor dataset
practo_df = pd.read_csv("datasets/docter.csv")

# Normalize column names
hospital_df.columns = hospital_df.columns.str.strip().str.lower()
practo_df.columns = practo_df.columns.str.strip().str.lower()

# =========================================
# Disease → Doctor Department Mapping
# =========================================
disease_keywords = {
    "Heart Disease": ["Cardiology", "Cardiologist", "Cardiac"],
    "Liver Disease": ["Gastroenterology", "Hepatology", "Liver", "Hepatologist"],
    "Diabetes": ["Endocrinology", "Diabetologist", "Endocrinologist"]
}

# =========================================
# Doctor Recommender Function
# =========================================
def recommend_doctors_from_sources(disease_list, top_n=5):
    all_recommendations = pd.DataFrame()

    for disease in disease_list:
        keywords = disease_keywords.get(disease, [])
        if not keywords:
            continue

        # --- Match in hospital dataset ---
        matches_hospital = hospital_df[hospital_df['department'].str.contains('|'.join(keywords), case=False, na=False)]
        if not matches_hospital.empty:
            matches_hospital = matches_hospital.copy()
            matches_hospital["Source"] = "Hospital Dataset"
            matches_hospital["Matched_for"] = disease
            all_recommendations = pd.concat([all_recommendations, matches_hospital], ignore_index=True)

        # --- Match in practo dataset ---
        matches_practo = practo_df[practo_df['speciality'].str.contains('|'.join(keywords), case=False, na=False)]
        if not matches_practo.empty:
            matches_practo = matches_practo.copy()
            matches_practo["Source"] = "Practo Dataset"
            matches_practo["Matched_for"] = disease
            all_recommendations = pd.concat([all_recommendations, matches_practo], ignore_index=True)

    # Return top N entries combined
    if all_recommendations.empty:
        return pd.DataFrame()
    return all_recommendations.head(top_n)

# =========================================
# Prediction Helper Function
# =========================================
def predict_disease(model, values, disease_name):
    arr = np.array([float(x) for x in values.split(",")]).reshape(1, -1)
    pred = model.predict(arr)[0]
    return (disease_name, "Likely" if pred == 1 else "Unlikely")

# =========================================
# Flask Routes
# =========================================
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        hv = request.form.get("heart_values", "").strip()
        lv = request.form.get("liver_values", "").strip()
        dv = request.form.get("diabetes_values", "").strip()

        results = []
        if hv:
            results.append(predict_disease(heart_model, hv, "Heart Disease"))
        if lv:
            results.append(predict_disease(liver_model, lv, "Liver Disease"))
        if dv:
            results.append(predict_disease(diabetes_model, dv, "Diabetes"))

        if not results:
            return render_template("result.html", message="⚠️ Please enter at least one disease report to proceed.")

        detected = [d for d, r in results if r == "Likely"]

        if not detected:
            return render_template("result.html", results=results, message="✅ No major disease detected.", doctors=None)

        # Recommend doctors
        recommendations = recommend_doctors_from_sources(detected, top_n=10)
        if recommendations.empty:
            message = "❌ No doctors found for the detected diseases."
            doctors = None
        else:
            message = "🩺 Doctor Recommendations"
            # Select relevant columns safely
            display_cols = ["doctor name", "department", "treatment type", "speciality",
                            "name", "degree", "city", "location", "consult fee",
                            "years of experience", "source", "matched_for"]
            doctors = recommendations[[c for c in display_cols if c in recommendations.columns]].to_dict(orient="records")

        return render_template("result.html", results=results, message=message, doctors=doctors)

    except Exception as e:
        return render_template("result.html", message=f"❌ Error occurred: {e}")


if __name__ == "__main__":
    app.run(debug=True)
