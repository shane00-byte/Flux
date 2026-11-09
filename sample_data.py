"""
sample_data.py
===============
Generates small SYNTHETIC datasets that match the column schema the app
expects (heart.csv, diabetes.csv, dataset.csv). These are randomly
generated and NOT clinically meaningful -- they exist only so the app
can be run and clicked through before you plug in the real datasets:

    Heart Disease   -> https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data
    Diabetes        -> https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
    Symptom checker -> https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset

Run directly to write the three CSVs into the current directory:

    python sample_data.py
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)


def make_heart_csv(n: int = 300) -> pd.DataFrame:
    df = pd.DataFrame({
        "age": RNG.integers(29, 78, n),
        "sex": RNG.choice(["Male", "Female"], n),
        "cp": RNG.choice(["typical angina", "atypical angina", "non-anginal", "asymptomatic"], n),
        "trestbps": RNG.integers(94, 200, n),
        "chol": RNG.integers(126, 400, n),
        "fbs": RNG.choice([True, False], n, p=[0.15, 0.85]),
        "restecg": RNG.choice(["normal", "st-t abnormality", "lv hypertrophy"], n),
        "thalch": RNG.integers(71, 202, n),
        "exang": RNG.choice([False, True], n, p=[0.7, 0.3]),
        "oldpeak": np.round(RNG.uniform(0, 6.2, n), 1),
        "slope": RNG.choice(["upsloping", "flat", "downsloping"], n),
        "ca": RNG.choice([0.0, 1.0, 2.0, 3.0], n),
        "thal": RNG.choice(["normal", "fixed defect", "reversable defect"], n),
    })
    risk_score = (
        (df["age"] > 55).astype(int) + (df["chol"] > 240).astype(int)
        + (df["trestbps"] > 140).astype(int) + df["exang"].astype(int)
    )
    df["num"] = (risk_score + RNG.integers(0, 2, n) >= 2).astype(int)
    return df


def make_diabetes_csv(n: int = 300) -> pd.DataFrame:
    df = pd.DataFrame({
        "Pregnancies": RNG.integers(0, 14, n),
        "Glucose": RNG.integers(70, 200, n),
        "BloodPressure": RNG.integers(50, 120, n),
        "SkinThickness": RNG.integers(0, 60, n),
        "Insulin": RNG.integers(0, 400, n),
        "BMI": np.round(RNG.uniform(16, 55, n), 1),
        "DiabetesPedigreeFunction": np.round(RNG.uniform(0.05, 2.4, n), 2),
        "Age": RNG.integers(21, 81, n),
    })
    risk_score = ((df["Glucose"] > 140).astype(int) + (df["BMI"] > 30).astype(int)
                  + (df["Age"] > 45).astype(int))
    df["Outcome"] = (risk_score + RNG.integers(0, 2, n) >= 2).astype(int)
    return df


def make_symptom_csv() -> pd.DataFrame:
    diseases = {
        "Common Cold": ["cough", "runny_nose", "sore_throat", "mild_fever", "sneezing"],
        "Migraine": ["headache", "nausea", "sensitivity_to_light", "blurred_vision"],
        "Typhoid": ["high_fever", "abdominal_pain", "weakness", "loss_of_appetite", "headache"],
        "Dengue": ["high_fever", "joint_pain", "skin_rash", "nausea", "fatigue"],
        "Pneumonia": ["cough", "high_fever", "chest_pain", "breathlessness", "fatigue"],
        "Gastroenteritis": ["vomiting", "diarrhoea", "abdominal_pain", "dehydration"],
        "Hypertension": ["headache", "dizziness", "chest_pain", "fatigue"],
        "Heart attack": ["chest_pain", "breathlessness", "sweating", "fatigue", "vomiting"],
        "Allergy": ["sneezing", "runny_nose", "skin_rash", "watery_eyes"],
        "Tuberculosis": ["cough", "weight_loss", "high_fever", "fatigue", "chest_pain"],
    }
    max_len = max(len(v) for v in diseases.values())
    rows = []
    for disease, symptoms in diseases.items():
        padded = symptoms + [np.nan] * (max_len - len(symptoms))
        rows.append([disease] + padded)
    cols = ["Disease"] + [f"Symptom_{i+1}" for i in range(max_len)]
    return pd.DataFrame(rows, columns=cols)


if __name__ == "__main__":
    make_heart_csv().to_csv("heart.csv", index=False)
    make_diabetes_csv().to_csv("diabetes.csv", index=False)
    make_symptom_csv().to_csv("dataset.csv", index=False)
    print("Wrote heart.csv, diabetes.csv, dataset.csv (synthetic sample data) to the current directory.")
