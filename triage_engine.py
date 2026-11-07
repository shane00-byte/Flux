"""
triage_engine.py
=================
Core ML pipeline for the Flux AI Triage Hub.

Kept separate from app.py (the Streamlit UI) so the data loading, model
training, prediction and severity-scoring logic can be imported and
exercised on their own -- e.g. in a test script, a notebook, or a batch
job -- without needing a running Streamlit session.

Clinical note
-------------
The guardrail thresholds below (e.g. "BP >= 180 is a hypertensive
crisis") are carried over unchanged from the original project design.
They are illustrative defaults for a decision-support demo, not
clinically validated cutoffs. Any real deployment must have these
values reviewed and signed off by a licensed clinician before use.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing import Optional

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder


# ---------------------------------------------------------------------------
# Severity vocabulary shared by every department + the triage dashboard
# ---------------------------------------------------------------------------

SEVERITY = {
    "IMMEDIATE": {"label": "Immediate", "icon": "\U0001F6A8", "color": "#B3261E", "rank": 0},
    "URGENT":    {"label": "Urgent",    "icon": "\u26A0\uFE0F", "color": "#B7791F", "rank": 1},
    "ROUTINE":   {"label": "Routine",   "icon": "\U0001FA7A", "color": "#2F6F4E", "rank": 2},
    "INFO":      {"label": "Inconclusive", "icon": "\u2139\uFE0F", "color": "#4A5568", "rank": 3},
}


@dataclass
class TriageResult:
    label: str
    probability: float
    severity: str                 # key into SEVERITY
    show_shap: bool
    guardrail_triggered: bool
    guardrail_reason: Optional[str] = None
    top_factors: Optional[list] = None   # [(feature, direction, shap_value), ...]


@dataclass
class TrainedModel:
    model: RandomForestClassifier
    feature_cols: list
    categorical_cols: list
    encoder: Optional[OrdinalEncoder]
    accuracy: float
    classes: np.ndarray
    n_train_rows: int


# ---------------------------------------------------------------------------
# Cardiology
# ---------------------------------------------------------------------------

CARDIOLOGY_FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs",
                        "restecg", "thalch", "exang", "oldpeak", "slope",
                        "ca", "thal"]
CARDIOLOGY_CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
CARDIOLOGY_LABELS = {
    "age": "Age", "sex": "Sex", "cp": "Chest pain type", "trestbps": "Resting BP",
    "chol": "Cholesterol", "fbs": "Fasting blood sugar > 120", "restecg": "Resting ECG",
    "thalch": "Max heart rate", "exang": "Exercise-induced angina", "oldpeak": "ST depression",
    "slope": "ST segment slope", "ca": "Colored vessels (fluoroscopy)", "thal": "Thalassemia",
}


def load_heart_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("heart.csv is completely empty.")
    if "target" in df.columns and "num" not in df.columns:
        df = df.rename(columns={"target": "num"})
    missing = set(CARDIOLOGY_FEATURES + ["num"]) - set(df.columns)
    if missing:
        raise ValueError(f"heart.csv is missing expected column(s): {sorted(missing)}")
    return df


def train_cardiology_model(df: pd.DataFrame) -> TrainedModel:
    X_raw = df[CARDIOLOGY_FEATURES].copy()
    y = (pd.to_numeric(df["num"], errors="coerce").fillna(0) > 0).astype(int)

    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X = X_raw.copy()
    X[CARDIOLOGY_CATEGORICAL] = encoder.fit_transform(
        X_raw[CARDIOLOGY_CATEGORICAL].astype(str)
    )
    numeric_cols = [c for c in X.columns if c not in CARDIOLOGY_CATEGORICAL]
    for col in numeric_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())

    model, acc = _fit_with_holdout(X, y)
    return TrainedModel(model=model, feature_cols=list(X.columns),
                         categorical_cols=CARDIOLOGY_CATEGORICAL, encoder=encoder,
                         accuracy=acc, classes=model.classes_, n_train_rows=len(X))


def encode_cardiology_input(raw: dict, trained: TrainedModel) -> pd.DataFrame:
    df = pd.DataFrame([raw], columns=trained.feature_cols)
    df[trained.categorical_cols] = trained.encoder.transform(
        df[trained.categorical_cols].astype(str)
    )
    for col in df.columns:
        if col not in trained.categorical_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def evaluate_cardiology(trestbps: float, chol: float, model: RandomForestClassifier,
                         input_data: pd.DataFrame) -> TriageResult:
    if trestbps >= 180 or trestbps <= 60:
        return TriageResult(
            label="Critical vital alert (blood pressure)", probability=100.0,
            severity="IMMEDIATE", show_shap=False, guardrail_triggered=True,
            guardrail_reason=(f"Resting BP of {trestbps} mmHg indicates an immediate "
                               "hypertensive/hypotensive crisis. Clinical override applied."),
        )
    if chol >= 350:
        return TriageResult(
            label="Critical vital alert (cholesterol)", probability=100.0,
            severity="IMMEDIATE", show_shap=False, guardrail_triggered=True,
            guardrail_reason=(f"Cholesterol of {chol} mg/dl is severely elevated. "
                               "Immediate cardiovascular review required."),
        )
    proba = model.predict_proba(input_data)[0]
    pred_idx = int(np.argmax(proba))
    pred = model.classes_[pred_idx]
    probability = proba[pred_idx] * 100  # Calculate directly from the winning class

    if pred == 1:
        return TriageResult(label="Elevated risk", probability=probability,
                         severity="URGENT", show_shap=True, guardrail_triggered=False)
    return TriageResult(label="Routine / low risk", probability=probability, 
                     severity="ROUTINE", show_shap=True, guardrail_triggered=False)


# ---------------------------------------------------------------------------
# Endocrinology
# ---------------------------------------------------------------------------

DIABETES_FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                      "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
DIABETES_LABELS = {
    "Pregnancies": "Pregnancies", "Glucose": "Glucose", "BloodPressure": "Blood pressure",
    "SkinThickness": "Skin thickness", "Insulin": "Insulin", "BMI": "BMI",
    "DiabetesPedigreeFunction": "Diabetes pedigree function", "Age": "Age",
}


def load_diabetes_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = set(DIABETES_FEATURES + ["Outcome"]) - set(df.columns)
    if missing:
        raise ValueError(f"diabetes.csv is missing expected column(s): {sorted(missing)}")
    return df


def train_diabetes_model(df: pd.DataFrame) -> TrainedModel:
    X = df[DIABETES_FEATURES].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.fillna(X.median())
    y = pd.to_numeric(df["Outcome"], errors="coerce").fillna(0).astype(int)

    model, acc = _fit_with_holdout(X, y)
    return TrainedModel(model=model, feature_cols=list(X.columns), categorical_cols=[],
                         encoder=None, accuracy=acc, classes=model.classes_, n_train_rows=len(X))


def evaluate_diabetes(bp: float, glucose: float, bmi: float, model: RandomForestClassifier,
                       input_data: pd.DataFrame) -> TriageResult:
    if bp >= 180 or bp <= 50:
        return TriageResult(
            label="Critical vital alert (blood pressure)", probability=100.0,
            severity="IMMEDIATE", show_shap=False, guardrail_triggered=True,
            guardrail_reason=(f"{bp} mmHg is outside safe clinical parameters. Immediate "
                               "intervention for hypertensive/hypotensive crisis required."),
        )
    if glucose <= 50 or glucose >= 200:
        return TriageResult(
            label="Critical vital alert (glucose)", probability=100.0,
            severity="IMMEDIATE", show_shap=False, guardrail_triggered=True,
            guardrail_reason=(f"{glucose} mg/dL is at a critical diagnostic threshold. "
                               "Immediate intervention required."),
        )
    if bmi <= 14.0 or bmi >= 45.0:
        return TriageResult(
            label="Critical vital alert (BMI)", probability=100.0,
            severity="IMMEDIATE", show_shap=False, guardrail_triggered=True,
            guardrail_reason=(f"A BMI of {bmi:.1f} indicates severe physiological distress "
                               "(malnutrition or morbid obesity). Clinical override applied."),
        )
    proba = model.predict_proba(input_data)[0]
    pred_idx = int(np.argmax(proba))
    pred = model.classes_[pred_idx]
    probability = proba[pred_idx] * 100  # Calculate directly from the winning class

    if pred == 1:
        return TriageResult(label="Elevated risk", probability=probability,
                         severity="URGENT", show_shap=True, guardrail_triggered=False)
    return TriageResult(label="Routine / low risk", probability=probability, 
                     severity="ROUTINE", show_shap=True, guardrail_triggered=False)


# ---------------------------------------------------------------------------
# General medicine (symptom checker)
# ---------------------------------------------------------------------------

SEVERE_DISEASES = [
    "AIDS", "Malaria", "Dengue", "Typhoid", "Tuberculosis", "Pneumonia",
    "Heart attack", "Paralysis (brain hemorrhage)", "Hepatitis B",
    "Hepatitis C", "Hepatitis D", "Hepatitis E",
]
# Sex-specific biological guardrails 
FEMALE_ONLY_CONDITIONS = ["pregnancy", "ovarian", "cervical", "endometriosis", "pcos", "pelvic inflammatory"]
MALE_ONLY_CONDITIONS = ["prostate", "testicular", "benign prostatic"]


def load_symptom_data(path: str):
    df = pd.read_csv(path)
    if "Disease" not in df.columns:
        raise ValueError("dataset.csv must have a 'Disease' column.")
    symptoms = set()
    for col in df.columns[1:]:
        symptoms.update(df[col].dropna().apply(lambda x: str(x).strip()).unique())
    symptoms = sorted(s for s in symptoms if s)

    X = pd.DataFrame(0, index=df.index, columns=symptoms)
    for col in df.columns[1:]:
        for i, val in df[col].dropna().items():
            X.at[i, str(val).strip()] = 1
    y = df["Disease"]
    return X, y, symptoms


def train_symptom_model(X: pd.DataFrame, y: pd.Series) -> TrainedModel:
    model, acc = _fit_with_holdout(X, y)
    return TrainedModel(model=model, feature_cols=list(X.columns), categorical_cols=[],
                         encoder=None, accuracy=acc, classes=model.classes_, n_train_rows=len(X))


def evaluate_symptoms(selected_symptoms: list, model: RandomForestClassifier,
                       input_data: pd.DataFrame, patient_sex: str) -> TriageResult:
    if len(selected_symptoms) == 0:
        return TriageResult(label="Healthy (asymptomatic)", probability=100.0,
                             severity="ROUTINE", show_shap=False, guardrail_triggered=False)

    proba = model.predict_proba(input_data)[0]
    pred_idx = int(np.argmax(proba))
    pred = model.classes_[pred_idx]
    probability = proba[pred_idx] * 100
    
    # BIOLOGICAL MISMATCH GUARDRAIL
    pred_lower = str(pred).lower()
    is_female_disease = any(cond in pred_lower for cond in FEMALE_ONLY_CONDITIONS)
    is_male_disease = any(cond in pred_lower for cond in MALE_ONLY_CONDITIONS)

    if patient_sex == "Male" and is_female_disease:
        return TriageResult(
            label="Biological Mismatch", probability=100.0, severity="INFO",
            show_shap=False, guardrail_triggered=True,
            guardrail_reason=f"Model predicted '{pred}', which is biologically incompatible with a Male patient. Clinical review required."
        )
    if patient_sex == "Female" and is_male_disease:
        return TriageResult(
            label="Biological Mismatch", probability=100.0, severity="INFO",
            show_shap=False, guardrail_triggered=True,
            guardrail_reason=f"Model predicted '{pred}', which is biologically incompatible with a Female patient. Clinical review required."
        )

    is_severe = pred in SEVERE_DISEASES

    if probability < 35.0:
        return TriageResult(label=str(pred), probability=probability, severity="INFO",
                             show_shap=True, guardrail_triggered=False)
    if is_severe and probability >= 70.0:
        return TriageResult(label=str(pred), probability=probability, severity="IMMEDIATE",
                             show_shap=True, guardrail_triggered=False)
    if is_severe and probability < 70.0:
        return TriageResult(label=str(pred), probability=probability, severity="URGENT",
                             show_shap=True, guardrail_triggered=False)
    if (not is_severe) and probability >= 60.0:
        return TriageResult(label=str(pred), probability=probability, severity="ROUTINE",
                             show_shap=True, guardrail_triggered=False)
    return TriageResult(label=str(pred), probability=probability, severity="INFO",
                         show_shap=True, guardrail_triggered=False)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _fit_with_holdout(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
    """Train a RandomForest, report holdout accuracy, then refit on all data
    so the deployed model uses every available row."""
    stratify = y if y.value_counts().min() >= 2 else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=stratify
        )
        probe = RandomForestClassifier(n_estimators=200, random_state=42)
        probe.fit(X_train, y_train)
        acc = accuracy_score(y_test, probe.predict(X_test))
    except ValueError:
        acc = float("nan")

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y)
    return model, acc


def top_shap_factors(explainer_values, input_data: pd.DataFrame, label_map: dict,
                      pred_class_idx: Optional[int] = None, k: int = 3) -> list:
    """Return the top-k features driving this prediction as
    (readable_name, 'raised'/'lowered', abs_shap_value, raw_input_value)."""
    values = explainer_values.values
    if values.ndim == 3:
        row = values[0, :, pred_class_idx] if pred_class_idx is not None else values[0, :, 0]
    else:
        row = values[0]

    order = np.argsort(-np.abs(row))[:k]
    out = []
    for idx in order:
        col = input_data.columns[idx]
        name = label_map.get(col, col)
        direction = "raised" if row[idx] > 0 else "lowered"
        out.append((name, direction, float(abs(row[idx])), input_data.iloc[0][col]))
    return out
