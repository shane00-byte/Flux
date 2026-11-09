# Flux AI Multi-Specialty Triage Hub

An explainable-AI clinical decision-support demo built with Streamlit. A
patient is registered once, then triaged through one of three
departments (Cardiology, Endocrinology, General Medicine). Each
department trains a RandomForest on a CSV dataset, predicts a risk
level, and explains the prediction with SHAP — plus a plain-language
summary of the top contributing factors.

> **Not a medical device.** This is a decision-support / portfolio
> project. The clinical guardrail thresholds (e.g. "BP ≥ 180 is a
> hypertensive crisis") are illustrative defaults, not values signed
> off by a clinician. Do not use this for real patient care without
> a full clinical and regulatory review.

## Project layout

```
app.py                 Streamlit UI (registration, triage, queue dashboard)
triage_engine.py        Data loading, model training, prediction, SHAP, severity scoring
report_generator.py     HTML / plain-text / PDF report builders
sample_data.py           Generates synthetic heart.csv / diabetes.csv / dataset.csv for local testing
requirements.txt
.streamlit/config.toml  App theme
```

`app.py` only orchestrates the UI — it has no ML or report-formatting
logic of its own, so `triage_engine.py` and `report_generator.py` can be
unit-tested without a running Streamlit session.

## Setup

```bash
pip install -r requirements.txt
```

You need three CSV files in the same folder as `app.py`:

| File | Source dataset | Required columns |
|---|---|---|
| `heart.csv` | [Heart Disease (Kaggle)](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data) | `age, sex, cp, trestbps, chol, fbs, restecg, thalch, exang, oldpeak, slope, ca, thal, num` |
| `diabetes.csv` | [Pima Indians Diabetes (Kaggle)](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) | `Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome` |
| `dataset.csv` | [Disease-Symptom dataset (Kaggle)](https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset) | `Disease, Symptom_1 ... Symptom_n` |

Don't have the real datasets yet? Generate small synthetic ones so you
can click through the app immediately:

```bash
python sample_data.py
```

(These are random and **not clinically meaningful** — swap in the real
datasets before demoing the results.)

Then run:

```bash
streamlit run app.py
```

## What's new in this refined version

**Bug fixes**
- **Report no longer vanishes.** In the original version, clicking
  *Download*, *Print*, or any other button after running diagnostics
  reran the script with the "Run AI Diagnostics" button reset to
  `False`, which hid the whole results section. Results are now kept
  in `st.session_state` so they persist until you change the
  department or register a new patient.
- Streamlit's `components.v1.html` (deprecated) replaced with `st.iframe`;
  `use_container_width` replaced with `width="stretch"`.
- Categorical encoding for Cardiology is now fit once on the training
  data and reused for every prediction, instead of being recomputed
  per-request (which could silently shift category codes).
- Patient-entered text (name/address) is HTML-escaped before it's
  injected into the report card, closing a small self-XSS gap.

**New features**
- **Triage Queue dashboard** — a running, session-level list of every
  patient triaged, with a count of Immediate / Urgent / Routine /
  Inconclusive cases and a CSV export. This turns the tool from a
  single-patient checker into something closer to an actual triage
  board.
- **Plain-language explanation** alongside the SHAP waterfall chart —
  "↑ Cholesterol (value: 280) raised the predicted risk" — so the
  explainability isn't locked behind a chart only a technical user can
  read.
- **Model trust panel** — an "About this model" expander per
  department showing how many records it trained on and its holdout
  validation accuracy.
- **PDF report export**, alongside the original print view and `.txt`
  download.
- **Friendly missing-data messages** — if a CSV isn't found, the app
  tells you which file and which columns it expects instead of
  throwing a raw exception.
- Sidebar clinical inputs are grouped into labeled expanders per
  department instead of one long unstructured list.
- A synthetic sample-data generator (`sample_data.py`) so the app is
  runnable out of the box.

**UI**
- Redesigned around a clinical, low-decoration palette (deep teal +
  neutral slate) with color reserved for severity status, not
  decoration, so an Immediate/Urgent/Routine badge is unambiguous at a
  glance.
- A small step indicator ("Step 1 of 2 / Step 2 of 2") makes the
  registration → triage flow explicit.
- Severity badges, a patient banner, and a redesigned report card.

## Ideas for further extension

- Persist the triage queue to a database instead of session memory, so
  it survives a page refresh / multiple users.
- Add authentication for clinical staff.
- Replace the illustrative guardrail thresholds with values reviewed
  by a clinician, and cite the source for each.
- Add unit tests for `triage_engine.py` (the module is structured so
  this is straightforward — see the manual smoke tests used while
  building this version for a starting point).
