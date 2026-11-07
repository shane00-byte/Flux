"""
app.py
======
Flux AI Multi-Specialty Triage Hub — Streamlit UI.

All ML training/prediction logic lives in triage_engine.py; all report
formatting lives in report_generator.py. This file is UI orchestration
only: registration, clinical input collection, results display, and
the triage queue dashboard.
"""

import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
import re 

from report_generator import build_html_report, build_pdf_report, build_text_report, now_str
from triage_engine import (
    CARDIOLOGY_LABELS,
    DIABETES_LABELS,
    SEVERITY,
    encode_cardiology_input,
    evaluate_cardiology,
    evaluate_diabetes,
    evaluate_symptoms,
    load_diabetes_data,
    load_heart_data,
    load_symptom_data,
    top_shap_factors,
    train_cardiology_model,
    train_diabetes_model,
    train_symptom_model,
)

DATA_DIR = "."  # heart.csv, diabetes.csv, dataset.csv are expected here

# ---------------------------------------------------------------------------
# PAGE CONFIG & STYLE
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Flux AI Triage Hub", page_icon="🏥", layout="wide",
                    initial_sidebar_state="expanded")

st.markdown("""
<style>
    .step-caption {
        color: #64748B; font-size: 0.85rem; letter-spacing: 0.02em;
        margin-bottom: 0.5rem; /* Fixed overlap */
    }

    .patient-banner {
        background: rgba(128,128,128,0.07);
        border: 1px solid rgba(128,128,128,0.25); border-left: 6px solid #145C6B;
        border-radius: 8px; padding: 0.85rem 1.1rem; margin-bottom: 0.6rem;
    }

    .severity-badge {
        display: inline-block; padding: 0.2rem 0.7rem; border-radius: 999px;
        font-weight: 600; font-size: 0.85rem; color: #FFFFFF;
    }

    .triage-report-card {
        background: rgba(128,128,128,0.07);
        border: 1px solid rgba(128,128,128,0.25); border-radius: 8px;
        padding: 1.1rem 1.3rem; font-family: 'SFMono-Regular', Consolas, 'Liberation Mono',
        Menlo, monospace;
    }
    .triage-report-card hr { border: none; border-top: 1px solid rgba(128,128,128,0.25); margin: 0.6rem 0; }
    .report-meta { color: #64748B; margin: 0 0 0.4rem 0; font-size: 0.85rem; }
    .report-footnote { color: #64748B; font-size: 0.78rem; margin: 0; }

    .factor-row {
        display: flex; justify-content: space-between; padding: 0.35rem 0;
        border-bottom: 1px solid rgba(128,128,128,0.15); font-size: 0.92rem;
    }

    @media print {
        [data-testid="stSidebar"], [data-testid="stHeader"] { display: none !important; }
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------------------------
def init_state():
    defaults = {
        "patient_registered": False,
        "patient": {"name": "", "age": 45, "sex": "Male", "height": 175.0,
                     "weight": 70.0, "bmi": 22.9, "address": ""},
        "triage_log": [],       
        "last_result": None,    
        "last_department": None,
        "last_shap_input": None,
        "last_pred_class_idx": None,
        "last_labels_map": None,    # Added to prevent key errors
        "last_trained_model": None, # Added to prevent key errors
        "history": ""  # For future use if needed
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()


# ---------------------------------------------------------------------------
# CACHED DATA / MODEL LOADERS
# ---------------------------------------------------------------------------
@st.cache_data
def get_heart_df():
    return load_heart_data(f"{DATA_DIR}/heart.csv")


@st.cache_resource
def get_heart_model(_df):
    return train_cardiology_model(_df)


@st.cache_data
def get_diabetes_df():
    return load_diabetes_data(f"{DATA_DIR}/diabetes.csv")


@st.cache_resource
def get_diabetes_model(_df):
    return train_diabetes_model(_df)


@st.cache_data
def get_symptom_data():
    return load_symptom_data(f"{DATA_DIR}/dataset.csv")


@st.cache_resource
def get_symptom_model(_X, _y):
    return train_symptom_model(_X, _y)


def missing_file_notice(filename: str, schema_hint: str):
    st.error(f"❌ Could not find **{filename}** in the app directory.")
    st.caption(
        f"Expected columns: {schema_hint}. Run `python sample_data.py` to generate "
        "small synthetic CSVs for testing, or add the real dataset alongside app.py."
    )


def severity_badge(severity_key: str) -> str:
    sev = SEVERITY[severity_key]
    return (f'<span class="severity-badge" style="background:{sev["color"]};">'
            f'{sev["icon"]} {sev["label"]}</span>')


# ---------------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------------
st.title("🏥 Flux AI Multi-Specialty Triage Hub")

# ==========================================
# STEP 1: PATIENT REGISTRATION / INTAKE
# ==========================================
if not st.session_state.patient_registered:
    st.markdown('<p class="step-caption">Step 1 of 2 · Patient registration</p>', unsafe_allow_html=True)
    st.subheader("📋 Patient Registration & Intake")
    st.write("Enter primary demographic and physiological baseline details prior to clinical triage.")
    st.divider()

    with st.form("patient_intake_form"):
        col_a, col_b = st.columns(2)
        with col_a:
            p_name = st.text_input("Full Name", placeholder="e.g. Rahul Sharma")
            p_age = st.number_input("Age (Years)", 1, 120, 45)
            p_sex = st.selectbox("Biological Sex", ["Male", "Female"])
            p_address = st.text_area("Residential Address", placeholder="Street, City, District")
        with col_b:
            p_height = st.number_input("Height (cm)", 100.0, 250.0, 175.0)
            p_weight = st.number_input("Weight (kg)", 20.0, 300.0, 70.0)
            p_history = st.text_area("Prior Medical History", placeholder="e.g., Hypertension, family history of diabetes, asthma")
            auto_bmi = p_weight / ((p_height / 100) ** 2)
            st.info(f"Derived Baseline BMI: **{auto_bmi:.1f} kg/m²**")

        submitted = st.form_submit_button("Register & Proceed to Clinical Triage", type="primary")

        if submitted:
            if not p_name.strip():
                st.error("Please enter the patient's name before proceeding.")
            else:
                st.session_state.patient = {
                    "name": p_name.strip(), "age": p_age, "sex": p_sex,
                    "height": p_height, "weight": p_weight, "bmi": auto_bmi,
                    "address": p_address.strip(), "history": p_history.strip() or "None reported",
                }
                st.session_state.patient_registered = True
                st.session_state.last_result = None
                st.rerun()

    # Enter-key moves focus to the next field / submits the form on the last one.
    st.iframe("""
    <script>
    const doc = window.parent.document;
    function setupEnterNavigation() {
        const inputs = Array.from(doc.querySelectorAll('input[type="text"], input[type="number"], textarea'));
        if (inputs.length === 0) { setTimeout(setupEnterNavigation, 500); return; }
        inputs.forEach((input, index) => {
            input.onkeydown = function (e) {
                if (e.key !== 'Enter' || input.tagName === 'TEXTAREA') return;
                e.preventDefault();
                if (index < inputs.length - 1) {
                    inputs[index + 1].focus();
                } else {
                    const btn = doc.querySelector('button[kind="primaryFormSubmit"]');
                    if (btn) btn.click();
                }
            };
        });
    }
    setTimeout(setupEnterNavigation, 500);
    </script>
    """, height=1)

# ==========================================
# STEP 2: CLINICAL TRIAGE & DIAGNOSTICS
# ==========================================
else:
    patient = st.session_state.patient
    st.markdown('<p class="step-caption">Step 2 of 2 · Clinical triage</p>', unsafe_allow_html=True)

    col_p1, col_p2 = st.columns([5, 1])
    with col_p1:
        st.markdown(
            f'<div class="patient-banner">'
            f'<b>Patient:</b> {patient["name"]} &nbsp;|&nbsp; '
            f'<b>Age:</b> {patient["age"]} &nbsp;|&nbsp; '
            f'<b>Sex:</b> {patient["sex"]} &nbsp;|&nbsp; '
            f'<b>BMI:</b> {patient["bmi"]:.1f} kg/m² &nbsp;|&nbsp; '
            f'<b>Address:</b> {patient["address"] or "N/A"}'
            f'</div>', unsafe_allow_html=True
        )
    with col_p2:
        if st.button("🔄 New Patient"):
            st.session_state.patient_registered = False
            st.session_state.last_result = None
            st.rerun()

    tab_triage, tab_queue = st.tabs(["🩺 Clinical Triage", f"📋 Triage Queue ({len(st.session_state.triage_log)})"])

    # ------------------------------------------------------------------
    # TAB 1: TRIAGE
    # ------------------------------------------------------------------
    with tab_triage:
        department = st.sidebar.selectbox("Select Triage Department", [
            "Cardiology (Acute Vitals)",
            "Endocrinology (Metabolic Labs)",
            "General Medicine (Symptoms)",
        ])
        if st.session_state.last_department != department:
            st.session_state.last_result = None  # switching department clears the old result
        st.sidebar.divider()
        st.sidebar.header("Clinical Parameters")

        try:
            # ============================= CARDIOLOGY =============================
            if department == "Cardiology (Acute Vitals)":
                st.subheader("🫀 Cardiology Triage")
                df = get_heart_df()
                trained = get_heart_model(df)
                age, sex = patient["age"], patient["sex"]

                with st.sidebar.expander("Vitals", expanded=True):
                    trestbps = st.number_input("Resting Blood Pressure", 80, 250, 120)
                    chol = st.number_input("Cholesterol", 100, 600, 200)
                    thalch = st.number_input("Max Heart Rate Achieved", 60, 220, 150)
                with st.sidebar.expander("Cardiac history", expanded=True):
                    cp = st.selectbox("Chest Pain Type", ['typical angina', 'atypical angina', 'non-anginal', 'asymptomatic'])
                    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [True, False])
                    restecg = st.selectbox("Resting ECG", ['normal', 'st-t abnormality', 'lv hypertrophy'])
                    exang = st.selectbox("Exercise Induced Angina", [False, True])
                    oldpeak = st.number_input("ST Depression", 0.0, 10.0, 1.0)
                    slope = st.selectbox("ST Segment Slope", ['upsloping', 'flat', 'downsloping'])
                    ca = st.selectbox("Fluoroscopy Colored Vessels", [0.0, 1.0, 2.0, 3.0])
                    thal = st.selectbox("Thalassemia", ['normal', 'fixed defect', 'reversable defect'])

                raw_input = {"age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
                             "fbs": fbs, "restecg": restecg, "thalch": thalch, "exang": exang,
                             "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal}
                input_data = encode_cardiology_input(raw_input, trained)
                labels_map = CARDIOLOGY_LABELS

                with st.expander("ℹ️ About this model"):
                    acc_txt = f"{trained.accuracy * 100:.1f}%" if trained.accuracy == trained.accuracy else "n/a"
                    st.caption(f"RandomForest trained on {trained.n_train_rows} records · "
                               f"holdout validation accuracy: {acc_txt}. Decision support only.")

            # ============================= ENDOCRINOLOGY =============================
            elif department == "Endocrinology (Metabolic Labs)":
                st.subheader("🩸 Endocrinology Triage")
                df = get_diabetes_df()
                trained = get_diabetes_model(df)

                with st.sidebar.expander("Lab values", expanded=True):
                    glucose = st.number_input("Glucose Level", 0, 400, 100)
                    bp = st.number_input("Blood Pressure", 0, 250, 70)
                    insulin = st.number_input("Insulin Level", 0, 900, 79)
                with st.sidebar.expander("Additional measures", expanded=True):
                    pregnancies = st.number_input("Pregnancies", 0, 20, 0)
                    skin = st.number_input("Skin Thickness", 0, 100, 20)
                    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)

                bmi = float(patient["bmi"])
                age_diab = patient["age"]
                raw_input = {"Pregnancies": pregnancies, "Glucose": glucose, "BloodPressure": bp,
                             "SkinThickness": skin, "Insulin": insulin, "BMI": bmi,
                             "DiabetesPedigreeFunction": dpf, "Age": age_diab}
                input_data = pd.DataFrame([raw_input], columns=trained.feature_cols)
                labels_map = DIABETES_LABELS

                with st.expander("ℹ️ About this model"):
                    acc_txt = f"{trained.accuracy * 100:.1f}%" if trained.accuracy == trained.accuracy else "n/a"
                    st.caption(f"RandomForest trained on {trained.n_train_rows} records · "
                               f"holdout validation accuracy: {acc_txt}. Decision support only.")

            # ============================= GENERAL MEDICINE =============================
            else:
                st.subheader("🦠 General Medicine Triage")
                X, y, all_symptoms = get_symptom_data()
                trained = get_symptom_model(X, y)
                
                # Filter out biologically impossible symptoms based on patient sex
                female_only_symptoms = ["menstruation"]  
                male_only_symptoms = ["prostate"]        
                
                valid_symptoms = []
                for sym in all_symptoms:
                    sym_lower = sym.lower()
                    if patient["sex"] == "Male" and any(f in sym_lower for f in female_only_symptoms):
                        continue
                    if patient["sex"] == "Female" and any(m in sym_lower for m in male_only_symptoms):
                        continue
                    valid_symptoms.append(sym)

                selected_symptoms = st.sidebar.multiselect("Search & Select Present Symptoms", valid_symptoms)

                input_data = pd.DataFrame(0, index=[0], columns=X.columns)
                for sym in selected_symptoms:
                    input_data.at[0, sym] = 1
                labels_map = {s: s.replace("_", " ").capitalize() for s in all_symptoms}

                with st.expander("ℹ️ About this model"):
                    acc_txt = f"{trained.accuracy * 100:.1f}%" if trained.accuracy == trained.accuracy else "n/a"
                    st.caption(f"RandomForest trained on {trained.n_train_rows} records across "
                               f"{len(trained.classes)} conditions · holdout validation accuracy: "
                               f"{acc_txt}. Decision support only.")

            st.divider()
            run_btn = st.button("Run AI Diagnostics", type="primary")

            if run_btn:
                if department == "Cardiology (Acute Vitals)":
                    result = evaluate_cardiology(trestbps, chol, trained.model, input_data)
                    pred_class_idx = 1
                elif department == "Endocrinology (Metabolic Labs)":
                    result = evaluate_diabetes(bp, glucose, bmi, trained.model, input_data)
                    pred_class_idx = 1
                else:
                    result = evaluate_symptoms(selected_symptoms, trained.model, input_data, patient["sex"])
                    pred_class_idx = (int(np.where(trained.model.classes_ == result.label)[0][0])
                                       if result.show_shap else None)

                st.session_state.last_result = result
                st.session_state.last_department = department
                st.session_state.last_shap_input = input_data
                st.session_state.last_pred_class_idx = pred_class_idx
                st.session_state.last_labels_map = labels_map
                st.session_state.last_trained_model = trained

                st.session_state.triage_log.append({
                    "Time": now_str(), "Patient": patient["name"], "Department": department,
                    "Finding": result.label, "Confidence": round(result.probability, 1),
                    "Priority": SEVERITY[result.severity]["label"], "_severity_rank": SEVERITY[result.severity]["rank"],
                })

            # -------------------- RESULTS (persist across reruns, e.g. a download click) --------------------
            result = st.session_state.last_result
            if result is not None and st.session_state.last_department == department:
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Risk Stratification")
                    sev = SEVERITY[result.severity]
                    st.markdown(severity_badge(result.severity), unsafe_allow_html=True)
                    st.markdown(f"### {result.label}")
                    st.metric("AI Confidence", f"{result.probability:.1f}%")
                    if result.guardrail_triggered:
                        st.error(f"🚨 {result.guardrail_reason}")
                    elif result.severity == "IMMEDIATE":
                        st.error(f"{sev['icon']} Immediate clinical attention recommended.")
                    elif result.severity == "URGENT":
                        st.warning(f"{sev['icon']} Elevated risk — clinical review advised.")
                    elif result.severity == "ROUTINE":
                        st.success(f"{sev['icon']} Routine / low risk.")
                    else:
                        st.info(f"{sev['icon']} Findings are inconclusive — clinical judgment required.")

                if result.show_shap:
                    with col2:
                        st.subheader("Diagnostic Transparency (SHAP)")
                        model_for_shap = st.session_state.last_trained_model.model
                        explainer = shap.TreeExplainer(model_for_shap)
                        shap_obj = explainer(st.session_state.last_shap_input)

                        fig, ax = plt.subplots()
                        pci = st.session_state.last_pred_class_idx
                        if len(shap_obj.shape) == 3:
                            shap.plots.waterfall(shap_obj[0, :, pci], show=False)
                        else:
                            shap.plots.waterfall(shap_obj[0], show=False)
                        st.pyplot(fig)
                        plt.close(fig)

                        factors = top_shap_factors(shap_obj, st.session_state.last_shap_input,
                                                    st.session_state.last_labels_map, pred_class_idx=pci)
                        st.caption("**In plain terms — top factors behind this result:**")
                        for name, direction, _, value in factors:
                            arrow = "⬆️" if direction == "raised" else "⬇️"
                            st.markdown(
                                f'<div class="factor-row"><span>{arrow} <b>{name}</b> '
                                f'(value: {value}) {direction} the predicted risk</span></div>',
                                unsafe_allow_html=True,
                            )

                # -------------------- PRINTABLE / EXPORTABLE REPORT --------------------
                st.divider()
                st.subheader("📄 Clinical Triage Report")
                timestamp = now_str()
                st.markdown(build_html_report(patient, department, result, timestamp),
                            unsafe_allow_html=True)

                col_print1, col_print2, col_print3 = st.columns([1, 1, 2])
                
                # Sanitize the filename to prevent OS save errors
                safe_name = re.sub(r'[\\/*?:"<>|]', "", patient['name']).strip().replace(' ', '_')
                safe_name = safe_name or "Patient"  # Fallback if name is entirely illegal characters
                
                with col_print1:
                    if st.button("🖨️ Print Patient Slip"):
                        st.iframe("<script>window.print();</script>", height=1)
                with col_print2:
                    st.download_button(
                        "💾 Download .txt",
                        data=build_text_report(patient, department, result, timestamp),
                        file_name=f"Triage_{safe_name}.txt",
                        mime="text/plain",
                    )
                with col_print3:
                    st.download_button(
                        "📑 Download PDF",
                        data=build_pdf_report(patient, department, result, timestamp),
                        file_name=f"Triage_{safe_name}.pdf",
                        mime="application/pdf",
                    )

        except FileNotFoundError:
            schema = {
                "Cardiology (Acute Vitals)": "age, sex, cp, trestbps, chol, fbs, restecg, thalch, exang, oldpeak, slope, ca, thal, num",
                "Endocrinology (Metabolic Labs)": "Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome",
                "General Medicine (Symptoms)": "Disease, Symptom_1 ... Symptom_n",
            }[department]
            fname = {"Cardiology (Acute Vitals)": "heart.csv", "Endocrinology (Metabolic Labs)": "diabetes.csv",
                     "General Medicine (Symptoms)": "dataset.csv"}[department]
            missing_file_notice(fname, schema)
        except Exception as e:
            st.error(f"❌ Diagnostic Pipeline Error: {e}")

    # ------------------------------------------------------------------
    # TAB 2: TRIAGE QUEUE & SYNDROMIC SURVEILLANCE DASHBOARD
    # ------------------------------------------------------------------
    with tab_queue:
        st.subheader("📋 Session Triage Queue")
        log = st.session_state.triage_log
        if not log:
            st.info("No patients have been triaged yet in this session. Results appear here "
                     "after you run diagnostics.")
        else:
            counts = {"IMMEDIATE": 0, "URGENT": 0, "ROUTINE": 0, "INFO": 0}
            for entry in log:
                for key, sev in SEVERITY.items():
                    if sev["label"] == entry["Priority"]:
                        counts[key] += 1

            m1, m2, m3, m4 = st.columns(4)
            m1.metric(f"{SEVERITY['IMMEDIATE']['icon']} Immediate", counts["IMMEDIATE"])
            m2.metric(f"{SEVERITY['URGENT']['icon']} Urgent", counts["URGENT"])
            m3.metric(f"{SEVERITY['ROUTINE']['icon']} Routine", counts["ROUTINE"])
            m4.metric(f"{SEVERITY['INFO']['icon']} Inconclusive", counts["INFO"])

            queue_df = pd.DataFrame(log).sort_values("_severity_rank").drop(columns=["_severity_rank"])
            queue_df["Patient"] = queue_df["Patient"].apply(
                lambda x: f"'{x}" if str(x).startswith(('=', '+', '-', '@')) else x
            )
            st.dataframe(queue_df, use_container_width=True, hide_index=True)

            col_q1, col_q2 = st.columns([1, 4])
            with col_q1:
                st.download_button(
                    "💾 Export queue (.csv)",
                    data=queue_df.to_csv(index=False),
                    file_name="triage_queue.csv",
                    mime="text/csv",
                )
            with col_q2:
                if st.button("🗑️ Clear queue"):
                    st.session_state.triage_log = []
                    st.rerun()

            # --- THE GROUNDBREAKING UPGRADE: EPIDEMIC SURVEILLANCE ---
            st.divider()
            st.subheader("🌍 Real-Time Syndromic Surveillance Engine")
            st.caption("Active monitoring for infectious disease spatial/temporal clustering in the triage queue.")
            
            infectious_diseases = [
                "Malaria", "Dengue", "Typhoid", "Tuberculosis", "Pneumonia", 
                "Hepatitis B", "Hepatitis C", "Hepatitis D", "Hepatitis E"
            ]
            
            outbreak_alerts = {}
            for entry in log:
                if entry["Department"] == "General Medicine (Symptoms)" and entry["Finding"] in infectious_diseases:
                    disease = entry["Finding"]
                    # Only flag high-confidence or severe cases for the outbreak monitor
                    if entry["Confidence"] >= 60.0:
                        outbreak_alerts[disease] = outbreak_alerts.get(disease, 0) + 1
            
            has_outbreak = False
            for disease, count in outbreak_alerts.items():
                if count >= 2:  # Threshold set to 2 for the expo demonstration
                    has_outbreak = True
                    st.error(f"🚨 **AUTOMATED OUTBREAK ALERT:** Anomaly detected. {count} localized cases of **{disease}** have entered triage during this session. Recommend immediate notification to local public health authorities.")
                    
            if not has_outbreak:
                st.success("✅ **Surveillance Status:** Stable. No anomalous clusters of infectious diseases detected in the current patient volume.")