import streamlit as st
import pandas as pd
import joblib
from fpdf import FPDF

st.set_page_config(page_title="Stroke Prediction System", layout="centered")

# =========================================================
# GLOBAL CSS STYLING
# =========================================================
st.markdown("""
<style>

.stApp {
    background:  #B6B09F;
    font-family: 'Segoe UI', Arial, sans-serif;
}
/* Title and Headings */
h1, h2, h3, h4, h5, h6 {
    color: #000000 !important;
}

# /* Body Text */
# p, span, label, div {
#     color: #000000;
# }


/* White card box */
# .container {
#     background-color: white;
#     padding: 30px;
#     border-radius: 18px;
#     box-shadow: 0 6px 18px rgba(0,0,0,0.12);
# }

/* Title */
h1, h2, h3 {
    color: #222222;
}

/* Subheading */
.block-title {
    color: #004080;
    font-size: 20px;
    padding-bottom: 6px;
}

/* Buttons */
.stButton > button {
    background-color: #222222;
    color: white !important;
    border-radius: 8px;
    padding: 10px 20px;
    border: none;
}

.stButton > button:hover {
    background-color: #155a8a;
    color: white;
}

/* Risk messages */
.success-box {
    background-color: #08CB00;
    padding: 12px;
    border-radius: 10px;
}

.warning-box {
    background-color: #fff5e6;
    padding: 12px;
    border-radius: 10px;
}

.error-box {
    background-color: #BE3144;
    padding: 12px;
    border-radius: 10px;
}

/* Footer */
.footer {
    text-align: center;
    color: gray;
    margin-top: 30px;
    font-size: 12px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# SESSION STATE
# =========================================================
if "page" not in st.session_state:
    st.session_state.page = "name_page"

if "patient_name" not in st.session_state:
    st.session_state.patient_name = ""

if "results" not in st.session_state:
    st.session_state.results = None


# =========================================================
# LOAD MODEL
# =========================================================
model = joblib.load("stroke_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")


# =========================================================
# HUMAN LABELS
# =========================================================
def readable(f):
    mapping = {
        "num__age": "Age",
        "num__avg_glucose_level": "Average Glucose Level",
        "num__bmi": "Body Mass Index (BMI)",
        "num__hypertension": "Hypertension",
        "num__heart_disease": "Heart Disease",
        "cat__gender_Male": "Gender: Male",
        "cat__gender_Female": "Gender: Female",
        "cat__ever_married_Yes": "Married",
        "cat__ever_married_No": "Not Married",
        "cat__Residence_type_Urban": "Urban Resident",
        "cat__Residence_type_Rural": "Rural Resident",
        "cat__smoking_status_smokes": "Smoker",
        "cat__smoking_status_never smoked": "Never Smoked",
        "cat__smoking_status_formerly smoked": "Former Smoker",
        "cat__work_type_Private": "Private Job",
        "cat__work_type_Self-employed": "Self Employed",
        "cat__work_type_Govt_job": "Government Job",
        "cat__work_type_children": "Child",
        "cat__work_type_Never_worked": "Never Worked",
        "cat__heart_disease_0": "No Heart Disease",
        "cat__hypertension_0": "No Hypertension"
    }
    return mapping.get(f, f)


# =========================================================
# PAGE 1
# =========================================================
def name_page():

    st.markdown("<div class='container'>", unsafe_allow_html=True)

    st.title("Stroke Risk Prediction System")
    st.write("Intelligent Healthcare Screening Tool")

    st.markdown("<hr>", unsafe_allow_html=True)

    st.subheader("Patient Registration")

    name = st.text_input("Enter Patient Name")

    if st.button("Next"):
        if name.strip() == "":
            st.warning("Please enter a valid name.")
        else:
            st.session_state.patient_name = name
            st.session_state.page = "details_page"
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# PAGE 2
# =========================================================
def details_page():

    st.markdown("<div class='container'>", unsafe_allow_html=True)

    st.title("Patient Health Details")
    st.write(f"Patient Name: {st.session_state.patient_name}")

    age = st.number_input("Age", 1, 120, 45)
    gender = st.selectbox("Gender", ["Male", "Female"])
    hypertension = st.selectbox("Hypertension (0 = No, 1 = Yes)", [0, 1])
    heart_disease = st.selectbox("Heart Disease (0 = No, 1 = Yes)", [0, 1])
    ever_married = st.selectbox("Ever Married", ["Yes", "No"])
    work_type = st.selectbox("Work Type",
                             ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
    residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
    avg_glucose = st.number_input("Average Glucose Level", 50.0, 300.0, 120.0)
    bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
    smoking_status = st.selectbox("Smoking Status",
                                  ["never smoked", "formerly smoked", "smokes"])

    if st.button("Predict Stroke Risk"):

        df = pd.DataFrame({
            "gender": [gender],
            "age": [age],
            "hypertension": [hypertension],
            "heart_disease": [heart_disease],
            "ever_married": [ever_married],
            "work_type": [work_type],
            "Residence_type": [residence_type],
            "avg_glucose_level": [avg_glucose],
            "bmi": [bmi],
            "smoking_status": [smoking_status]
        })

        X = preprocessor.transform(df)
        cols = preprocessor.get_feature_names_out()

        X_df = pd.DataFrame(
            X.toarray() if hasattr(X, "toarray") else X,
            columns=cols
        )

        prob = model.predict_proba(X_df)[0][1]

        if prob < 0.30:
            risk = "Low Risk"
        elif prob < 0.60:
            risk = "Medium Risk"
        else:
            risk = "High Risk"

        st.session_state.results = {
            "prob": prob,
            "risk": risk,
            "features": X_df.iloc[0]
        }

        st.session_state.page = "results_page"
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# PAGE 3
# =========================================================
def results_page():

    st.markdown("<div class='container'>", unsafe_allow_html=True)

    name = st.session_state.patient_name
    res = st.session_state.results

    prob = res["prob"]
    risk = res["risk"]

    st.title("Stroke Risk Prediction Result")

    st.write(f"Patient Name: {name}")

    if risk == "Low Risk":
        st.markdown("<div class='success-box'>Low Risk Detected</div>", unsafe_allow_html=True)
    elif risk == "Medium Risk":
        st.markdown("<div class='warning-box'>Moderate Risk Detected</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='error-box'>High Risk Detected</div>", unsafe_allow_html=True)

    st.write(f"Predicted Stroke Probability: {prob:.2f}")

    # Summary text
    if risk == "Low Risk":
        summary = "The likelihood of stroke is currently low. Continue healthy lifestyle habits."
    elif risk == "Medium Risk":
        summary = "Moderate stroke risk detected. Medical advice is recommended."
    else:
        summary = "High stroke risk detected. Immediate medical consultation is advised."

    st.write(summary)

    # Factor effects
    coeff = model.coef_[0]
    contrib = res["features"] * coeff
    contrib = contrib.sort_values(ascending=False)

    st.subheader("Risk Factor Contribution")

    st.write("Factors increasing risk:")
    for f, v in contrib.head(10).items():
        if v > 0:
            st.write(f"- {readable(f)}")

    st.write("Protective factors:")
    for f, v in contrib.tail(10).items():
        if v < 0:
            st.write(f"- {readable(f)}")

    # PDF report
    st.subheader("Download Patient Report")

    def make_pdf():
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        # Title
        pdf.cell(0, 10, "Stroke Risk Prediction Report", ln=True)

        # Patient details
        pdf.cell(0, 10, f"Patient Name: {name}", ln=True)
        pdf.cell(0, 10, f"Risk Level: {risk}", ln=True)
        pdf.cell(0, 10, f"Probability: {prob:.2f}", ln=True)

        # Summary
        pdf.ln(5)
        pdf.multi_cell(0, 10, f"Summary:\n{summary}")

        # Risk Factors
        pdf.ln(5)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Risk Increasing Factors:", ln=True)
        pdf.set_font("Arial", size=12)

        for f, v in contrib.head(10).items():
            if v > 0:
                pdf.cell(0, 10, f"- {readable(f)}", ln=True)

        pdf.ln(3)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Protective (Risk Reducing) Factors:", ln=True)
        pdf.set_font("Arial", size=12)

        for f, v in contrib.tail(10).items():
            if v < 0:
                pdf.cell(0, 10, f"- {readable(f)}", ln=True)

        return pdf.output(dest="S").encode("latin-1")

        pdf.multi_cell(0, 10, f"Summary:\n{summary}")

        return pdf.output(dest="S").encode("latin-1")

    st.download_button("Download PDF Report",
                       data=make_pdf(),
                       file_name="stroke_report.pdf",
                       mime="application/pdf")

    if st.button("Back to Start"):
        st.session_state.page = "name_page"
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# ROUTING
# =========================================================
if st.session_state.page == "name_page":
    name_page()
elif st.session_state.page == "details_page":
    details_page()
else:
    results_page()

# st.markdown("<div class='footer'>Healthcare AI Project UI</div>", unsafe_allow_html=True)
