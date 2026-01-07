import streamlit as st
import pandas as pd
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

st.markdown(
    """
    <h1 style="text-align:center;">❤️ Heart Disease Prediction App</h1>
    <p style="text-align:center;">Clinical Decision Support System</p>
    <hr>
    """,
    unsafe_allow_html=True
)

# ======================================================
# LOAD MODELS & OBJECTS (UNCHANGED)
# ======================================================
model = joblib.load(r'C:\Users\moham\Downloads\Amit_Course\Graduation_Project\Heart_Disease_Prediction\PKL_Files\CatBoost_Model.pkl')
min_max_scaler = joblib.load(r'C:\Users\moham\Downloads\Amit_Course\Graduation_Project\Heart_Disease_Prediction\PKL_Files\MinMax_Scaler.pkl')

ordinal_encoder = joblib.load(r'C:\Users\moham\Downloads\Amit_Course\Graduation_Project\Heart_Disease_Prediction\PKL_Files\Ordinal_Encoder.pkl')
binary_encoder = joblib.load(r'C:\Users\moham\Downloads\Amit_Course\Graduation_Project\Heart_Disease_Prediction\PKL_Files\Binary_Encoders.pkl')
target_encoder = joblib.load(r'C:\Users\moham\Downloads\Amit_Course\Graduation_Project\Heart_Disease_Prediction\PKL_Files\Target_Encoder.pkl')

selected_features = joblib.load(r'C:\Users\moham\Downloads\Amit_Course\Graduation_Project\Heart_Disease_Prediction\PKL_Files\Selected_Features.pkl')
numerical_columns = joblib.load(r'C:\Users\moham\Downloads\Amit_Course\Graduation_Project\Heart_Disease_Prediction\PKL_Files\Numerical_Columns.pkl')
ordinal_columns = joblib.load(r'C:\Users\moham\Downloads\Amit_Course\Graduation_Project\Heart_Disease_Prediction\PKL_Files\Ordinal_Columns.pkl')
binary_columns = joblib.load(r'C:\Users\moham\Downloads\Amit_Course\Graduation_Project\Heart_Disease_Prediction\PKL_Files\Binary_Columns.pkl')

# ======================================================
# SIDEBAR — PERSONAL INFO
# ======================================================
st.sidebar.header("👤 Personal Information")

Age = st.sidebar.number_input("Age", 1, 120, 30)
Gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
BMI = st.sidebar.number_input("BMI", 10.0, 50.0, 25.0)
Sleep_Hours = st.sidebar.number_input("Sleep Hours", 4, 24, 8)

# ======================================================
# MAIN TABS
# ======================================================
tab1, tab2, tab3 = st.tabs([
    "🩺 Medical History",
    "🧪 Lab Results",
    "📊 Lifestyle"
])

# ======================================================
# MEDICAL HISTORY
# ======================================================
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        Blood_Pressure = st.number_input("Blood Pressure (mm Hg)", 50, 200, 120)
        High_Blood_Pressure = st.selectbox("High Blood Pressure", ["Yes", "No"])
        Diabetes = st.selectbox("Diabetes", ["Yes", "No"])
        Family_Heart_Disease = st.selectbox("Family Heart Disease", ["Yes", "No"])

    with col2:
        Cholesterol_Level = st.number_input("Cholesterol Level (mg/dL)", 100, 400, 200)
        Low_HDL_Cholesterol = st.selectbox("Low HDL Cholesterol", ["Yes", "No"])
        High_LDL_Cholesterol = st.selectbox("High LDL Cholesterol", ["Yes", "No"])

# ======================================================
# LAB RESULTS
# ======================================================
with tab2:
    col1, col2 = st.columns(2)

    with col1:
        Triglyceride_Level = st.number_input("Triglyceride Level (mg/dL)", 100, 1000, 200)
        Fasting_Blood_Sugar = st.number_input("Fasting Blood Sugar (mg/dL)", 70, 300, 100)

    with col2:
        CRP_Level = st.number_input("CRP Level (mg/L)", 0.1, 10.0, 1.0)
        Homocysteine_Level = st.number_input("Homocysteine Level (nmol/L)", 0.1, 10.0, 1.0)

# ======================================================
# LIFESTYLE
# ======================================================
with tab3:
    col1, col2 = st.columns(2)

    with col1:
        Smoking = st.selectbox("🚬 Smoking", ["Yes", "No"])
        Exercise_Habits = st.selectbox("🏃 Exercise Habits", ["High", "Medium", "Low"])
        Stress_Level = st.selectbox("😟 Stress Level", ["High", "Medium", "Low"])

    with col2:
        Alcohol_Consumption = st.selectbox("🍺 Alcohol Consumption", ["High", "Medium", "Low"])
        Sugar_Consumption = st.selectbox("🍬 Sugar Consumption", ["Low", "Medium", "High"])

# ======================================================
# FEATURE ENGINEERING (UNCHANGED)
# ======================================================
def age_group(age):
    if age < 40:
        return 'Young'
    elif age <= 59:
        return 'Middle_Aged'
    else:
        return 'Senior'

def Blood_Pressure_CategoryFun(blood):
    if 120 <= blood < 129:
        return 'Normal'
    elif 130 <= blood < 139:
        return 'Elevated'
    else:
        return 'High'

def Cholesterol_Level_category(chol):
    if chol < 200:
        return 'Desirable'
    elif 200 <= chol < 239:
        return 'Borderline_High'
    else:
        return 'High'

def BMI_category(bmi):
    if 18 <= bmi < 24.9:
        return 'Normal_weight'
    elif 24.9 <= bmi < 29.9:
        return 'Overweight'
    else:
        return 'Obesity'

def LifeStyle(smoking, alcohol, exercise, sugar):
    score = 0
    score += 1 if smoking == 'Yes' else 0
    score += 2 if alcohol == 'High' else 1 if alcohol == 'Medium' else 0
    score += 2 if exercise == 'Low' else 1 if exercise == 'Medium' else 0
    score += 2 if sugar == 'High' else 1 if sugar == 'Medium' else 0
    return score

def sleep_Quality(sleep):
    if sleep < 6 or sleep > 9:
        return 'Poor'
    return 'Good'

# ======================================================
# INPUT DATAFRAME (UNCHANGED)
# ======================================================
input_data = pd.DataFrame([{
    'Age': Age,
    'Gender': Gender,
    'Blood Pressure': Blood_Pressure,
    'Cholesterol Level': Cholesterol_Level,
    'Exercise Habits': Exercise_Habits,
    'Smoking': Smoking,
    'Family Heart Disease': Family_Heart_Disease,
    'Diabetes': Diabetes,
    'BMI': BMI,
    'High Blood Pressure': High_Blood_Pressure,
    'Low HDL Cholesterol': Low_HDL_Cholesterol,
    'High LDL Cholesterol': High_LDL_Cholesterol,
    'Alcohol Consumption': Alcohol_Consumption,
    'Stress Level': Stress_Level,
    'Sleep Hours': Sleep_Hours,
    'Sugar Consumption': Sugar_Consumption,
    'Triglyceride Level': Triglyceride_Level,
    'Fasting Blood Sugar': Fasting_Blood_Sugar,
    'CRP Level': CRP_Level,
    'Homocysteine Level': Homocysteine_Level,
    'Age_Group': age_group(Age),
    'Blood_Pressure_Category': Blood_Pressure_CategoryFun(Blood_Pressure),
    'Cholesterol_Category': Cholesterol_Level_category(Cholesterol_Level),
    'BMI_Category': BMI_category(BMI),
    'life_style_score': LifeStyle(
        Smoking, Alcohol_Consumption, Exercise_Habits, Sugar_Consumption
    ),
    'Sleep_Quality': sleep_Quality(Sleep_Hours)
}])

# ======================================================
# PREPROCESSING (UNCHANGED)
# ======================================================
input_data[numerical_columns] = min_max_scaler.transform(input_data[numerical_columns])
input_data[ordinal_columns] = ordinal_encoder.transform(input_data[ordinal_columns])

for col in binary_columns:
    input_data[col] = binary_encoder[col].transform(input_data[col])

model_input = input_data[selected_features]

# ======================================================
# PREDICTION + UI ADDITIONS
# ======================================================
st.markdown("---")
st.subheader("🔍 Prediction")

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_btn = st.button("❤️ Predict Heart Disease", use_container_width=True)

if predict_btn:
    with st.spinner("Analyzing patient data..."):
        proba = model.predict_proba(model_input)[0]
        yes_index = np.where(target_encoder.classes_ == "Yes")[0][0]
        probability = proba[yes_index]
        result = "Yes" if probability > 0.5 else "No"

    # -------------------------------
    # RESULT
    # -------------------------------
    if result == "Yes":
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")

    # -------------------------------
    # 📈 RISK PROBABILITY GAUGE
    # -------------------------------
    st.subheader("📈 Heart Disease Risk Probability")

    st.metric(
        label="Heart Disease Risk",
        value=f"{probability*100:.2f} %",
        delta="High Risk" if probability > 0.5 else "Low Risk"
    )


# ======================================================
# FOOTER
# ======================================================
st.markdown("---")
st.caption("⚠️ Educational use only — not a medical diagnosis")
