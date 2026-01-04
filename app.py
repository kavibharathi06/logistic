import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Heart Disease Prediction App", layout="centered")

st.title("❤️ Heart Disease Prediction App")
st.write("Enter patient details to predict 10-year heart disease risk")

# Inputs
male = st.number_input("Gender (Male=1, Female=0)", min_value=0, max_value=1, value=1)
age = st.number_input("Age", min_value=30, max_value=80, value=45)
education = st.number_input("Education Level (1–4)", min_value=1, max_value=4, value=2)
currentSmoker = st.number_input("Current Smoker (0/1)", min_value=0, max_value=1, value=0)
cigsPerDay = st.number_input("Cigarettes Per Day", min_value=0.0, max_value=70.0, value=0.0)
BPMeds = st.number_input("On BP Medication (0/1)", min_value=0, max_value=1, value=0)
prevalentStroke = st.number_input("Previous Stroke (0/1)", min_value=0, max_value=1, value=0)
prevalentHyp = st.number_input("Hypertension (0/1)", min_value=0, max_value=1, value=0)
diabetes = st.number_input("Diabetes (0/1)", min_value=0, max_value=1, value=0)
totChol = st.number_input("Total Cholesterol", min_value=100.0, max_value=400.0, value=200.0)
sysBP = st.number_input("Systolic BP", min_value=90.0, max_value=250.0, value=120.0)
diaBP = st.number_input("Diastolic BP", min_value=60.0, max_value=150.0, value=80.0)
BMI = st.number_input("BMI", min_value=15.0, max_value=50.0, value=25.0)
heartRate = st.number_input("Heart Rate", min_value=50.0, max_value=150.0, value=75.0)
glucose = st.number_input("Glucose Level", min_value=40.0, max_value=400.0, value=80.0)

# Predict button
if st.button("Predict"):
    new_data = np.array([[ 
        male, age, education, currentSmoker, cigsPerDay,
        BPMeds, prevalentStroke, prevalentHyp, diabetes,
        totChol, sysBP, diaBP, BMI, heartRate, glucose
    ]])

    new_data_scaled = scaler.transform(new_data)
    prediction = model.predict(new_data_scaled)[0]

    if prediction == 1:
        st.success("⚠️ High Risk of Heart Disease")
    else:
        st.warning("✅ Low Risk of Heart Disease")
