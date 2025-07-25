import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load('insurance_model.pkl')
scaler = joblib.load('scaler.pkl')  # Assumes you scaled age and bmi

st.title("💰 Insurance Charges Predictor")

# --- User Inputs ---
age = st.slider("Age", 18, 100, 30)
bmi = st.slider("BMI", 15.0, 50.0, 25.0)
smoker = st.selectbox("Smoker", ["No", "Yes"])

# --- Encode Inputs ---
smoker = 1 if smoker == "Yes" else 0

# --- Scale age & bmi ---
scaled_values = scaler.transform([[bmi, age]])  # Assuming order: [bmi, age]
bmi_scaled, age_scaled = scaled_values[0]

# --- Final Input ---
X = np.array([[bmi_scaled, age_scaled, smoker]])

# --- Prediction ---
prediction = model.predict(X)[0]

# --- Output ---
st.subheader(f"Estimated Insurance Charges: **${prediction:,.2f}**")
