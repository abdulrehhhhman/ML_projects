import streamlit as st
import pandas as pd
import numpy as np
import pickle

# 1. Page Setup
st.set_page_config(page_title="💧 Full Water Potability Predictor", page_icon="💧", layout="centered")

# 2. Load Trained Model
with open("myModel.pkl", "rb") as f:
    model = pickle.load(f)

# 3. UI
st.title("💧 Full Water Potability Predictor")
st.write("Enter the following water quality parameters to predict if the water is **safe to drink**:")

# 4. Input Form
with st.form("input_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        ph = st.number_input("pH (0–14)", 0.0, 14.0, step=0.1)
        hardness = st.number_input("Hardness (mg/L)", 0.0)
        solids = st.number_input("Solids (ppm)", 0.0)
    with col2:
        chloramines = st.number_input("Chloramines (ppm)", 0.0)
        sulfate = st.number_input("Sulfate (mg/L)", 0.0)
        conductivity = st.number_input("Conductivity (μS/cm)", 0.0)
    with col3:
        organic_carbon = st.number_input("Organic Carbon (ppm)", 0.0)
        trihalomethanes = st.number_input("Trihalomethanes (μg/L)", 0.0)
        turbidity = st.number_input("Turbidity (NTU)", 0.0)

    submitted = st.form_submit_button("🔍 Predict")

# 5. Prediction
if submitted:
    input_data = np.array([[ph, hardness, solids, chloramines, sulfate,
                            conductivity, organic_carbon, trihalomethanes, turbidity]])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("✅ This water is **potable**! Safe to drink.")
    else:
        st.error("❌ This water is **not potable**. Unsafe to drink.")

