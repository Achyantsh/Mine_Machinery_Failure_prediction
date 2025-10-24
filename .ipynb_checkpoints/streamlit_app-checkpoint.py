import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# Load trained model and scaler
xgb = joblib.load("xgb_model.joblib")
scaler = joblib.load("scaler.joblib")
FEATURES = [
    'Type', 'Air_temperature_K', 'Process_temperature_K',
    'Rotational_speed_rpm', 'Torque_Nm', 'Tool_wear_min'
]
type_map = {'H':0, 'L':1, 'M':2}

st.title("Mining Equipment Failure Prediction")
st.write("Enter sensor readings:")

type_option = st.selectbox("Machine Type", list(type_map.keys()))
inputs = [
    type_map[type_option],
    st.number_input("Air temperature (K)", value=298.2),
    st.number_input("Process temperature (K)", value=308.6),
    st.number_input("Rotational speed (rpm)", value=1500),
    st.number_input("Torque (Nm)", value=40.0),
    st.number_input("Tool wear (min)", value=10)
]

if st.button("Predict"):
    X = pd.DataFrame([inputs], columns=FEATURES)
    X_scaled = scaler.transform(X)
    pred = xgb.predict(X_scaled)[0]
    risk = xgb.predict_proba(X_scaled)[0,1]
    st.write(f"## Prediction: {'Failure' if pred==1 else 'No Failure'}")
    st.write(f"### Predicted Failure Probability: {risk:.2f}")
    # SHAP explainer
    explainer = shap.Explainer(xgb, X_scaled, feature_names=FEATURES)
    shap_values = explainer(X_scaled)
    st.subheader("Feature contribution (SHAP values):")
    plt.figure(figsize=(8,4))
    shap.plots.bar(shap_values, show=False)
    st.pyplot(plt.gcf())
    plt.clf()

    if risk > 0.5:
        st.error("Warning: High risk of failure. Maintenance needed!")
    else:
        st.success("Risk low. Equipment likely healthy.")

st.markdown("---")
st.caption("Powered by XGBoost, SHAP & Streamlit")
