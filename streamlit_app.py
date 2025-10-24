import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

xgb = joblib.load("xgb_model.joblib")
scaler = joblib.load("scaler.joblib")
feature_cols = ['Type', 'Air temperature [K]', 'Process temperature [K]',
                'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
type_dict = {'L': 0, 'M': 1, 'H': 2}

st.set_page_config(page_title="Mining Failure Predictor", page_icon="⛏", layout="centered")
st.markdown("<h1 style='color:#F7CB3B;'>⛏️ Mining Equipment Failure Prediction</h1>", unsafe_allow_html=True)
st.subheader("Enter sensor readings:")

col1, col2 = st.columns(2)
with col1:
    m_type = st.selectbox("Machine Type", list(type_dict.keys()))
    airtemp = st.number_input("Air temperature [K]", min_value=150.0, max_value=600.0, value=295.0, step=0.01, format="%.2f")
    process_temp = st.number_input("Process temperature [K]", min_value=150.0, max_value=600.0, value=305.0, step=0.01, format="%.2f")
    rot_speed = st.number_input("Rotational speed [rpm]", min_value=0, max_value=10000, value=1650)
with col2:
    torque = st.number_input("Torque [Nm]", min_value=0.0, max_value=700.0, value=40.0, step=0.01, format="%.2f")
    tool_wear = st.number_input("Tool wear [min]", min_value=0, max_value=1000, value=15)

mean_values = {
    'Type': 1,
    'Air temperature [K]': 295.0,
    'Process temperature [K]': 305.0,
    'Rotational speed [rpm]': 1650,
    'Torque [Nm]': 40.0,
    'Tool wear [min]': 15
}

if st.button('Predict'):
    user_row = np.array([[type_dict[m_type], airtemp, process_temp, rot_speed, torque, tool_wear]])
    user_row_scaled = scaler.transform(user_row)
    risk_score = float(xgb.predict_proba(user_row_scaled)[0,1])
    prediction = "🛑 Failure" if risk_score >= 0.4 else "✅ Healthy"
    st.markdown(f"<h2>Prediction: <span style='color:#B026FF;'>{prediction}</span></h2>", unsafe_allow_html=True)
    st.markdown(f"<h3>Failure Probability: <span style='color:#00E676;'>{risk_score:.2f}</span></h3>", unsafe_allow_html=True)
    st.progress(risk_score if risk_score <= 1 else 1.0)
    if risk_score > 0.7:
        st.error("🔴 Very high failure risk! Immediate action/advice required!")
    elif risk_score > 0.4:
        st.warning("🟠 Elevated risk: Plan maintenance soon.")
    else:
        st.success("🟢 Low risk: Machine is likely healthy.")
    
    # --- POP OUT DEPENDENCY CHART ONLY WHEN FAILURE PROBABILITY > 0.4 ---
    if risk_score > 0.4:
        # Instance-level feature dependency
        impacts = []
        for idx, feat in enumerate(feature_cols):
            altered_row = user_row.copy()
            altered_row[0, idx] = mean_values[feat]
            altered_row_scaled = scaler.transform(altered_row)
            altered_score = float(xgb.predict_proba(altered_row_scaled)[0,1])
            impact = risk_score - altered_score
            impacts.append((feat, impact))
        impacts.sort(key=lambda x: abs(x[1]), reverse=True)
        
        st.markdown("#### Which feature drives this prediction most?")
        feat_labels = [x[0] for x in impacts]
        imp_values = [x[1] for x in impacts]
        fig, ax = plt.subplots()
        ax.barh(feat_labels, imp_values, color="#29B6F6")
        ax.set_xlabel("Change in Failure Probability (if set to mean)")
        ax.set_title("Instance-level Feature Dependency")
        st.pyplot(fig)

st.subheader("Global Feature Importance")

if st.button("Show Model Feature Importance"):
    fi = xgb.feature_importances_
    fig, ax = plt.subplots()
    ax.barh(feature_cols, fi, color="#4CAF50")
    ax.set_xlabel("Importance (Gain/Weight/Cover)")
    ax.set_title("XGBoost Global Feature Importance")
    st.pyplot(fig)

st.markdown("<hr><span style='color: #FFD600;'> Mining ML Project by Achyant Shrivastava©️ 2025</span>", unsafe_allow_html=True)
