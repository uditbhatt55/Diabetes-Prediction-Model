import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go

# --- Load models ---
xgb_model = joblib.load('xgb_model.joblib')
lgb_model = joblib.load('lgb_model.joblib')
scaler = joblib.load('scaler.joblib')

# --- Page Config ---
st.set_page_config(page_title=" Diabetes Predictor | Udit Bhatt™", page_icon="", layout="wide")

# --- Custom Header ---
st.markdown("""
    <h1 style='text-align:center; color:#FF4B4B;'> Diabetes Risk Prediction System</h1>
    <h4 style='text-align:center; color:gray;'>An ML Project by <b>Udit Bhatt™</b></h4>
    <hr style='border:1px solid #FF4B4B;'>
""", unsafe_allow_html=True)

# --- Layout Split ---
col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader(" Enter Patient Medical Information")
    with st.form("patient_form"):
        Pregnancies = st.number_input("Pregnancies", 0, 20, 2)
        Glucose = st.number_input("Glucose Level", 0, 200, 120)
        BloodPressure = st.number_input("Blood Pressure", 0, 130, 70)
        SkinThickness = st.number_input("Skin Thickness", 0, 100, 20)
        Insulin = st.number_input("Insulin Level", 0, 900, 80)
        BMI = st.number_input("BMI", 0.0, 70.0, 28.0)
        DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
        Age = st.number_input("Age", 20, 100, 35)
        submitted = st.form_submit_button("Predict ")

with col2:
    st.info("Model Details")
    st.metric("Accuracy", "77.3%")
    st.metric("ROC-AUC", "0.83")
    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966327.png", width=180)
    st.caption("Diabetes Prediction Model")


if submitted:
    data = pd.DataFrame([{
        'Pregnancies': Pregnancies,
        'Glucose': Glucose,
        'BloodPressure': BloodPressure,
        'SkinThickness': SkinThickness,
        'Insulin': Insulin,
        'BMI': BMI,
        'DiabetesPedigreeFunction': DiabetesPedigreeFunction,
        'Age': Age,
        'BMI_Age': BMI * Age,
        'Glucose_Age': Glucose * Age,
        'Glucose_BMI': Glucose * BMI,
        'Age_Group': 2
    }])

    scaled_data = scaler.transform(data)
    p_xgb = xgb_model.predict_proba(scaled_data)[:, 1]
    p_lgb = lgb_model.predict_proba(scaled_data)[:, 1]
    final_proba = (p_xgb + p_lgb) / 2
    pred = "Diabetic" if final_proba[0] > 0.5 else "Non-Diabetic"

    st.markdown("---")
    st.subheader(" Prediction Result")
    if pred == "Diabetic":
        st.error(f"⚠️ {pred} | Probability: {final_proba[0]*100:.2f}%")
    else:
        st.success(f"✅ {pred} | Probability: {final_proba[0]*100:.2f}%")

    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=final_proba[0]*100,
        title={'text': "Diabetes Probability (%)"},
        gauge={'axis': {'range': [0, 100]},
               'bar': {'color': "red" if pred == "Diabetic" else "green"}}
    ))
    st.plotly_chart(fig, use_container_width=True, key="gauge_chart")

    st.write("###  Probability Visualization")

    import plotly.express as px

    prob_df = pd.DataFrame({
        'Category': ['Probability of Diabetes', 'Remaining Healthy'],
        'Value': [final_proba[0]*100, 100 - final_proba[0]*100]
    })

    fig_prob = px.bar(
        prob_df,
        x='Value',
        y='Category',
        orientation='h',
        text='Value',
        color='Category',
        color_discrete_map={
            'Probability of Diabetes': '#E74C3C',
            'Remaining Healthy': '#27AE60'
        },
        title="Patient’s Diabetes Probability Breakdown"
    )
    fig_prob.update_layout(showlegend=False, xaxis_title="%", yaxis_title=None)

    st.plotly_chart(fig_prob, use_container_width=True, key="prob_chart")

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center; color:gray; font-size:15px;'>
Developed with ❤️ by <b>Udit Bhatt™</b><br>
<small>© 2025 | Machine Learning Project</small><br>
<small>Powered by <b style='color:#FF4B4B;'>XGBoost</b> + <b style='color:#FF4B4B;'>LightGBM</b> Ensemble Model</small><br><br>
<b style='color:#FF4B4B;'>👨🏻‍⚕️ Thank you for visiting! Come back soon!⚕️</b>
</div>
""", unsafe_allow_html=True)


