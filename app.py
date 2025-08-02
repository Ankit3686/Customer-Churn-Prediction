import streamlit as st
import pandas as pd
import joblib

# Load model, features list, and label encoders (dict format)
model = joblib.load("Customer_pro.pkl")
features = joblib.load("features.pkl")  # List of feature column names
label_encoders = joblib.load("label_encoders.pkl")  # Dict of label encoders

st.set_page_config(page_title="Customer Churn Predictor", layout="wide")
st.title("📊 Telco Customer Churn Prediction App")
st.markdown("Predict customer churn using demographic, usage, and billing info.")

with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    **Churn** means: When a customer stops using your service  
    This app predicts customer churn using ML model (RandomForest/XGBoost)
    """)

st.subheader("🔧 Customer Information Input")
col1, col2 = st.columns(2)
user_input = {}

with col1:
    st.markdown("### 👤 Demographics")
    for col in ['gender', 'SeniorCitizen', 'Partner', 'Dependents']:
        if col == 'SeniorCitizen':
            user_input[col] = st.radio(col, [0, 1], horizontal=True)
        else:
            le = label_encoders[col]
            options = list(le.classes_)
            selected = st.selectbox(col, options)
            user_input[col] = le.transform([selected])[0]

    st.markdown("### 📱 Services")
    for col in ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']:
        le = label_encoders[col]
        options = list(le.classes_)
        selected = st.selectbox(col, options)
        user_input[col] = le.transform([selected])[0]

with col2:
    st.markdown("### 💳 Billing & Contract")
    for col in ['Contract', 'PaperlessBilling', 'PaymentMethod']:
        le = label_encoders[col]
        options = list(le.classes_)
        selected = st.selectbox(col, options)
        user_input[col] = le.transform([selected])[0]

    user_input['MonthlyCharges'] = st.number_input("MonthlyCharges", 0.0, 500.0, 70.0)
    user_input['TotalCharges'] = st.number_input("TotalCharges", 0.0, 10000.0, 1000.0)
    user_input['tenure'] = st.slider("tenure (months)", 0, 100, 12)

if st.button("🚀 Predict Churn"):
    input_df = pd.DataFrame([user_input])
    input_df = input_df[features]
    
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    st.subheader("📈 Prediction Result")
    if prediction == 1:
        st.error("🔴 Customer is likely to churn!")
    else:
        st.success("🟢 Customer is likely to stay.")

    st.metric(label="📊 Churn Probability", value=f"{prob:.2%}")
