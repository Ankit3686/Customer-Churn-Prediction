import streamlit as st
import pandas as pd
import pickle

# Load model and encoders using pickle
with open("Customer_pro.pkl", "rb") as f:
    model = pickle.load(f)

with open("features.pkl", "rb") as f:
    features = pickle.load(f)

with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)  # This must be a dictionary!

# Verify label_encoders is a dictionary (optional debug)
if not isinstance(label_encoders, dict):
    st.error("❌ 'label_encoders.pkl' file must contain a dictionary, but got something else.")
    st.stop()

st.set_page_config(page_title="Customer Churn Predictor", layout="wide")
st.title("📊 Telco Customer Churn Prediction App")

st.markdown("Predict whether a customer will churn (leave your service) using their demographic, usage, and billing information.")

# ---- Sidebar explanation ----
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    **Churn** means:  
    When a customer stops using your service.  
    This app helps you predict whether a customer is likely to leave your company's service or not.
    """)
    st.write("🔍 Model used: Random Forest / XGBoost")
    st.write("📁 Dataset: Telco Customer Churn")

# ---- Input UI ----
st.subheader("🔧 Customer Information Input")

col1, col2 = st.columns(2)
user_input = {}

with col1:
    st.markdown("### 👤 Demographics")
    for col in ['gender', 'SeniorCitizen', 'Partner', 'Dependents']:
        if col == 'SeniorCitizen':
            user_input[col] = st.radio(col, [0, 1], horizontal=True)
        else:
            le = label_encoders.get(col)
            if le:
                options = list(le.classes_)
                selected = st.selectbox(col, options)
                user_input[col] = le.transform([selected])[0]
            else:
                st.error(f"❌ Encoder not found for column: {col}")
                st.stop()

    st.markdown("### 📱 Service Usage")
    for col in ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']:
        le = label_encoders.get(col)
        if le:
            options = list(le.classes_)
            selected = st.selectbox(col, options)
            user_input[col] = le.transform([selected])[0]
        else:
            st.error(f"❌ Encoder not found for column: {col}")
            st.stop()

with col2:
    st.markdown("### 📄 Contract & Billing")
    for col in ['Contract', 'PaperlessBilling', 'PaymentMethod']:
        le = label_encoders.get(col)
        if le:
            options = list(le.classes_)
            selected = st.selectbox(col, options)
            user_input[col] = le.transform([selected])[0]
        else:
            st.error(f"❌ Encoder not found for column: {col}")
            st.stop()

    user_input['MonthlyCharges'] = st.number_input("MonthlyCharges", min_value=0.0, value=70.0)
    user_input['TotalCharges'] = st.number_input("TotalCharges", min_value=0.0, value=1000.0)
    user_input['tenure'] = st.slider("tenure (in months)", 0, 100, value=12)

# ---- Prediction ----
if st.button("🚀 Predict Churn"):
    input_df = pd.DataFrame([user_input])

    try:
        input_df = input_df[features]  # ensure correct column order
    except KeyError as e:
        st.error(f"❌ Feature mismatch: {e}")
        st.stop()

    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    st.subheader("📈 Prediction Result")
    if prediction == 1:
        st.error("🔴 **Customer is likely to churn!**")
    else:
        st.success("🟢 **Customer is likely to stay.**")

    st.metric(label="📊 Probability of Churn", value=f"{prob:.2%}")

    if prediction == 1 and prob >= 0.7:
        st.warning("⚠️ Take action to retain this customer (e.g. offer discounts or support).")
