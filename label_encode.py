import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle

# Dataset load karo (CSV path sahi se set karo)
df = pd.read_csv("Telco-Customer-Churn.csv")

# Categorical columns jinke liye encoding chahiye
categorical_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                    'PaperlessBilling', 'PaymentMethod']

# Label encoders ko dictionary mein store karo
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Save label_encoders as dictionary
with open("label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

print("✅ label_encoders.pkl successfully saved in correct format!")

