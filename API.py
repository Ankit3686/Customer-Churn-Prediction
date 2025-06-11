from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load('Customer_pro.pkl')
features = joblib.load('features.pkl')
label_encoders = joblib.load('label_encoders.pkl')

class CustomerData(BaseModel):
    data: dict

@app.get("/")
def read():
    return {"Customer Churn Prediction"}
    
@app.post("/predict")
def predict_churn(payload: CustomerData):
    input_data = payload.data
    input_df = pd.DataFrame([input_data])

    # Ensure column order matches
    input_df = input_df[features]

    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]
    return {"prediction": int(prediction), "probability": prob}
