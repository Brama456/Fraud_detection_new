from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load("xgb_fraud_pipeline.joblib")

class ClaimInput(BaseModel):
    claim_id: str
    customer_id: str
    vehicle_type: str
    vehicle_make: str
    vehicle_model: str
    vehicle_age: int
    engine_capacity: int
    registration_zone: str
    coverage_type: str
    ncb_percentage: int
    add_ons: int
    policy_tenure: int
    claim_duration: int
    vehicle_ownership_years: int
    age: int
    gender: str
    driving_experience: int
    previous_claims: int
    customer_tenure: int
    claim_channel: str
    incident_time: str
    claim_submission_delay: int
    location_type: str
    reported_to_police: str
    damage_severity: str
    driver_at_fault: str
    odometer_reading: int
    idv: int
    claim_amount: int
    repair_cost_estimate: int
    claim_history: str
    description_of_current_damage: str

@app.post("/predict")
def predict_fraud(data: ClaimInput):
    df = pd.DataFrame([data.dict()])

    # Add feature engineering (same as in your Streamlit app)
    df['experience_age_ratio'] = df['driving_experience'] / df['age']
    df['claim_to_idv_ratio'] = df['claim_amount'] / df['idv']
    df['repair_to_claim_ratio'] = df['repair_cost_estimate'] / df['claim_amount']
    df['vehicle_value_per_year'] = df['idv'] / (df['vehicle_age'] + 1)
    df['claim_amount_per_year'] = df['claim_amount'] / (df['policy_tenure'] + 1)
    df['previous_claim_ratio'] = df['previous_claims'] / (df['customer_tenure'] + 1)
    df['claim_delay_weighted'] = df['claim_submission_delay'] / (df['claim_duration'] + 1)
    df['severity_score'] = df['damage_severity'].map({'minor': 1, 'moderate': 2, 'severe': 3})
    df['coverage_risk_score'] = df['coverage_type'].map({'third-party': 2, 'comprehensive': 1})
    df['channel_risk_score'] = df['claim_channel'].map({'Online': 1.5, 'Agent': 1.0, 'Branch': 0.5})

    prob = model.predict_proba(df)[0][1]
    prediction = int(prob >= 0.5)
    return {"prediction": prediction, "probability": float(prob)}
@app.get("/")
def read_root():
    return {"message": "Welcome to the Insurance Fraud Detection API. Use POST /predict with claim data."}
