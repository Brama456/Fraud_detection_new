from fastapi import FastAPI, Request
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
def predict(data: ClaimInput):
    input_df = pd.DataFrame([data.dict()])
    # Add engineered features (same logic as streamlit)
    input_df['experience_age_ratio'] = input_df['driving_experience'] / input_df['age']
    input_df['claim_to_idv_ratio'] = input_df['claim_amount'] / input_df['idv']
    input_df['repair_to_claim_ratio'] = input_df['repair_cost_estimate'] / input_df['claim_amount']
    input_df['vehicle_value_per_year'] = input_df['idv'] / (input_df['vehicle_age'] + 1)
    input_df['claim_amount_per_year'] = input_df['claim_amount'] / (input_df['policy_tenure'] + 1)
    input_df['previous_claim_ratio'] = input_df['previous_claims'] / (input_df['customer_tenure'] + 1)
    input_df['claim_delay_weighted'] = input_df['claim_submission_delay'] / (input_df['claim_duration'] + 1)
    input_df['severity_score'] = input_df['damage_severity'].map({'minor': 1, 'moderate': 2, 'severe': 3})
    input_df['coverage_risk_score'] = input_df['coverage_type'].map({'third-party': 2, 'comprehensive': 1})
    input_df['channel_risk_score'] = input_df['claim_channel'].map({'Online': 1.5, 'Agent': 1.0, 'Branch': 0.5})

    pred_prob = model.predict_proba(input_df)[0][1]
    return {"fraud_probability": float(pred_prob)}