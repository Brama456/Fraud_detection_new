import streamlit as st
import pandas as pd
import random
import joblib
import time
from lime.lime_tabular import LimeTabularExplainer
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Insurance Claim Fraud Detector", layout="centered")

@st.cache_resource
def load_model():
    return joblib.load(r"C:\Users\Bramarambika\Downloads\Workoopolis\fraud_detection_cloud\xgb_fraud_pipeline.joblib")

model = load_model()
st.title("Insurance Fraud Detection")
st.markdown("Provide claim and vehicle details to predict potential fraud.")

damage_parts = ['windshield', 'bumper', 'engine', 'side door', 'mirror', 'paintwork', 'headlight', 'roof', 'rear light', 'tire']
current_damage_templates = [
    "Vehicle hit from behind, resulting in {} damage.",
    "Involved in a collision; {} and {} damaged.",
    "Scratches on the {} after parking mishap.",
    "{} broken during highway incident.",
    "{} and {} misaligned due to pothole hit.",
    "Total damage on {}, needs full replacement."
]
claim_history_templates = [
    "Had {} minor claim(s) in the past.",
    "No previous claims recorded.",
    "Reported {} accident(s) over {} years.",
    "Previously claimed for {} and {} damages.",
    "Frequent claims for {} related issues."
]

def random_claim_history():
    template = random.choice(claim_history_templates)
    if template.count('{}') == 2:
        return template.format(
            random.randint(1, 3), random.randint(2, 5)
        ) if "over" in template else template.format(
            random.choice(damage_parts), random.choice(damage_parts)
        )
    elif template.count('{}') == 1:
        return template.format(random.randint(1, 4))
    return template

def random_damage_description():
    template = random.choice(current_damage_templates)
    if template.count('{}') == 2:
        return template.format(random.choice(damage_parts), random.choice(damage_parts))
    elif template.count('{}') == 1:
        return template.format(random.choice(damage_parts))
    return template

def user_input_features():
    claim_id = st.text_input("Claim ID", "CLM123456")
    customer_id = st.text_input("Customer ID", "CUST123456")
    vehicle_type = st.selectbox("Vehicle Type", ['Sedan', 'SUV', 'Hatchback', 'Truck', 'Bike', 'Cruiser', 'Convertible', 'Electric', 'Luxury', 'Mini'])
    vehicle_make = st.selectbox("Vehicle Make", ['Toyota', 'Honda', 'Ford', 'Hyundai', 'Suzuki', 'Kia', 'Tata', 'Mahindra', 'BMW', 'Audi', 'Nissan'])
    vehicle_model = st.text_input("Vehicle Model", "Corolla")
    vehicle_age = st.number_input("Vehicle Age (years)", 0, 30, 5)
    engine_capacity = st.number_input("Engine Capacity (cc)", 800, 5000, 1200)
    registration_zone = st.selectbox("Registration Zone", ['A', 'B', 'C'])
    coverage_type = st.selectbox("Coverage Type", ['comprehensive', 'third-party'])
    ncb_percentage = st.selectbox("NCB Percentage", [0, 20, 25, 35, 45, 50])
    add_ons = st.number_input("Number of Add-ons", 0, 10, 1)
    policy_tenure = st.number_input("Policy Tenure (years)", 1, 30, 3)
    claim_duration = st.number_input("Claim Duration (days)", 1, 365, 30)
    vehicle_ownership_years = st.number_input("Vehicle Ownership (years)", 0, 30, 3)
    age = st.number_input("Customer Age", 18, 100, 35)
    gender = st.selectbox("Gender", ['male', 'female'])
    driving_experience = st.number_input("Driving Experience (years)", 0, 80, 10)
    previous_claims = st.number_input("Previous Claims", 0, 20, 1)
    customer_tenure = st.number_input("Customer Tenure (years)", 0, 30, 5)
    claim_channel = st.selectbox("Claim Channel", ['Online', 'Agent', 'Branch'])
    incident_time = st.selectbox("Incident Time", ['day', 'night'])
    claim_submission_delay = st.number_input("Claim Submission Delay (days)", 0, 60, 2)
    location_type = st.selectbox("Incident Location Type", ['urban', 'rural', 'highway'])
    reported_to_police = st.selectbox("Reported to Police", ['yes', 'no'])
    damage_severity = st.selectbox("Damage Severity", ['minor', 'moderate', 'severe'])
    driver_at_fault = st.selectbox("Driver At Fault", ['yes', 'no'])
    odometer_reading = st.number_input("Odometer Reading (km)", 0, 500000, 50000)
    idv = st.number_input("Insured Declared Value (INR)", 10000, 5000000, 800000)
    claim_amount = st.number_input("Claim Amount (INR)", 1000, 10000000, 100000)
    repair_cost_estimate = st.number_input("Repair Cost Estimate (INR)", 1000, 10000000, 70000)
    claim_history = st.text_area("Claim History", random_claim_history())
    description_of_current_damage = st.text_area("Description of Current Damage", random_damage_description())

    return pd.DataFrame({
        'claim_id': [claim_id],
        'customer_id': [customer_id],
        'vehicle_type': [vehicle_type],
        'vehicle_make': [vehicle_make],
        'vehicle_model': [vehicle_model],
        'vehicle_age': [vehicle_age],
        'engine_capacity': [engine_capacity],
        'registration_zone': [registration_zone],
        'coverage_type': [coverage_type],
        'ncb_percentage': [ncb_percentage],
        'add_ons': [add_ons],
        'policy_tenure': [policy_tenure],
        'claim_duration': [claim_duration],
        'vehicle_ownership_years': [vehicle_ownership_years],
        'age': [age],
        'gender': [gender],
        'driving_experience': [driving_experience],
        'previous_claims': [previous_claims],
        'customer_tenure': [customer_tenure],
        'claim_channel': [claim_channel],
        'incident_time': [incident_time],
        'claim_submission_delay': [claim_submission_delay],
        'location_type': [location_type],
        'reported_to_police': [reported_to_police],
        'damage_severity': [damage_severity],
        'driver_at_fault': [driver_at_fault],
        'odometer_reading': [odometer_reading],
        'idv': [idv],
        'claim_amount': [claim_amount],
        'repair_cost_estimate': [repair_cost_estimate],
        'claim_history': [claim_history],
        'description_of_current_damage': [description_of_current_damage]
    })

def add_engineered_features(df):
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
    return df

input_df = user_input_features()
input_df = add_engineered_features(input_df)

if st.button("Predict Fraud"):
    with st.spinner("Analyzing claim..."):
        start_time = time.time()
        try:
            pred_prob = model.predict_proba(input_df)[0][1]
            is_fraud = pred_prob >= 0.5
            elapsed_time = time.time() - start_time

            st.markdown("### 🔍 Prediction Result")
            if is_fraud:
                st.error(f"\U0001F6A8 Fraudulent Claim Detected! (Confidence: {pred_prob * 100:.2f}%)")
            else:
                st.success(f"\u2705 Genuine Claim (Confidence: {(1 - pred_prob) * 100:.2f}%)")

            st.info(f"🕒 Prediction Time: {elapsed_time:.4f} seconds")

            preprocessor = model.named_steps["preprocessor"]
            X_transformed = preprocessor.transform(input_df)
            feature_names = preprocessor.get_feature_names_out()

            explainer = LimeTabularExplainer(
                training_data=X_transformed,
                feature_names=feature_names,
                mode="classification"
            )

            exp = explainer.explain_instance(
                X_transformed[0],
                model.named_steps["classifier"].predict_proba,
                num_features=10
            )

            st.markdown("### 🔍 LIME Explanation")
            fig = exp.as_pyplot_figure()
            plt.tight_layout()
            st.pyplot(fig)

        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
