import streamlit as st
import requests
import pandas as pd
import time
from datetime import datetime

st.set_page_config(page_title="Insurance Fraud Detector", layout="centered")
st.title("Insurance Claim Fraud Prediction")
st.markdown("Enter claim details below:")

# Input form
claim_id = st.text_input("Claim ID", "CLM001")
customer_id = st.text_input("Customer ID", "CUST001")
vehicle_type = st.selectbox("Vehicle Type", ['Sedan', 'SUV', 'Hatchback', 'Truck'])
vehicle_make = st.selectbox("Vehicle Make", ['Toyota', 'Honda', 'Ford', 'Hyundai'])
vehicle_model = st.text_input("Vehicle Model", "Corolla")
vehicle_age = st.number_input("Vehicle Age (years)", 0, 50, 5)
engine_capacity = st.number_input("Engine Capacity (cc)", 600, 6000, 1500)
registration_zone = st.selectbox("Registration Zone", ['A', 'B', 'C'])
coverage_type = st.selectbox("Coverage Type", ['comprehensive', 'third-party'])
ncb_percentage = st.selectbox("NCB Percentage", [0, 20, 25, 35, 45, 50])
add_ons = st.number_input("Number of Add-ons", 0, 20, 2)
policy_tenure = st.number_input("Policy Tenure (years)", 1, 50, 3)
claim_duration = st.number_input("Claim Duration (days)", 1, 365, 30)
vehicle_ownership_years = st.number_input("Vehicle Ownership (years)", 0, 40, 5)
age = st.number_input("Customer Age", 18, 100, 35)
gender = st.selectbox("Gender", ['male', 'female'])
driving_experience = st.number_input("Driving Experience (years)", 0, 80, 10)
previous_claims = st.number_input("Previous Claims", 0, 50, 1)
customer_tenure = st.number_input("Customer Tenure (years)", 0, 50, 6)
claim_channel = st.selectbox("Claim Channel", ['Online', 'Agent', 'Branch'])
incident_time = st.selectbox("Incident Time", ['day', 'night'])
claim_submission_delay = st.number_input("Claim Submission Delay (days)", 0, 90, 2)
location_type = st.selectbox("Incident Location Type", ['urban', 'rural', 'highway'])
reported_to_police = st.selectbox("Reported to Police", ['yes', 'no'])
damage_severity = st.selectbox("Damage Severity", ['minor', 'moderate', 'severe'])
driver_at_fault = st.selectbox("Driver At Fault", ['yes', 'no'])
odometer_reading = st.number_input("Odometer Reading (km)", 0, 500000, 40000)
idv = st.number_input("Insured Declared Value (INR)", 10000, 10000000, 800000)
claim_amount = st.number_input("Claim Amount (INR)", 1000, 10000000, 100000)
repair_cost_estimate = st.number_input("Repair Cost Estimate (INR)", 1000, 10000000, 75000)
claim_history = st.text_area("Claim History", "Had 1 minor claim(s) in the past.")
description_of_current_damage = st.text_area("Current Damage", "Scratches on the side door after parking mishap.")

# Prepare input
data = {
    "claim_id": claim_id,
    "customer_id": customer_id,
    "vehicle_type": vehicle_type,
    "vehicle_make": vehicle_make,
    "vehicle_model": vehicle_model,
    "vehicle_age": vehicle_age,
    "engine_capacity": engine_capacity,
    "registration_zone": registration_zone,
    "coverage_type": coverage_type,
    "ncb_percentage": ncb_percentage,
    "add_ons": add_ons,
    "policy_tenure": policy_tenure,
    "claim_duration": claim_duration,
    "vehicle_ownership_years": vehicle_ownership_years,
    "age": age,
    "gender": gender,
    "driving_experience": driving_experience,
    "previous_claims": previous_claims,
    "customer_tenure": customer_tenure,
    "claim_channel": claim_channel,
    "incident_time": incident_time,
    "claim_submission_delay": claim_submission_delay,
    "location_type": location_type,
    "reported_to_police": reported_to_police,
    "damage_severity": damage_severity,
    "driver_at_fault": driver_at_fault,
    "odometer_reading": odometer_reading,
    "idv": idv,
    "claim_amount": claim_amount,
    "repair_cost_estimate": repair_cost_estimate,
    "claim_history": claim_history,
    "description_of_current_damage": description_of_current_damage
}

# Simple latency function (UI only)
def track_latency(start_time):
    latency = time.time() - start_time
    return latency

# Prediction call
if st.button("Predict Fraud"):
    with st.spinner("Analyzing claim..."):
        try:
            start_time = time.time()
            response = requests.post(
                "https://fastapi-dot-fraud-detection-464016.uc.r.appspot.com/predict",
                json=data
            )

            if response.status_code == 200:
                result = response.json()
                pred = result["prediction"]
                prob = result["probability"]
                elapsed = track_latency(start_time)

                st.markdown("### 🔍 Prediction Result")
                if pred == 1:
                    st.error(f"🚨 Fraudulent Claim Detected! (Confidence: {prob * 100:.2f}%)")
                else:
                    st.success(f"✅ Genuine Claim (Confidence: {(1 - prob) * 100:.2f}%)")
                st.info(f"🕒 Prediction Time: {elapsed:.4f} seconds")

                if elapsed > 5:
                    st.warning("⚠️ High latency detected. Consider reviewing model/server.")
            else:
                st.error(f"❌ Error {response.status_code}: {response.text}")
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
