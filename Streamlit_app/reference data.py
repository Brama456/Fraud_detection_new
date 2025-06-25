import pandas as pd

# Load your original dataset (before training)
df = pd.read_excel(r"C:\Users\Bramarambika\Downloads\Workoopolis\fd deployment unwanted\dataset1_realistic.xlsx")

# Apply same feature engineering
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

df = add_engineered_features(df)

# Save only necessary columns (the ones used by model)
model_input = df[[
    'vehicle_type', 'vehicle_make', 'vehicle_age', 'engine_capacity', 'registration_zone',
    'coverage_type', 'ncb_percentage', 'add_ons', 'policy_tenure', 'claim_duration',
    'vehicle_ownership_years', 'age', 'gender', 'driving_experience', 'previous_claims',
    'customer_tenure', 'claim_channel', 'incident_time', 'claim_submission_delay',
    'location_type', 'reported_to_police', 'damage_severity', 'driver_at_fault',
    'odometer_reading', 'idv', 'claim_amount', 'repair_cost_estimate',
    'experience_age_ratio', 'claim_to_idv_ratio', 'repair_to_claim_ratio',
    'vehicle_value_per_year', 'claim_amount_per_year', 'previous_claim_ratio',
    'claim_delay_weighted', 'severity_score', 'coverage_risk_score', 'channel_risk_score'
]]

# Save first 100 rows as reference
model_input.sample(100, random_state=42).to_csv("reference_data.csv", index=False)
