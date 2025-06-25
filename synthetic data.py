import pandas as pd
import numpy as np
import random

# Initialize
np.random.seed(42)
random.seed(42)
n_samples = 50000

# Constants
vehicle_types = ['Sedan', 'SUV', 'Hatchback', 'Truck', 'Bike', 'Cruiser', 'Convertible', 'Electric', 'Luxury', 'Mini']
vehicle_makes = ['Toyota', 'Honda', 'Ford', 'Hyundai', 'Suzuki', 'Kia', 'Tata', 'Mahindra', 'BMW', 'Audi', 'Nissan']
coverage_types = ['comprehensive', 'third-party']
damage_severity = ['minor', 'moderate', 'severe']
registration_zones = ['A', 'B', 'C']
claim_channels = ['Online', 'Agent', 'Branch']
location_types = ['urban', 'rural', 'highway']
incident_times = ['day', 'night']
genders = ['male', 'female']
yes_no = ['yes', 'no']
damage_parts = ['windshield', 'bumper', 'engine', 'side door', 'mirror', 'paintwork', 'headlight', 'roof', 'rear light', 'tire']

# IDV Ranges Based on Vehicle Make
brand_idv_ranges = {
    'Suzuki': (50000, 800000),
    'Tata': (50000, 800000),
    'Hyundai': (50000, 800000),
    'Kia': (50000, 800000),
    'Toyota': (600000, 1200000),
    'Ford': (600000, 1200000),
    'Mahindra': (600000, 1200000),
    'BMW': (1200000, 2500000),
    'Audi': (1200000, 2500000),
    'Honda': (500000, 1000000),
    'Nissan': (1000000, 1800000)
}

# Generate base structured data
vehicle_make_list = np.random.choice(vehicle_makes, n_samples)
data = {
    'claim_id': [f'CLM{100000 + i}' for i in range(n_samples)],
    'customer_id': [f'CUST{100000 + i}' for i in range(n_samples)],
    'vehicle_type': np.random.choice(vehicle_types, n_samples),
    'vehicle_make': vehicle_make_list,
    'vehicle_model': [f'MODEL-{random.randint(100,999)}' for _ in range(n_samples)],
    'vehicle_age': np.random.randint(0, 15, n_samples),
    'engine_capacity': np.random.randint(800, 3000, n_samples),
    'registration_zone': np.random.choice(registration_zones, n_samples),
    'coverage_type': np.random.choice(coverage_types, n_samples),
    'ncb_percentage': np.random.choice([0, 20, 25, 35, 45, 50], n_samples),
    'add_ons': np.random.randint(0, 4, n_samples),
    'policy_tenure': np.random.randint(1, 10, n_samples),
    'claim_duration': np.random.randint(1, 365, n_samples),
    'vehicle_ownership_years': np.random.randint(0, 15, n_samples),
    'age': np.random.randint(18, 75, n_samples),
    'gender': np.random.choice(genders, n_samples),
    'driving_experience': np.random.randint(0, 40, n_samples),
    'previous_claims': np.random.randint(0, 5, n_samples),
    'customer_tenure': np.random.randint(1, 15, n_samples),
    'claim_channel': np.random.choice(claim_channels, n_samples),
    'incident_time': np.random.choice(incident_times, n_samples),
    'claim_submission_delay': np.random.randint(0, 30, n_samples),
    'location_type': np.random.choice(location_types, n_samples),
    'reported_to_police': np.random.choice(yes_no, n_samples),
    'damage_severity': np.random.choice(damage_severity, n_samples),
    'driver_at_fault': np.random.choice(yes_no, n_samples),
    'odometer_reading': np.random.randint(5000, 200000, n_samples),
}

# Add IDV based on make
idvs = []
for make in vehicle_make_list:
    low, high = brand_idv_ranges[make]
    idvs.append(round(random.uniform(low, high), 2))
data['idv'] = idvs

# Convert to DataFrame
df = pd.DataFrame(data)

# Compute costs
df['claim_amount'] = np.round(df['idv'] * np.random.uniform(0.1, 1.2, n_samples), 2)
df['repair_cost_estimate'] = np.round(df['claim_amount'] * np.random.uniform(0.5, 1.2, n_samples), 2)

# Add claim_history
claim_history_templates = [
    "Had {} minor claim(s) in the past.",
    "No previous claims recorded.",
    "Reported {} accident(s) over {} years.",
    "Previously claimed for {} and {} damages.",
    "Frequent claims for {} related issues."
]
df['claim_history'] = [
    random.choice(claim_history_templates).format(
        random.randint(1, 3),
        random.choice(damage_parts),
        random.choice(damage_parts)
    ) if random.random() < 0.7 else "No previous claims recorded."
    for _ in range(n_samples)
]

# Add description_of_current_damage
current_damage_templates = [
    "Vehicle hit from behind, resulting in {} damage.",
    "Involved in a collision; {} and {} damaged.",
    "Scratches on the {} after parking mishap.",
    "{} broken during highway incident.",
    "{} and {} misaligned due to pothole hit.",
    "Total damage on {}, needs full replacement."
]
df['description_of_current_damage'] = [
    random.choice(current_damage_templates).format(
        random.choice(damage_parts),
        random.choice(damage_parts)
    )
    for _ in range(n_samples)
]

# Realistic fraud_flag generation based on heuristics
def compute_fraud_score(row):
    score = 0
    claim_ratio = row['claim_amount'] / row['idv']
    if claim_ratio > 0.9:
        score += 3
    elif claim_ratio > 0.7:
        score += 2
    elif claim_ratio > 0.5:
        score += 1

    repair_ratio = row['repair_cost_estimate'] / row['claim_amount']
    if repair_ratio > 1.1:
        score += 2
    elif repair_ratio > 0.9:
        score += 1

    if row['claim_submission_delay'] < 2:
        score += 2
    elif row['claim_submission_delay'] > 20:
        score += 2

    if row['previous_claims'] >= 3:
        score += 3
    elif row['previous_claims'] == 2:
        score += 1

    if row['driver_at_fault'] == 'yes':
        score += 1

    if row['location_type'] == 'urban':
        score += 1

    if row['damage_severity'] == 'minor' and claim_ratio > 0.6:
        score += 2

    if row['claim_channel'] == 'Agent':
        score += 1

    if row['vehicle_age'] < 2 and claim_ratio > 0.7:
        score += 1

    return score

df['fraud_score'] = df.apply(compute_fraud_score, axis=1)

# Threshold to determine fraud_flag - tweak to balance fraud ratio
fraud_threshold = 5
df['fraud_flag'] = (df['fraud_score'] >= fraud_threshold).astype(int)

df.drop(columns=['fraud_score'], inplace=True)

# Shuffle and save
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(df[['claim_id', 'vehicle_make', 'idv', 'claim_amount', 'fraud_flag', 'claim_history', 'description_of_current_damage']].head())

# Save to Excel
df.to_excel(r"C:\Users\Bramarambika\Downloads\Workoopolis\fraud detection\dataset1_realistic.xlsx", index=False)
