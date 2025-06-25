import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_excel(r"C:\Users\Bramarambika\Downloads\Workoopolis\dataset1_realistic.xlsx")
df = df.drop(columns=['claim_id', 'customer_id'], errors='ignore')

severity_map = {'minor': 1, 'moderate': 2, 'severe': 3}
coverage_map = {'comprehensive': 0, 'third-party': 1}
channel_map = {'Online': 1, 'Agent': 0.5, 'Branch': 0.3}

df['claim_to_idv_ratio'] = df['claim_amount'] / (df['idv'] + 1)
df['repair_to_claim_ratio'] = df['repair_cost_estimate'] / (df['claim_amount'] + 1)
df['severity_score'] = df['damage_severity'].map(severity_map)
df['claim_delay_weighted'] = df['claim_submission_delay'] * df['severity_score']
df['claim_amount_per_year'] = df['claim_amount'] / (df['policy_tenure'] + 1)
df['vehicle_value_per_year'] = df['idv'] / (df['vehicle_age'] + 1)
df['experience_age_ratio'] = df['driving_experience'] / (df['age'] + 1)
df['previous_claim_ratio'] = df['previous_claims'] / (df['customer_tenure'] + 1)
df['coverage_risk_score'] = df['coverage_type'].map(coverage_map)
df['channel_risk_score'] = df['claim_channel'].map(channel_map)

X = df.drop("fraud_flag", axis=1)
y = df["fraud_flag"]

text_cols = ['claim_history', 'description_of_current_damage']
categorical_cols = X.select_dtypes(include=['object']).drop(columns=text_cols).columns.tolist()
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
numeric_cols = [col for col in numeric_cols if col not in text_cols]

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

text_transformer = Pipeline(steps=[
    ('vectorizer', TfidfVectorizer(max_features=50))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols),
        ('text1', text_transformer, 'claim_history'),
        ('text2', text_transformer, 'description_of_current_damage')
    ],
    remainder='drop'
)

clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.08,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    ))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(" Classification Report:\n", classification_report(y_test, y_pred))
print(" Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

joblib.dump(clf, "xgb_fraud_pipeline.joblib")
print(" Model saved.")
