# train_model.py - High-Recall XGBoost Fraud Detection Model

import pandas as pd
import numpy as np
import joblib
import os
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score, f1_score, confusion_matrix

# --- Configuration ---
DATA_PATH = 'data/creditcard.csv' 
MODEL_DIR = 'models' 
RANDOM_STATE = 42

# --- 1. Load Data ---
try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    print(f"ERROR: Data file not found at {DATA_PATH}. Please ensure creditcard.csv is in the /data folder.")
    exit()

# Features (V1-V28, Amount) and Target (0=Non-Fraud, 1=Fraud)
# Drop 'Time' as it's not a direct pattern feature
X = df.drop(['Time', 'Class'], axis=1) 
y = df['Class'] 

# --- 2. Preprocessing & Splitting ---

# 80/20 split, using stratify to keep the rare fraud cases balanced in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=RANDOM_STATE, 
    stratify=y
)

# Scale Features: Essential for ensuring model convergence and performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# --- 3. Imbalance Handling & Model Definition ---

# 3a. Calculate the Class Weight (CRITICAL FOR HIGH RECALL)
# This weight tells XGBoost how much more important the fraud class (1) is than non-fraud (0).
count_non_fraud = y_train.value_counts()[0]
count_fraud = y_train.value_counts()[1]
scale_pos_weight_value = count_non_fraud / count_fraud

print("-" * 50)
print(f"Total Transactions: {len(df)}")
print(f"Non-Fraud in Train: {count_non_fraud} | Fraud in Train: {count_fraud}")
print(f"XGBoost Class Weight (scale_pos_weight): {scale_pos_weight_value:.2f}")
print("-" * 50)


# 3b. Define XGBoost Model (The specialized, high-performance algorithm)
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    n_estimators=100,
    learning_rate=0.1,
    random_state=RANDOM_STATE,
    # APPLY THE WEIGHT: This is your resume highlight for handling imbalance
    scale_pos_weight=scale_pos_weight_value,  
    n_jobs=-1,
    use_label_encoder=False,
    eval_metric='logloss' 
)

# --- 4. Training ---

print("Starting XGBoost model training...")
# Train the model on the scaled features
xgb_model.fit(X_train_scaled, y_train)
print("Training complete.")


# --- 5. Evaluation (Focus on Risk Metrics) ---

y_pred = xgb_model.predict(X_test_scaled)

test_recall = recall_score(y_test, y_pred)
test_f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("-" * 50)
print("--- Model Performance on Test Set (Risk-Focused) ---")
print(f"XGBoost Test Recall (Fraud Coverage): {test_recall:.4f} (Target >= 0.90)")
print(f"XGBoost Test F1-Score: {test_f1:.4f}")
print("Confusion Matrix (Rows=Actual, Columns=Predicted):")
print(conf_matrix)
print("-" * 50)


# --- 6. Persistence (Save Model and Scaler) ---

# 6a. Ensure the /models directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# 6b. Save the two critical binary files using joblib
# NOTE: The model file name must match what the FastAPI app loads later.
joblib.dump(xgb_model, os.path.join(MODEL_DIR, 'credit_fraud.pkl')) 
joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))

print(f"Model and Scaler successfully saved to: {MODEL_DIR}/")
print("Stage 1 complete and assets ready for deployment.")