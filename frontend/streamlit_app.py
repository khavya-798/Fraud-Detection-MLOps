import streamlit as st
import pandas as pd
import joblib

# --- MODEL LOADING ---
@st.cache_resource
def load_model_assets():
    """Loads the pre-trained model and scaler from file."""
    try:
        model = joblib.load('models/credit_fraud.pkl')
        scaler = joblib.load('models/scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        st.error("Model or scaler file not found. Make sure 'models/credit_fraud.pkl' and 'models/scaler.pkl' exist.")
        return None, None

MODEL, SCALER = load_model_assets()


# --- PREDICTION FUNCTION ---
def predict(input_data):
    """
    Takes a dictionary of features, scales them, and returns a prediction.
    """
    if MODEL is None or SCALER is None:
        return None, "Model not loaded"

    try:
        input_df = pd.DataFrame([input_data])
        features_to_scale = [col for col in input_df.columns if col != 'Time']
        scaled_features = SCALER.transform(input_df[features_to_scale])
        prediction_val = MODEL.predict(scaled_features)
        return prediction_val, "Prediction successful"
    except Exception as e:
        return None, f"Error during prediction: {e}"

# --- USER INTERFACE ---
st.title("Credit Card Fraud Detection App")

# --- FIX: USE THE CORRECT PATH TO THE IMAGE ---
st.image("frontend/image.png")

# --- FIX: Updated "About" section to remove dead links and irrelevant info ---
st.markdown("""
## About
This application uses a pre-trained XGBoost model to predict if a credit card transaction is fraudulent. 

The model was trained on an anonymized dataset from Kaggle. This user interface allows you to input values for all the features to get a real-time prediction.

This project demonstrates an end-to-end MLOps workflow, from model training to deployment as an interactive web application.
""")


st.sidebar.header('Transaction Features')
amount_input = st.sidebar.number_input("Transaction Amount ($)", min_value=0.0, value=100.0)

st.sidebar.markdown("---")
st.sidebar.markdown("**Anonymized Features (V1-V28)**")
v_inputs = {}
for i in range(1, 29):
    feature_name = f'V{i}'
    v_inputs[feature_name] = st.sidebar.slider(f"{feature_name}", -50.0, 50.0, 0.0, step=0.1)

# --- MAIN LOGIC ---
if st.button("Get Prediction"):
    if MODEL is not None:
        values = {"Time": 40000.0, "Amount": float(amount_input)}
        values.update(v_inputs)

        st.write("### Input Details:")
        st.json(values)

        prediction, message = predict(values)

        if prediction is not None:
            is_fraud = prediction[0]
            prediction_text = "Fraudulent" if is_fraud == 1 else "Not Fraudulent"
            st.write(f"## Prediction Result: **{prediction_text}**")
        else:
            st.error(message)

