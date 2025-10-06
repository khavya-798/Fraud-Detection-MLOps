import streamlit as st
import requests

st.title("Credit Card Fraud Detection App")
st.image("image.png")

st.markdown("""
## About
This application predicts if a transaction is fraudulent based on a machine learning model.
You can adjust all of the model's input features in the sidebar to see the prediction.
""")

# --- Sidebar Inputs ---
st.sidebar.header('Transaction Features')

# --- Main Features ---
amount_input = st.sidebar.number_input("Transaction Amount ($)", min_value=0.0, value=100.0)

# --- Anonymized V-Features ---
st.sidebar.markdown("---")
st.sidebar.markdown("**Anonymized Features (V1-V28)**")

v_inputs = {}
# Create a slider for each of the V1-V28 features
for i in range(1, 29):
    feature_name = f'V{i}'
    v_inputs[feature_name] = st.sidebar.slider(f"{feature_name}", -50.0, 50.0, 0.0, step=0.1)

# --- Main Page Logic ---
if st.button("Get Prediction"):

    # Create the payload dictionary
    values = {
        "Time": 40000.0,  # Using a fixed placeholder for Time
        "Amount": float(amount_input)
    }
    values.update(v_inputs)

    st.write("### Input Details Sent to Model:")
    st.json(values)

    # --- API Call ---
    try:
        res = requests.post("http://backend:8000/predict", json=values)

        if res.status_code == 200:
            prediction_result = res.json()
            
            # --- FINAL FIX ---
            # 1. Use the correct key 'is_fraud' to get the prediction value (0 or 1).
            is_fraud = prediction_result.get('is_fraud')

            # 2. Convert the numeric result (0 or 1) to a user-friendly string.
            if is_fraud == 1:
                prediction_text = "Fraudulent"
            elif is_fraud == 0:
                prediction_text = "Not Fraudulent"
            else:
                prediction_text = "Unknown" # Fallback for unexpected values
            
            st.write(f"## Prediction Result: **{prediction_text}**")
            
        else:
            st.error(f"Backend returned an error. Status Code: {res.status_code}")
            st.json(res.json())

    except requests.exceptions.RequestException as e:
        st.error(f"Could not connect to the backend service. Error: {e}")

