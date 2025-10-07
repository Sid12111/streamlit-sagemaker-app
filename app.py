import streamlit as st
import pandas as pd
from utils.sagemaker_client import invoke_sagemaker_endpoint

st.set_page_config(page_title="SageMaker Prediction App", layout="centered")
st.title("AWS SageMaker Prediction App")

# Load sample data
st.subheader("Sample Input Data")
data = pd.read_csv("sample_data.csv")
st.dataframe(data)

# Manual input
st.subheader("Manual Input for Prediction")
feature1 = st.number_input("Feature 1", value=0)
feature2 = st.number_input("Feature 2", value=0)

if st.button("Predict"):
    payload = {"feature1": feature1, "feature2": feature2}
    prediction = invoke_sagemaker_endpoint("YOUR_ENDPOINT_NAME", payload)
    st.success(f"Prediction: {prediction}")

# Batch predictions
st.subheader("Batch Predictions")
if st.button("Predict for Sample Data"):
    predictions = []
    for _, row in data.iterrows():
        payload = {"feature1": row["feature1"], "feature2": row["feature2"]}
        pred = invoke_sagemaker_endpoint("YOUR_ENDPOINT_NAME", payload)
        predictions.append(pred)
    data["Prediction"] = predictions
    st.dataframe(data)
