import streamlit as st
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Set MLflow tracking URI
mlflow.set_tracking_uri("file:///C:/Users/DELL/Desktop/Guvi/mlflow_logs")

# Load the registered model from MLflow
registered_model_path = "models:/FAANG_Stock_Best_Model/1"

# Load the MinMaxScaler
try:
    scaler = joblib.load("scaler.pkl")
except FileNotFoundError:
    st.error("Missing 'scaler.pkl'. Please ensure it was saved during training.")
    st.stop()

st.set_page_config(page_title="FAANG Stock Price Predictor", layout="wide")

st.sidebar.header("Load Model")
try:
    model = mlflow.sklearn.load_model(registered_model_path)
    st.success("Model loaded successfully from MLflow.")
except Exception as e:
    st.error("Failed to load model from MLflow.")
    st.code(str(e))
    st.stop()

# App title
st.title("FAANG Stock Price Predictor")

# Company selection
company = st.sidebar.selectbox("Select Company", ['Apple', 'Amazon', 'Meta', 'Netflix', 'Google'])

# User input fields
st.sidebar.header("Enter Stock Data")
open_price = st.sidebar.slider("Open Price", 0.0, 1000.0, 500.0)
high_price = st.sidebar.slider("High Price", 0.0, 1000.0, 550.0)
low_price = st.sidebar.slider("Low Price", 0.0, 1000.0, 480.0)
volume = st.sidebar.number_input("Volume", min_value=0.0, value=5_000_000.0, step=10000.0)

# Combine user inputs
user_input = np.array([[open_price, high_price, low_price, volume]])

st.subheader(f"Predicted Close Price for {company}")

if st.button("Predict"):
    try:
        scaled_input = scaler.transform(user_input)
        prediction = model.predict(scaled_input)[0]
        st.success(f"Predicted Close Price: ${prediction:.2f}")
    except Exception as e:
        st.error("Prediction failed.")
        st.code(str(e))

# Evaluation section
if st.checkbox("Show Model Evaluation Metrics"):
    try:
        X_test = pd.read_csv("X_test.csv")
        y_test = pd.read_csv("y_test.csv")["Close"].values

        st.markdown(f"Test data shape: {X_test.shape}")
        expected_features = ['Open', 'High', 'Low', 'Volume']
        actual_features = X_test.columns.tolist()

        if actual_features != expected_features:
            st.warning(f"Column mismatch. Expected: {expected_features} but got: {actual_features}")
            st.stop()

        # Scale test features
        X_test_scaled = scaler.transform(X_test)
        y_pred = model.predict(X_test_scaled)

        st.subheader("Evaluation Metrics")
        st.write(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
        st.write(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False):.2f}")
        st.write(f"R² Score: {r2_score(y_test, y_pred):.2f}")

        st.subheader("Actual vs Predicted Close Price")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(y_test, label="Actual", alpha=0.7)
        ax.plot(y_pred, label="Predicted", alpha=0.7)
        ax.set_xlabel("Sample")
        ax.set_ylabel("Close Price")
        ax.legend()
        st.pyplot(fig)

    except Exception as e:
        st.error("Could not evaluate or plot test data.")
        st.code(str(e))

st.markdown("---")
st.caption("Built using Streamlit and MLflow | FAANG Stock Predictor")
