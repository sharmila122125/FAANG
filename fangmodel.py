# fangmodel.py

import streamlit as st
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# ✅ Set MLflow tracking URI (MUST match training script)
mlflow.set_tracking_uri("file:///C:/Users/DELL/Desktop/Guvi/mlflow_logs")

# ✅ Load model from MLflow using updated Run ID
model_path = "runs:/ae654c391d02487e9e9e295a241534b3/best_model"
st.set_page_config(page_title="FAANG Stock Price Predictor", layout="wide")

st.sidebar.header("📦 Load Model")
try:
    model = mlflow.sklearn.load_model(model_path)
    st.success("✅ Model loaded successfully from MLflow.")
except Exception as e:
    st.error("❌ Failed to load model from MLflow.")
    st.code(str(e))
    st.stop()

# App title
st.title("📈 FAANG Stock Price Predictor")

company = st.sidebar.selectbox("Select Company", ['Apple', 'Amazon', 'Meta', 'Netflix', 'Google'])

# User inputs
st.sidebar.header("📊 Enter Stock Data")
open_price = st.sidebar.slider("Open Price", 0.0, 1000.0, 500.0)
high_price = st.sidebar.slider("High Price", 0.0, 1000.0, 550.0)
low_price = st.sidebar.slider("Low Price", 0.0, 1000.0, 480.0)
volume = st.sidebar.number_input("Volume", min_value=0.0, value=5_000_000.0, step=10000.0)

user_input = np.array([[open_price, high_price, low_price, volume]])

st.subheader("📍 Predicted Close Price")
if st.button("🔍 Predict"):
    prediction = model.predict(user_input)[0]
    st.success(f"📉 Predicted Close Price: **${prediction:.2f}**")

# Evaluation metrics
if st.checkbox("📊 Show Model Evaluation Metrics"):
    try:
        X_test = pd.read_csv("X_test.csv", index_col=0)
        y_test = pd.read_csv("y_test.csv", index_col=0).values.flatten()
        y_pred = model.predict(X_test)

        st.subheader("📌 Evaluation Metrics")
        st.write(f"**MAE:** {mean_absolute_error(y_test, y_pred):.2f}")
        st.write(f"**RMSE:** {mean_squared_error(y_test, y_pred, squared=False):.2f}")
        st.write(f"**R² Score:** {r2_score(y_test, y_pred):.2f}")

        st.subheader("📈 Actual vs Predicted")
        plt.figure(figsize=(10, 4))
        plt.plot(y_test, label="Actual", alpha=0.7)
        plt.plot(y_pred, label="Predicted", alpha=0.7)
        plt.legend()
        plt.xlabel("Sample")
        plt.ylabel("Close Price")
        st.pyplot(plt)
    except Exception as e:
        st.error("⚠️ Could not evaluate test data.")
        st.code(str(e))

st.markdown("---")
st.caption("Built with ❤️ using Streamlit + MLflow | FAANG Stock Predictor")
