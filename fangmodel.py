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
registered_model_path = "models:/FAANG_Random_Forest_Model/11"

# Load scaler
try:
    scaler = joblib.load("scaler.pkl")
except FileNotFoundError:
    st.error("Missing 'scaler.pkl'. Please ensure it was saved during training.")
    st.stop()

# Load model
try:
    model = mlflow.sklearn.load_model(registered_model_path)
except Exception as e:
    st.error("Model load failed.")
    st.code(str(e))
    st.stop()

# App Layout
st.set_page_config(page_title="FAANG Stock Price Predictor", layout="wide")
st.title(" FAANG Stock Price Predictor (Random Forest)")

# Company input
companies = ['Amazon', 'Apple', 'Facebook', 'Google', 'Netflix']
company_columns = [f"Company_{c}" for c in companies]
company = st.sidebar.selectbox("Select Company", companies)

# Feature inputs
st.sidebar.header("Enter Stock Details")
open_price = st.sidebar.slider("Open Price", 0.0, 1000.0, 500.0)
high_price = st.sidebar.slider("High Price", 0.0, 1000.0, 550.0)
low_price = st.sidebar.slider("Low Price", 0.0, 1000.0, 480.0)
volume = st.sidebar.number_input("Volume", min_value=0.0, value=5000000.0, step=10000.0)
eps = st.sidebar.slider("EPS", 0.0, 100.0, 5.0)
target_price = st.sidebar.slider("Target Price", 0.0, 1000.0, 500.0)

# One-hot encode selected company
company_vector = [1 if col == f"Company_{company}" else 0 for col in company_columns]

# Combine and scale features
numeric_input = [open_price, high_price, low_price, volume, eps, target_price]
try:
    scaled_numeric = scaler.transform([numeric_input])[0]
except Exception as e:
    st.error("Feature scaling failed. Ensure scaler was trained on 6 features.")
    st.code(str(e))
    st.stop()

final_input = np.array([scaled_numeric.tolist() + company_vector])

# Prediction
st.subheader(f"Predicted Close Price for {company}")
if st.button("Predict"):
    try:
        prediction = model.predict(final_input)[0]
        st.success(f"Predicted Close Price: ${prediction:.2f}")

        # Show as downloadable DataFrame
        prediction_df = pd.DataFrame({
            "Company": [company],
            "Open": [open_price],
            "High": [high_price],
            "Low": [low_price],
            "Volume": [volume],
            "EPS": [eps],
            "Target Price": [target_price],
            "Predicted Close Price": [round(prediction, 2)]
        })

        st.dataframe(prediction_df)
        csv = prediction_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Prediction", data=csv, file_name="prediction.csv", mime='text/csv')

    except Exception as e:
        st.error("Prediction failed.")
        st.code(str(e))

# Evaluation Metrics
if st.checkbox(" Show Model Evaluation Metrics"):
    try:
        X_test = pd.read_csv("X_test.csv")
        y_test = pd.read_csv("y_test.csv")["Close"].values

        expected_numeric = ['Open', 'High', 'Low', 'Volume', 'EPS', 'Target Price']
        expected_columns = expected_numeric + company_columns

        if list(X_test.columns) != expected_columns:
            st.warning("Mismatch in test feature columns. Retrain model or check test file.")
            st.text(f"Expected: {expected_columns}")
            st.text(f"Found: {list(X_test.columns)}")
            st.stop()

        #  Skip re-scaling (already scaled in training)
        y_pred = model.predict(X_test)

        st.subheader("Evaluation Metrics")
        st.write(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
        st.write(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False):.2f}")
        st.write(f"R² Score: {r2_score(y_test, y_pred):.2f}")

        # Plot actual vs predicted
        st.subheader("📈 Actual vs Predicted Close Price")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(y_test, label="Actual", alpha=0.7)
        ax.plot(y_pred, label="Predicted", alpha=0.7)
        ax.set_xlabel("Sample")
        ax.set_ylabel("Close Price")
        ax.legend()
        st.pyplot(fig)

    except Exception as e:
        st.error("Evaluation failed.")
        st.code(str(e))

st.markdown("---")
st.caption("Built with Streamlit and MLflow | FAANG Random Forest Regressor")
