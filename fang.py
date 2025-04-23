import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow.sklearn

# ========== APP TITLE ==========
st.set_page_config(page_title="FAANG Stock Price Predictor", layout="centered")
st.title("📈 FAANG Stock Price Predictor")

# ========== COMPANY SELECTION ==========
companies = ["Apple", "Amazon", "Meta", "Netflix", "Google"]
tickers = {
    "Apple": "AAPL",
    "Amazon": "AMZN",
    "Meta": "META",
    "Netflix": "NFLX",
    "Google": "GOOGL"
}

st.sidebar.header("Enter Stock Details")
selected_company = st.sidebar.selectbox("Company", companies)
selected_ticker = tickers[selected_company]

# ========== USER INPUTS ==========
open_price = st.sidebar.slider("Open Price", 0.0, 5000.0, 100.0)
high_price = st.sidebar.slider("High Price", 0.0, 5000.0, 120.0)
low_price = st.sidebar.slider("Low Price", 0.0, 5000.0, 90.0)
volume = st.sidebar.number_input("Volume", min_value=0, value=1000000)

user_input = np.array([[open_price, high_price, low_price, volume]])

# ========== LOAD MODEL ==========
try:
    model = mlflow.sklearn.load_model("faang_model")
except Exception as e:
    st.error("❌ Model could not be loaded. Make sure 'faang_model/' folder exists.")
    st.stop()

# ========== SCALE USER INPUT ==========
feature_mins = np.array([0, 0, 0, 0])
feature_maxs = np.array([5000, 5000, 5000, 1000000000])
scaler = MinMaxScaler()
scaler.fit(np.array([feature_mins, feature_maxs]))
scaled_input = scaler.transform(user_input)

# ========== PREDICTION ==========
if st.button("Predict"):
    predicted = model.predict(scaled_input)[0]
    st.success(f"📊 Predicted Closing Price for {selected_company} ({selected_ticker}): **${predicted:.2f}**")

# ========== MODEL EVALUATION ==========
try:
    st.divider()
    st.subheader("📈 Model Evaluation (Test Set)")

    # Load test data
    X_test = pd.read_csv("X_test.csv", index_col=0).values
    y_test = pd.read_csv("y_test.csv", index_col=0).values.ravel()

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", f"{mae:.2f}")
    col2.metric("RMSE", f"{rmse:.2f}")
    col3.metric("R² Score", f"{r2:.2f}")

    st.subheader("📊 Actual vs Predicted Prices")
    fig, ax = plt.subplots()
    ax.plot(y_test, label="Actual", linewidth=2)
    ax.plot(y_pred, label="Predicted", linestyle="--")
    ax.set_xlabel("Index")
    ax.set_ylabel("Close Price")
    ax.legend()
    st.pyplot(fig)

except FileNotFoundError:
    st.warning("📂 Test data (X_test.csv / y_test.csv) not found — evaluation skipped.")
except Exception as e:
    st.error(f"⚠️ Error during evaluation: {str(e)}")
