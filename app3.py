import streamlit as st
import joblib
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta

# Load the trained ARIMA model
@st.cache_resource()
def load_model():
    return joblib.load("time_series_model_fixed.pkl")

# Function to make predictions
def predict(model, periods, start_datetime):
    forecast = model.forecast(steps=periods)
    
    # Apply smoothing to prevent sharp drops
    smoothed_forecast = forecast.ewm(span=5, adjust=False).mean()
    
    # Generate date range based on selected start datetime
    forecast_index = pd.date_range(start=start_datetime, periods=periods, freq='D')
    return pd.DataFrame({"Date": forecast_index, "Predicted Price": smoothed_forecast})

# Decision Support System: Buy/Sell/Hold Recommendation
def get_recommendation(forecast):
    last_price = forecast.iloc[-2]["Predicted Price"]  # Previous forecasted price
    next_price = forecast.iloc[-1]["Predicted Price"]  # Next predicted price
    percentage_change = ((next_price - last_price) / last_price) * 100
    
    if percentage_change > 2:
        return "📈 BUY Recommendation"
    elif percentage_change < -2:
        return "📉 SELL Recommendation"
    else:
        return "🤔 HOLD Recommendation"

# Streamlit UI
st.set_page_config(page_title="NSE Stock Prediction", layout="wide")
st.title("📈 Nigerian Stock Exchange Price Prediction")

st.sidebar.header("User Input")

# 📅 Date Picker for Start Date
start_date = st.sidebar.date_input("Select Start Date", datetime.today())

# ⏰ Time Picker for Start Time
start_time = st.sidebar.time_input("Select Start Time", datetime.now().time())

# Combine selected date and time
start_datetime = datetime.combine(start_date, start_time)

# 📊 User Inputs
periods = st.sidebar.slider("Prediction Period (days)", min_value=1, max_value=365, value=30)
manual_price = st.sidebar.number_input("Manually Enter a Stock Price", min_value=0.0, format="%.2f")

# Load Model
model = load_model()
st.success("✅ Model loaded successfully!")

# Make Prediction
forecast_df = predict(model, periods, start_datetime)

# Manual Prediction Comparison
def manual_recommendation(manual_price, forecast):
    next_price = forecast.iloc[-1]["Predicted Price"]  # Next predicted price
    percentage_change = ((next_price - manual_price) / manual_price) * 100

    if next_price > manual_price and percentage_change > 2:
        return "📈 BUY (Price Expected to Rise)"
    elif next_price < manual_price and percentage_change < -2:
        return "📉 SELL (Price Expected to Drop)"
    else:
        return "🤔 HOLD (Minimal Change Expected)"

decision = manual_recommendation(manual_price, forecast_df)

# Display Forecast Results
st.subheader("📊 Forecasted Prices")
st.dataframe(forecast_df.style.format({"Predicted Price": "{:.2f}"}))

# Show Recommendation
st.subheader("💡 Investment Decision Support")
st.markdown(f"### {decision}")

# Plot Forecasted Prices
fig = px.line(forecast_df, x="Date", y="Predicted Price", title="Stock Price Forecast",
              labels={"Date": "Date", "Predicted Price": "Price (₦)"}, markers=True)
st.plotly_chart(fig)
