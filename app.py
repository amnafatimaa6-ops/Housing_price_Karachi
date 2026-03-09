# 1️⃣ Imports
import streamlit as st
import pandas as pd
from housing_model import load_and_preprocess, train_models
from sklearn.preprocessing import PolynomialFeatures

# 2️⃣ Page setup
st.set_page_config(page_title="Karachi Housing Price Predictor", layout="wide")
st.title("🏠 Karachi Housing Price Predictor")

# 3️⃣ Load & preprocess
df_encoded, scaler, df = load_and_preprocess("House_prices (1).csv")

# 4️⃣ Train models
lr_model, dt_model, rf_model, gb_model, feature_cols, accuracies = train_models(df_encoded)

# 5️⃣ Display model accuracy
acc_lr, acc_dt, acc_rf, acc_gb = accuracies
st.subheader("Model Accuracy (R²)")
st.write(f"**Linear Regression:** {acc_lr:.2f}")
st.write(f"**Decision Tree:** {acc_dt:.2f}")
st.write(f"**Random Forest:** {acc_rf:.2f}")
st.write(f"**Gradient Boosting:** {acc_gb:.2f}")

# 6️⃣ User input
st.header("Enter Property Details")
col1, col2, col3, col4 = st.columns(4)
with col1:
    prop_type = st.selectbox("Property Type", df['property type'].unique())
with col2:
    bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3)
with col3:
    bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)
with col4:
    area = st.number_input("Area (sqft)", min_value=100, value=1200)

col5, col6 = st.columns(2)
with col5:
    location = st.selectbox("Location", df['location'].unique())
with col6:
    furnishing = st.selectbox("Furnishing Status", df['furnishing_status'].unique())

def format_price(num):
    if num >= 1e7:
        return f"{num/1e7:.2f} Cr"
    elif num >= 1e5:
        return f"{num/1e5:.2f} Lakh"
    else:
        return f"{num:,.0f}"

# 7️⃣ Predict button must be **after** all the above code
if st.button("Predict Price"):
    # Code snippet I sent earlier goes here
    ...
