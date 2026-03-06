# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
from housing_model import load_and_preprocess, train_models

st.set_page_config(page_title="Karachi Housing Price Predictor", layout="wide")
st.title("🏠 Karachi Housing Price Predictor")

# Load CSV automatically from repo
try:
    df_encoded, scaler, df = load_and_preprocess("House_prices (1).csv")
    st.success("✅ Dataset loaded & preprocessed!")
except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    st.stop()

# Train models
with st.spinner("Training models..."):
    lr_model, dt_model, rf_model, feature_cols, accuracies = train_models(df_encoded)
acc_lr, acc_dt, acc_rf = accuracies

# Show accuracy
st.subheader("Model Accuracy (R²)")
st.write(f"**Linear Regression:** {acc_lr:.2f}")
st.write(f"**Decision Tree:** {acc_dt:.2f}")
st.write(f"**Random Forest:** {acc_rf:.2f}")

# User input
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
    """Convert raw number to Lakh/Cr format."""
    if num >= 10_000_000:
        return f"{num/10_000_000:.2f} Cr"
    elif num >= 100_000:
        return f"{num/100_000:.2f} Lakh"
    else:
        return f"{num:,.0f}"  # below 1 Lakh, show raw

if st.button("Predict Price"):
    input_df = pd.DataFrame({
        'bedrooms':[bedrooms],
        'bathrooms':[bathrooms],
        'area sqft':[area],
        'total_rooms':[bedrooms+bathrooms],
        'price_per_sqft':[0],
        'property type_House':[1 if prop_type=="House" else 0],
        'property type_Apartment':[1 if prop_type=="Apartment" else 0],
        'furnishing_status_Furnished':[1 if furnishing=="Furnished" else 0],
        'furnishing_status_Unfurnished':[1 if furnishing=="Unfurnished" else 0],
        **{f'location_{loc}': 1 if loc==location else 0 for loc in df['location'].unique()}
    })

    # Scale numeric
    num_features = ['bedrooms','bathrooms','area sqft','total_rooms','price_per_sqft']
    input_df[num_features] = scaler.transform(input_df[num_features])

    # Align columns
    input_df = input_df.reindex(columns=feature_cols, fill_value=0)

    # Predict
    pred_lr = lr_model.predict(input_df)[0]
    pred_dt = dt_model.predict(input_df)[0]
    pred_rf = rf_model.predict(input_df)[0]

    st.subheader("Predicted Price")
    st.write(f"**Linear Regression:** {format_price(pred_lr)}")
    st.write(f"**Decision Tree:** {format_price(pred_dt)}")
    st.write(f"**Random Forest:** {format_price(pred_rf)}")

    avg_pred = (pred_lr + pred_dt + pred_rf) / 3
    st.write(f"**Average Prediction:** {format_price(avg_pred)}")
