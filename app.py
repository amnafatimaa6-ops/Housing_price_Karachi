import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load trained models and scaler
with open("trained_models.pkl", "rb") as f:
    data = pickle.load(f)

lr_model, features = data["linear"]
scaler = data["scaler"]

# User input
st.title("Karachi House Price Prediction")

property_type = st.selectbox("Property Type", ["House", "Apartment", "Villa"])
bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3)
bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)
area_sqft = st.number_input("Area (sqft)", min_value=100, max_value=10000, value=1200)
location = st.selectbox("Location", ["Bahria Town", "DHA City", "Clifton"])
furnishing_status = st.selectbox("Furnishing Status", ["Furnished", "Unfurnished"])

# Build input DataFrame
input_df = pd.DataFrame({
    "bedrooms": [bedrooms],
    "bathrooms": [bathrooms],
    "area sqft": [area_sqft],
    "total_rooms": [bedrooms + bathrooms],
    "price_per_sqft": [0],  # placeholder
    "log_price": [0],       # placeholder
    "log_area": [np.log1p(area_sqft)]
})

# One-hot encoding manually to match model training
for col in features:
    if col not in input_df.columns:
        input_df[col] = 0

# Set the proper dummy columns
property_col = f"property type_{property_type}"
location_col = f"location_{location}"
furnishing_col = f"furnishing_status_{furnishing_status}"

for c in [property_col, location_col, furnishing_col]:
    if c in input_df.columns:
        input_df[c] = 1

# Standardize numeric features
num_features = ['bedrooms', 'bathrooms', 'area sqft', 'price_per_sqft', 'total_rooms', 'log_price', 'log_area']
input_df[num_features] = scaler.transform(input_df[num_features])

# Predict
predicted_price = lr_model.predict(input_df[features])[0]
st.success(f"Predicted Price: PKR {predicted_price:,.0f}")
