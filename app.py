import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder

# Load dataset
df = pd.read_csv("House_prices (1).csv")

st.title("🏠 Karachi Housing Price Predictor")
st.write("Enter property details to predict the price.")

# User Inputs
property_type = st.selectbox("Property Type", df['property type'].unique())
bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3)
bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)
area = st.number_input("Area (sqft)", min_value=100, max_value=10000, value=1200)
location = st.selectbox("Location", df['location'].unique())
furnishing = st.selectbox("Furnishing Status", df['furnishing_status'].unique())

# Prepare training data
X = df[['property type', 'bedrooms', 'bathrooms', 'area sqft', 'location', 'furnishing_status']]
y = df['price']

# One-hot encoding
encoder = OneHotEncoder(drop='first', sparse=False)
X_encoded = encoder.fit_transform(X[['property type', 'location', 'furnishing_status']])
X_final = np.concatenate([X_encoded, X[['bedrooms', 'bathrooms', 'area sqft']].values], axis=1)

# Train Random Forest
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_final, y)

# Prepare input for prediction
input_df = pd.DataFrame({
    'property type': [property_type],
    'bedrooms': [bedrooms],
    'bathrooms': [bathrooms],
    'area sqft': [area],
    'location': [location],
    'furnishing_status': [furnishing]
})

input_encoded = encoder.transform(input_df[['property type', 'location', 'furnishing_status']])
input_final = np.concatenate([input_encoded, input_df[['bedrooms', 'bathrooms', 'area sqft']].values], axis=1)

# Predict
if st.button("Predict Price"):
    prediction = model.predict(input_final)
    st.success(f"Estimated House Price: PKR {prediction[0]:,.0f}")
