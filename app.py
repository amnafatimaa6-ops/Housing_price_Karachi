import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("housing_model.pkl")

st.title("🏠 Karachi Housing Price Predictor")

st.write("Enter property details to predict the price.")

# User Inputs
area = st.number_input("Area (Square Feet)", min_value=100, max_value=10000, value=1200)
bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3)
bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)
location = st.selectbox(
    "Location",
    ["DHA", "Clifton", "Gulshan", "North Nazimabad", "PECHS"]
)

# Encode location (simple example)
location_map = {
    "DHA": 0,
    "Clifton": 1,
    "Gulshan": 2,
    "North Nazimabad": 3,
    "PECHS": 4
}

location_encoded = location_map[location]

# Prediction button
if st.button("Predict Price"):

    input_data = pd.DataFrame({
        "area": [area],
        "bedrooms": [bedrooms],
        "bathrooms": [bathrooms],
        "location": [location_encoded]
    })

    prediction = model.predict(input_data)

    st.success(f"Estimated House Price: PKR {prediction[0]:,.0f}")
