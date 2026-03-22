import streamlit as st
import pandas as pd
from model import train_model

st.title("🏡 House Price Predictor (Live Model)")

# -------------------------
# 💰 Price formatter
# -------------------------
def format_price_pkr(price):
    crore = price // 10000000
    lakh = (price % 10000000) // 100000
    if crore > 0:
        return f"{int(crore)} crore {int(lakh)} lakh"
    else:
        return f"{int(lakh)} lakh"

# -------------------------
# Load Model (cached)
# -------------------------
@st.cache_resource
def load_model():
    return train_model()

model, location_avg, model_r2 = load_model()  # we will pass r2 from model.py

# -------------------------
# User Inputs
# -------------------------
bedrooms = st.slider("Bedrooms", 1, 10, 3)
bathrooms = st.slider("Bathrooms", 1, 7, 3)
area = st.number_input("Area (sqft)", 300, 10000, 1500)
location = st.selectbox("Location", list(location_avg.index))
property_type = st.selectbox("Property Type", ["House", "Flat"])
furnishing_status = st.selectbox("Furnishing Status", ["Furnished", "Unfurnished"])

# -------------------------
# Prediction
# -------------------------
if st.button("Predict Price 💰"):

    # Add average price adjustment based on location
    input_data = pd.DataFrame([{
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'area sqft': area,
        'location_avg_price': location_avg[location],
        'property_type_House': 1 if property_type == "House" else 0,
        'furnishing_status_Unfurnished': 1 if furnishing_status == "Unfurnished" else 0
    }])

    prediction = model.predict(input_data)[0]

    formatted_price = format_price_pkr(prediction)

    st.success(f"Estimated Price: {formatted_price} 💰")

    # Show additional info
    st.info(f"⚡ Model Confidence (R²): {model_r2*100:.1f}%")
    st.info("📌 Note: This prediction is based on **average prices per location** and typical property types. Actual prices may vary.")
