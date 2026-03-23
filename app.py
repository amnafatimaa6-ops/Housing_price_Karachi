import streamlit as st
import pandas as pd
import numpy as np
from model import train_model
import plotly.express as px

st.set_page_config(page_title="🏡 House Price Predictor", page_icon="🏠")
st.title("🏡 House Price Predictor (Live Model)")

# -------------------------
# Load model once
# -------------------------
@st.cache_resource
def load_model():
    return train_model()

location_ppsqft, model_r2 = load_model()

# -------------------------
# User Inputs (Sidebar)
# -------------------------
st.sidebar.header("Property Details")
bedrooms = st.sidebar.slider("Bedrooms", 1, 10, 3)
bathrooms = st.sidebar.slider("Bathrooms", 1, 7, 3)
area = st.sidebar.slider("Area (sqft)", 300, 10000, 1500)
location = st.sidebar.selectbox("Location", list(location_ppsqft.index))
property_type = st.sidebar.selectbox("Property Type", ["House", "Flat"])
furnishing = st.sidebar.selectbox("Furnishing", ["Furnished", "Unfurnished"])

# -------------------------
# Helper: format price in Cr/Lakh
# -------------------------
def format_price(amount):
    crore = amount // 10_00_000
    lakh = (amount % 10_00_000) // 100_000
    if crore > 0:
        return f"{int(crore)} Cr {int(lakh)} Lakh"
    else:
        return f"{int(lakh)} Lakh"

# -------------------------
# Prediction
# -------------------------
if st.button("Predict Price 💰"):

    # Base price = median price per sqft * area
    base_price = location_ppsqft[location] * area

    # Adjust for property type
    if property_type == "House":
        base_price *= 1.10
    else:
        base_price *= 0.92

    # Adjust for furnishing
    if furnishing == "Furnished":
        base_price *= 1.05
    else:
        base_price *= 0.97

    # Adjust for total rooms slightly
    base_price *= (1 + (bedrooms + bathrooms) * 0.01)

    # Clamp extreme values
    min_limit = location_ppsqft[location] * 500
    max_limit = location_ppsqft[location] * 8000
    base_price = max(min(base_price, max_limit), min_limit)

    # Format price
    formatted_price = format_price(int(base_price))

    # Show result
    st.success(f"Estimated Price: {formatted_price}")
    st.info(f"⚠️ Note: This price is based on average historical data in {location}. It is an estimate, not exact.")

    # Interactive price trend graph
    st.subheader("📊 Price Trend by Area")
    areas = np.linspace(300, 10000, 50)
    prices = []

    for a in areas:
        p = location_ppsqft[location] * a
        p *= 1.10 if property_type == "House" else 0.92
        p *= 1.05 if furnishing == "Furnished" else 0.97
        p *= (1 + (bedrooms + bathrooms) * 0.01)
        p = max(min(p, max_limit), min_limit)
        prices.append(p)

    fig = px.line(
        x=areas,
        y=prices,
        labels={'x': 'Area (sqft)', 'y': 'Estimated Price (PKR)'},
        title=f"Price Trend in {location}"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Model confidence
    st.write(f"📊 Model Confidence (Estimated): {model_r2:.2%}")
