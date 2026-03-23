import streamlit as st
import pandas as pd
from model import train_model
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="🏡 House Price Predictor", page_icon="🏠")

st.title("🏡 House Price Predictor")

# -------------------------
# Load model
# -------------------------
@st.cache_resource
def load_model():
    return train_model()

location_ppsqft, model_r2 = load_model()

# -------------------------
# Sidebar Inputs
# -------------------------
st.sidebar.header("Property Details")

bedrooms = st.sidebar.slider("Bedrooms", 1, 10, 3)
bathrooms = st.sidebar.slider("Bathrooms", 1, 7, 3)
area = st.sidebar.slider("Area (sqft)", 300, 10000, 1500)

location = st.sidebar.selectbox("Location", list(location_ppsqft.index))
property_type = st.sidebar.selectbox("Property Type", ["House", "Flat"])
furnishing = st.sidebar.selectbox("Furnishing", ["Furnished", "Unfurnished"])

# -------------------------
# Format price
# -------------------------
def format_price(amount):
    crore = amount // 10_00_000
    lakh = (amount % 10_00_000) // 100_000
    return f"{int(crore)} Cr {int(lakh)} Lakh" if crore > 0 else f"{int(lakh)} Lakh"

# -------------------------
# Prediction
# -------------------------
if st.button("Predict Price 💰"):

    # Base price
    base_price = location_ppsqft[location] * area

    # Adjustments
    if property_type == "House":
        base_price *= 1.25
    else:
        base_price *= 0.85

    if furnishing == "Furnished":
        base_price *= 1.10
    else:
        base_price *= 0.95

    # Rooms effect
    base_price *= (1 + (bedrooms + bathrooms) * 0.02)

    # -------------------------
    # Price Range
    # -------------------------
    variation = 0.15

    min_price = base_price * (1 - variation)
    max_price = base_price * (1 + variation)

    formatted_min = format_price(int(min_price))
    formatted_max = format_price(int(max_price))

    # -------------------------
    # Output
    # -------------------------
    st.success(f"💰 Estimated Price Range: {formatted_min} – {formatted_max}")

    st.info(
        f"📍 Based on average price per sqft in {location}. "
        "Adjusted for property type, furnishing, and size."
    )

    st.write(f"📊 Model Confidence: {model_r2:.0%}")

    st.caption("🏡 Tip: Actual price depends on exact block, condition, and market timing.")

    # -------------------------
    # Graph: Price vs Area
    # -------------------------
    st.subheader("📊 Price Trend (Area vs Price)")

    areas = np.linspace(300, 10000, 50)
    prices = []

    for a in areas:
        p = location_ppsqft[location] * a

        if property_type == "House":
            p *= 1.25
        else:
            p *= 0.85

        if furnishing == "Furnished":
            p *= 1.10
        else:
            p *= 0.95

        p *= (1 + (bedrooms + bathrooms) * 0.02)

        prices.append(p)

    fig, ax = plt.subplots()
    ax.plot(areas, prices)
    ax.set_xlabel("Area (sqft)")
    ax.set_ylabel("Estimated Price (PKR)")
    ax.set_title(f"Price Trend in {location}")

    st.pyplot(fig)
