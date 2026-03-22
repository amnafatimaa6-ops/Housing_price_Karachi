import streamlit as st
import pandas as pd
from model import train_model

st.set_page_config(page_title="🏡 House Price Predictor", page_icon="🏠")
st.title("🏡 House Price Predictor (Live Model)")

# -------------------------
# Load model once safely
# -------------------------
@st.cache_resource
def load_model_safe():
    try:
        return train_model()
    except FileNotFoundError:
        st.error("House_prices.csv not found! Please upload it in the correct folder.")
        return None, None, None
    except KeyError as e:
        st.error(f"Data is missing a required column: {e}")
        return None, None, None

model, location_avg, model_r2 = load_model_safe()

if model is None:
    st.stop()  # Stop app if model failed to load

# -------------------------
# User Inputs
# -------------------------
st.sidebar.header("Property Details")
bedrooms = st.sidebar.slider("Bedrooms", 1, 10, 3)
bathrooms = st.sidebar.slider("Bathrooms", 1, 7, 3)
area = st.sidebar.slider("Area (sqft)", 300, 10000, 1500)
location = st.sidebar.selectbox("Location", list(location_avg.index))
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

    # Prepare input data
    input_data = pd.DataFrame([{
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'area sqft': area,
        'property_type_House': int(property_type == 'House'),
        'furnishing_status_Unfurnished': int(furnishing == 'Unfurnished')
    }])

    # Predict relative price
    pred_rel = model.predict(input_data)[0]

    # Multiply by location average
    prediction = pred_rel * location_avg[location]

    # Format price
    formatted_price = format_price(int(prediction))

    # Show results
    st.success(f"Estimated Price: {formatted_price}")
    st.info(f"⚠️ Note: This price is based on average historical data in {location}. Houses are generally more expensive than flats.")

    # Model confidence
    st.write(f"📊 Model Confidence (R² Score): {model_r2:.2%}")
