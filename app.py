import streamlit as st
import pandas as pd
from model import train_model

st.set_page_config(page_title="🏡 House Price Predictor", page_icon="🏠")

st.title("🏡 House Price Predictor (Live Model)")

# -------------------------
# Load model once
# -------------------------
@st.cache_resource
def load_model():
    return train_model()

model, location_avg, model_r2 = load_model()

# -------------------------
# User Inputs
# -------------------------
st.sidebar.header("Property Details")
bedrooms = st.sidebar.slider("Bedrooms", 1, 10, 3)
bathrooms = st.sidebar.slider("Bathrooms", 1, 7, 3)
area = st.sidebar.number_input("Area (sqft)", 300, 10000, 1500)
location = st.sidebar.selectbox("Location", list(location_avg.index))
property_type = st.sidebar.selectbox("Property Type", ["House", "Flat"])
furnishing = st.sidebar.selectbox("Furnishing", ["Furnished", "Unfurnished"])

# -------------------------
# Prediction
# -------------------------
if st.button("Predict Price 💰"):

    # Prepare input data
    input_data = pd.DataFrame([{
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'area sqft': area,
        'location_avg_price': location_avg[location],
        'property_type_House': int(property_type == 'House'),
        'furnishing_status_Unfurnished': int(furnishing == 'Unfurnished')
    }])

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Format price in PKR
    formatted_price = f"PKR {int(prediction):,}"

    # Show results
    st.success(f"Estimated Price: {formatted_price}")
    st.info(f"⚠️ Note: This price is based on average historical data in {location}.")

    # Model confidence
    st.write(f"📊 Model Confidence (R² Score): {model_r2:.2%}")
