import streamlit as st
import pandas as pd
import plotly.graph_objects as go
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
area = st.sidebar.slider("Area (sqft)", 300, 10000, 1500)
location = st.sidebar.selectbox("Location", list(location_avg.index))
property_type = st.sidebar.selectbox("Property Type", ["House", "Flat"])
furnishing = st.sidebar.selectbox("Furnishing", ["Furnished", "Unfurnished"])

# -------------------------
# Helper: format price safely in Cr/Lakh
# -------------------------
def format_price_lakh(amount_lakh):
    if amount_lakh >= 100:  # >= 1 Cr
        crore = int(amount_lakh // 100)
        lakh = int(amount_lakh % 100)
        return f"{crore} Cr {lakh} Lakh"
    else:
        return f"{int(round(amount_lakh))} Lakh"

# -------------------------
# Prediction
# -------------------------
if st.button("Predict Price 💰"):

    # Base input
    input_data = pd.DataFrame([{
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'area sqft': area,
        'location_avg_price': location_avg[location],
        'property_type_House': int(property_type == 'House'),
        'furnishing_status_Unfurnished': int(furnishing == 'Unfurnished')
    }])

    # Predict price
    prediction = model.predict(input_data)[0]

    # Use location average to limit crazy predictions
    loc_avg_price = location_avg[location]
    min_estimate = loc_avg_price * 0.8  # 20% lower
    max_estimate = loc_avg_price * 1.2  # 20% higher

    # Convert PKR to Lakh
    prediction_lakh = prediction / 100_000
    low_price = min(min_estimate / 100_000, prediction_lakh)
    high_price = max(max_estimate / 100_000, prediction_lakh)

    # Format prices
    formatted_low = format_price_lakh(low_price)
    formatted_high = format_price_lakh(high_price)

    # Display results
    st.success(f"Estimated Price Range: {formatted_low} – {formatted_high}")
    st.info(f"⚠️ Note: This is an **average estimate** based on historical prices in {location}.")

    # Show model confidence
    st.write(f"📊 Model Confidence (R² Score): {model_r2:.2%}")

    # -------------------------
    # Plot interactive graph
    # -------------------------
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prediction_lakh,
        title={'text': "Estimated Price (Lakh)"},
        gauge={'axis': {'range': [0, max(high_price*1.2, 500)]},  # dynamic max
               'bar': {'color': "green"},
               'steps': [
                   {'range': [0, low_price], 'color': "lightgray"},
                   {'range': [low_price, high_price], 'color': "yellow"},
                   {'range': [high_price, max(high_price*1.2, 500)], 'color': "red"}]}
    ))
    st.plotly_chart(fig, use_container_width=True)
