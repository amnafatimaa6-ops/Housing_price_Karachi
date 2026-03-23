import streamlit as st
import pandas as pd
from model import train_model
import plotly.express as px
import plotly.graph_objects as go

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
# Helper: format price in Cr/Lakh
# -------------------------
def format_price(amount):
    crore = amount // 10_000_00
    lakh = (amount % 10_000_00) // 100_000
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
        'location_avg_price': location_avg[location],
        'property_type_House': int(property_type == 'House'),
        'furnishing_status_Unfurnished': int(furnishing == 'Unfurnished')
    }])

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Use ±10% range for estimate
    low_price = prediction * 0.9
    high_price = prediction * 1.1

    formatted_pred = format_price(int(prediction))
    formatted_low = format_price(int(low_price))
    formatted_high = format_price(int(high_price))

    st.success(f"Estimated Price Range: {formatted_low} – {formatted_high}")
    st.info(f"⚠️ Note: This is an **average estimate** based on historical data for {location}.")

    # Model confidence
    st.write(f"📊 Model Confidence (R² Score): {model_r2:.2%}")

    # -------------------------
    # Plot: Location Average Prices
    # -------------------------
    loc_df = pd.DataFrame({
        'Location': location_avg.index,
        'Average_Price': location_avg.values
    })

    fig = px.bar(
        loc_df,
        x='Location',
        y='Average_Price',
        title='🏘️ Average Property Prices by Location',
        labels={'Average_Price': 'Price (PKR)', 'Location': 'Location'},
        hover_data={'Average_Price': ':,.0f'},
        color='Average_Price',
        color_continuous_scale='Viridis'
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

    # -------------------------
    # Plot: Predicted Price Range Gauge
    # -------------------------
    fig_range = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = prediction,
        title = {'text': "💰 Your Property Estimate"},
        gauge = {
            'axis': {'range': [low_price, high_price]},
            'bar': {'color': "green"},
            'steps' : [
                {'range': [low_price, prediction], 'color': "lightgreen"},
                {'range': [prediction, high_price], 'color': "lightcoral"}]
        }
    ))
    st.plotly_chart(fig_range, use_container_width=True)
