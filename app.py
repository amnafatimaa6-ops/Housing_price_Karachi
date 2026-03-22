import streamlit as st
import pandas as pd
from model import train_model

st.title("🏡 House Price Predictor (Live Model)")

# Train model ONCE (cached for performance)
@st.cache_resource
def load_model():
    return train_model()

model, location_avg = load_model()

# -------------------------
# User Inputs
# -------------------------
bedrooms = st.slider("Bedrooms", 1, 10, 3)
bathrooms = st.slider("Bathrooms", 1, 7, 3)
area = st.number_input("Area (sqft)", 300, 10000, 1500)
location = st.selectbox("Location", list(location_avg.index))

# -------------------------
# Prediction
# -------------------------
if st.button("Predict Price 💰"):

    input_data = pd.DataFrame([{
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'area sqft': area,
        'location_avg_price': location_avg[location]
    }])

    prediction = model.predict(input_data)[0]

    st.success(f"Estimated Price: PKR {int(prediction):,}")
