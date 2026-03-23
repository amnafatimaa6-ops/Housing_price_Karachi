import streamlit as st
import pandas as pd
from model import train_model

st.set_page_config(page_title="🏡 House Price Predictor", page_icon="🏠")

st.title("🏡 House Price Predictor (Live Model)")

# -------------------------
# Load model safely
# -------------------------
@st.cache_resource
def load_model():
    try:
        return train_model()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

model, location_avg, model_r2 = load_model()

if model is None:
    st.stop()

# -------------------------
# Sidebar Inputs
# -------------------------
st.sidebar.header("Property Details")

bedrooms = st.sidebar.slider("Bedrooms", 1, 10, 3)
bathrooms = st.sidebar.slider("Bathrooms", 1, 7, 3)
area = st.sidebar.slider("Area (sqft)", 300, 10000, 1500)

location = st.sidebar.selectbox("Location", list(location_avg.index))
property_type = st.sidebar.selectbox("Property Type", ["House", "Flat"])
furnishing = st.sidebar.selectbox("Furnishing", ["Furnished", "Unfurnished"])

# -------------------------
# Format price function
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

    # Prepare input
    input_data = pd.DataFrame([{
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'area sqft': area,
        'location_avg_price': location_avg[location],
        'property_type_House': int(property_type == 'House'),
        'furnishing_status_Unfurnished': int(furnishing == 'Unfurnished')
    }])

    try:
        # Predict price per sqft
        pred_ppsqft = model.predict(input_data)[0]

        # Final price
        prediction = pred_ppsqft * area

        formatted_price = format_price(int(prediction))

        # Output
        st.success(f"💰 Estimated Price: {formatted_price}")

        st.info(
            f"⚠️ This prediction is based on average trends in {location}. "
            "Final market prices may vary based on exact property conditions."
        )

        # Model confidence
        st.write(f"📊 Model Confidence (R² Score): {model_r2:.2%}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
