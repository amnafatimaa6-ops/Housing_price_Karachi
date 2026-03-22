import streamlit as st
import pandas as pd
import pickle

# Load model
model = pickle.load(open("model.pkl", "rb"))
location_avg = pickle.load(open("location_avg.pkl", "rb"))

st.title("🏡 Simple House Price Predictor")

# Inputs
bedrooms = st.slider("Bedrooms", 1, 10, 3)
bathrooms = st.slider("Bathrooms", 1, 7, 3)
area = st.number_input("Area (sqft)", 300, 10000, 1500)

location = st.selectbox("Location", list(location_avg.index))

# Prepare input
input_data = pd.DataFrame([{
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'area sqft': area,
    'location_avg_price': location_avg[location]
}])

# Predict
if st.button("Predict Price 💰"):
    prediction = model.predict(input_data)[0]
    st.success(f"Estimated Price: PKR {int(prediction):,}")
