# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
from housing_model import load_and_preprocess, train_models

st.set_page_config(page_title="Karachi Housing Price Predictor", layout="wide")
st.title("🏠 Karachi Housing Price Predictor")

# Upload CSV
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("✅ Dataset loaded successfully!")
    st.dataframe(df.head())

    # Preprocess
    df_encoded, scaler = load_and_preprocess(uploaded_file)
    st.write("✅ Data preprocessed.")

    # Train models
    with st.spinner("Training models..."):
        lr_model, dt_model, rf_model, feature_cols = train_models(df_encoded)
    st.success("✅ Models trained!")

    # User input
    st.header("Enter Property Details to Predict Price")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        prop_type = st.selectbox("Property Type", df['property type'].unique())
    with col2:
        bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3)
    with col3:
        bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)
    with col4:
        area = st.number_input("Area (sqft)", min_value=100, value=1200)

    col5, col6 = st.columns(2)
    with col5:
        location = st.selectbox("Location", df['location'].unique())
    with col6:
        furnishing = st.selectbox("Furnishing Status", df['furnishing_status'].unique())

    if st.button("Predict Price"):
        # Build dataframe
        input_df = pd.DataFrame({
            'bedrooms': [bedrooms],
            'bathrooms': [bathrooms],
            'area sqft': [area],
            'total_rooms': [bedrooms + bathrooms],
            'price_per_sqft': [0],  # placeholder, will scale
            'property type_House': [1 if prop_type=="House" else 0],
            'property type_Apartment': [1 if prop_type=="Apartment" else 0],
            'furnishing_status_Furnished': [1 if furnishing=="Furnished" else 0],
            'furnishing_status_Unfurnished': [1 if furnishing=="Unfurnished" else 0],
            # Add location dummies
            **{f'location_{loc}': 1 if loc==location else 0 for loc in df['location'].unique()}
        })

        # Scale numeric
        input_df[['bedrooms', 'bathrooms', 'area sqft', 'total_rooms', 'price_per_sqft']] = scaler.transform(
            input_df[['bedrooms', 'bathrooms', 'area sqft', 'total_rooms', 'price_per_sqft']]
        )

        # Align with training columns
        input_df = input_df.reindex(columns=feature_cols, fill_value=0)

        # Predict
        pred_lr = lr_model.predict(input_df)[0]
        pred_dt = dt_model.predict(input_df)[0]
        pred_rf = rf_model.predict(input_df)[0]

        st.write(f"**Linear Regression Prediction:** PKR {pred_lr:,.0f}")
        st.write(f"**Decision Tree Prediction:** PKR {pred_dt:,.0f}")
        st.write(f"**Random Forest Prediction:** PKR {pred_rf:,.0f}")
