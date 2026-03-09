# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
from housing_model import load_and_preprocess, train_models
from sklearn.preprocessing import PolynomialFeatures

# 1️⃣ Page setup
st.set_page_config(page_title="Karachi Housing Price Predictor", layout="wide")
st.title("🏠 Karachi Housing Price Predictor")

# 2️⃣ Load & preprocess dataset
try:
    df_encoded, scaler, df = load_and_preprocess("House_prices (1).csv")
    st.success("✅ Dataset loaded & preprocessed!")
except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    st.stop()

# 3️⃣ Train models
with st.spinner("Training models..."):
    lr_model, dt_model, rf_model, gb_model, feature_cols, accuracies = train_models(df_encoded)

# 4️⃣ Display model R²
acc_lr, acc_dt, acc_rf, acc_gb = accuracies
st.subheader("Model Accuracy (R²)")
st.write(f"**Linear Regression:** {acc_lr:.2f}")
st.write(f"**Decision Tree:** {acc_dt:.2f}")
st.write(f"**Random Forest:** {acc_rf:.2f}")
st.write(f"**Gradient Boosting:** {acc_gb:.2f}")

# 5️⃣ User input
st.header("Enter Property Details")
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

def format_price(num):
    if num >= 1e7:
        return f"{num/1e7:.2f} Cr"
    elif num >= 1e5:
        return f"{num/1e5:.2f} Lakh"
    else:
        return f"{num:,.0f}"

# 6️⃣ Predict button
if st.button("Predict Price"):
    # Prepare input
    input_df = pd.DataFrame({
        'bedrooms':[bedrooms],
        'bathrooms':[bathrooms],
        'area sqft':[area],
        'total_rooms':[bedrooms+bathrooms],
        'price_per_sqft':[0],  # placeholder, scaled later
        'location_avg_price':[df[df['location']==location]['price'].mean()],
        'property type_House':[1 if prop_type=="House" else 0],
        'property type_Apartment':[1 if prop_type=="Apartment" else 0],
        'furnishing_status_Furnished':[1 if furnishing=="Furnished" else 0],
        'furnishing_status_Unfurnished':[1 if furnishing=="Unfurnished" else 0]
    })

    # Polynomial features
    num_features = ['bedrooms','bathrooms','area sqft','total_rooms','price_per_sqft','location_avg_price']
    poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
    poly_features = poly.fit_transform(input_df[num_features])
    poly_feature_names = poly.get_feature_names_out(num_features)
    df_poly = pd.DataFrame(poly_features, columns=poly_feature_names)

    # Merge with categorical
    input_df_final = pd.concat([df_poly, input_df.drop(columns=num_features)], axis=1)

    # Scale numeric
    input_df_final[poly_feature_names] = scaler.transform(df_poly)

    # Align columns
    input_df_final = input_df_final.reindex(columns=feature_cols, fill_value=0)

    # Predict with all models
    preds = np.array([
        lr_model.predict(input_df_final)[0],
        dt_model.predict(input_df_final)[0],
        rf_model.predict(input_df_final)[0],
        gb_model.predict(input_df_final)[0]
    ])
    avg_pred = preds.mean()

    # Confidence: lower std deviation → higher confidence
    std_dev = preds.std()
    confidence = max(0, 100 - std_dev/avg_pred*100)  # simple % confidence proxy

    # Display predictions
    st.subheader("Predicted Price")
    st.write(f"**Linear Regression:** {format_price(preds[0])}")
    st.write(f"**Decision Tree:** {format_price(preds[1])}")
    st.write(f"**Random Forest:** {format_price(preds[2])}")
    st.write(f"**Gradient Boosting:** {format_price(preds[3])}")
    st.write(f"**Average Prediction:** {format_price(avg_pred)}")
    st.write(f"**Model Confidence:** {confidence:.1f}%")
