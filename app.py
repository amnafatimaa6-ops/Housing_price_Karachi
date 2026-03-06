# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ------------------------
# Load and preprocess data
# ------------------------
@st.cache_data
def load_data(csv_file="House_prices (1).csv"):
    df = pd.read_csv(csv_file)
    
    # Clean furnishing status
    df['furnishing_status'] = df['furnishing_status'].replace('huuuhuhhhhhhh', 'Furnished')
    
    # Feature engineering
    df['total_rooms'] = df['bedrooms'] + df['bathrooms']
    df['price_per_sqft'] = df['price'] / df['area sqft']
    
    # One-hot encode categorical features
    df_encoded = pd.get_dummies(df, columns=['property type', 'furnishing_status', 'location'], drop_first=True)
    
    # Standardize numeric features
    scaler = StandardScaler()
    num_features = ['bedrooms', 'bathrooms', 'area sqft', 'total_rooms', 'price_per_sqft']
    df_encoded[num_features] = scaler.fit_transform(df_encoded[num_features])
    
    return df_encoded, scaler

# ------------------------
# Train models live
# ------------------------
@st.cache_data
def train_models(df_encoded):
    y = df_encoded['price']
    X = df_encoded.drop(columns=['price'])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    lr = LinearRegression().fit(X_train, y_train)
    dt = DecisionTreeRegressor(random_state=42).fit(X_train, y_train)
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1).fit(X_train, y_train)
    
    return lr, dt, rf, X.columns

# ------------------------
# App UI
# ------------------------
st.title("🏠 Karachi Housing Price Predictor")

df_encoded, scaler = load_data()
lr_model, dt_model, rf_model, feature_cols = train_models(df_encoded)

# User input
st.sidebar.header("Enter Property Details")
prop_type = st.sidebar.selectbox("Property Type", ["House", "Apartment", "Penthouse"])
bedrooms = st.sidebar.number_input("Bedrooms", min_value=1, max_value=10, value=3)
bathrooms = st.sidebar.number_input("Bathrooms", min_value=1, max_value=10, value=2)
area_sqft = st.sidebar.number_input("Area (sqft)", min_value=100, max_value=10000, value=1500)
location = st.sidebar.selectbox("Location", ["Bahria Town", "DHA City", "Clifton", "Gulshan"])

# Prepare input for model
input_df = pd.DataFrame({
    "bedrooms": [bedrooms],
    "bathrooms": [bathrooms],
    "area sqft": [area_sqft],
    "total_rooms": [bedrooms + bathrooms],
    "price_per_sqft": [area_sqft / (bedrooms + bathrooms)]  # dummy, won't affect much
})

# One-hot encode same way
for col in feature_cols:
    if col.startswith("property type_"):
        input_df[col] = 1 if prop_type in col else 0
    elif col.startswith("furnishing_status_"):
        input_df[col] = 0  # default to Furnished
    elif col.startswith("location_"):
        input_df[col] = 1 if location in col else 0

# Standardize numeric features
num_features = ['bedrooms', 'bathrooms', 'area sqft', 'total_rooms', 'price_per_sqft']
input_df[num_features] = scaler.transform(input_df[num_features])

# Predict
if st.button("Predict Price"):
    pred_lr = lr_model.predict(input_df)[0]
    pred_dt = dt_model.predict(input_df)[0]
    pred_rf = rf_model.predict(input_df)[0]
    
    st.success(f"Linear Regression: PKR {pred_lr:,.0f}")
    st.success(f"Decision Tree: PKR {pred_dt:,.0f}")
    st.success(f"Random Forest: PKR {pred_rf:,.0f}")
