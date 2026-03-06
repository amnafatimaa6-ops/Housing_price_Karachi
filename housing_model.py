# -*- coding: utf-8 -*-
"""housing_model.py
Fixed version to match your app.
Prepares data, trains models, and returns models for Streamlit.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

def load_and_preprocess(csv_file="House_prices (1).csv"):
    # Load dataset
    df = pd.read_csv(csv_file)

    # Fix bad furnishing entries
    df['furnishing_status'] = df['furnishing_status'].replace('huuuhuhhhhhhh', 'Furnished')

    # Make sure numeric columns are numeric
    df['bedrooms'] = pd.to_numeric(df['bedrooms'], errors='coerce')
    df['bathrooms'] = pd.to_numeric(df['bathrooms'], errors='coerce')
    df['area sqft'] = pd.to_numeric(df['area sqft'], errors='coerce')
    df['price'] = pd.to_numeric(df['price'], errors='coerce')

    # Drop rows with missing or zero area/price
    df = df.dropna(subset=['bedrooms', 'bathrooms', 'area sqft', 'price'])
    df = df[df['area sqft'] > 0]

    # Feature engineering
    df['total_rooms'] = df['bedrooms'] + df['bathrooms']
    df['price_per_sqft'] = df['price'] / df['area sqft']

    # One-hot encode categorical columns exactly as your app expects
    df_encoded = pd.get_dummies(
        df,
        columns=['property type', 'furnishing_status', 'location'],
        drop_first=True
    )

    # Standardize numeric features
    scaler = StandardScaler()
    num_features = ['bedrooms', 'bathrooms', 'area sqft', 'total_rooms', 'price_per_sqft']
    df_encoded[num_features] = scaler.fit_transform(df_encoded[num_features])

    return df_encoded, scaler


def train_models(df_encoded):
    # Target variable
    y = df_encoded['price']
    X = df_encoded.drop(columns=['price'])

    # Replace any inf or NaN in X just in case
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train models
    lr_model = LinearRegression().fit(X_train, y_train)
    dt_model = DecisionTreeRegressor(random_state=42).fit(X_train, y_train)
    rf_model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1).fit(X_train, y_train)

    return lr_model, dt_model, rf_model, X.columns
