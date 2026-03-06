# -*- coding: utf-8 -*-
"""housing_model.py
Full working version for Streamlit CSV upload.
Handles preprocessing, scaling, one-hot encoding, and trains LR, DT, RF.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

def load_and_preprocess(csv_file):
    """
    Accepts either file path (str) or Streamlit uploaded file object.
    Returns processed dataframe and fitted scaler.
    """
    # Handle uploaded file
    if not isinstance(csv_file, str):
        csv_file.seek(0)
    df = pd.read_csv(csv_file)

    # Fix furnishing status
    df['furnishing_status'] = df['furnishing_status'].replace('huuuhuhhhhhhh', 'Furnished')

    # Ensure numeric columns
    for col in ['bedrooms', 'bathrooms', 'area sqft', 'price']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop invalid rows
    df = df.dropna(subset=['bedrooms', 'bathrooms', 'area sqft', 'price'])
    df = df[df['area sqft'] > 0]

    # Feature engineering
    df['total_rooms'] = df['bedrooms'] + df['bathrooms']
    df['price_per_sqft'] = df['price'] / df['area sqft']

    # One-hot encode categorical features
    df_encoded = pd.get_dummies(
        df,
        columns=['property type', 'furnishing_status', 'location'],
        drop_first=True
    )

    # Standardize numeric features
    scaler = StandardScaler()
    num_features = ['bedrooms', 'bathrooms', 'area sqft', 'total_rooms', 'price_per_sqft']
    df_encoded[num_features] = scaler.fit_transform(df_encoded[num_features])

    # Replace any remaining NaNs/Infs
    df_encoded = df_encoded.replace([np.inf, -np.inf], np.nan).fillna(0)

    return df_encoded, scaler

def train_models(df_encoded):
    """
    Trains Linear Regression, Decision Tree, Random Forest.
    Returns trained models and feature columns.
    """
    if 'price' not in df_encoded.columns:
        raise ValueError("Target column 'price' missing in dataset.")

    y = df_encoded['price']
    X = df_encoded.drop(columns=['price'])

    # Ensure numeric stability
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train models
    lr_model = LinearRegression().fit(X_train, y_train)
    dt_model = DecisionTreeRegressor(random_state=42).fit(X_train, y_train)
    rf_model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1).fit(X_train, y_train)

    return lr_model, dt_model, rf_model, X.columns
