# -*- coding: utf-8 -*-
"""housing_model.py
Cleaned and modular version for Streamlit usage.
Prepares data, trains models, and returns trained models for prediction.
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

    # Fix furnishing status if there are bad entries
    df['furnishing_status'] = df['furnishing_status'].replace('huuuhuhhhhhhh', 'Furnished')

    # Feature engineering
    df['total_rooms'] = df['bedrooms'] + df['bathrooms']
    df['price_per_sqft'] = df['price'] / df['area sqft']
    df['log_price'] = np.log1p(df['price'])
    df['log_area'] = np.log1p(df['area sqft'])

    # One-hot encode categorical features
    df_encoded = pd.get_dummies(
        df,
        columns=['property type', 'furnishing_status', 'location'],
        drop_first=True
    )

    # Standardize numeric features
    scaler = StandardScaler()
    num_features = ['bedrooms', 'bathrooms', 'area sqft', 'price', 'price_per_sqft', 'total_rooms', 'log_price', 'log_area']
    df_encoded[num_features] = scaler.fit_transform(df_encoded[num_features])

    return df_encoded, scaler

def train_linear_regression(df_encoded):
    # Use raw price as target
    y = df_encoded['price']
    X = df_encoded.drop(columns=['price', 'log_price'])  # keep features only

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    return lr_model, X.columns  # return trained model and feature names

def train_decision_tree(df_encoded):
    y = df_encoded['price']
    X = df_encoded.drop(columns=['price', 'log_price'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    dt_model = DecisionTreeRegressor(random_state=42)
    dt_model.fit(X_train, y_train)

    return dt_model, X.columns

def train_random_forest(df_encoded):
    y = df_encoded['price']
    X = df_encoded.drop(columns=['price', 'log_price'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)

    return rf_model, X.columns

# Convenience function to train all three models at once
def train_all_models(csv_file="House_prices (1).csv"):
    df_encoded, scaler = load_and_preprocess(csv_file)
    lr, features_lr = train_linear_regression(df_encoded)
    dt, features_dt = train_decision_tree(df_encoded)
    rf, features_rf = train_random_forest(df_encoded)
    return {
        "linear": (lr, features_lr),
        "decision_tree": (dt, features_dt),
        "random_forest": (rf, features_rf),
        "scaler": scaler,
        "df_encoded": df_encoded
    }
