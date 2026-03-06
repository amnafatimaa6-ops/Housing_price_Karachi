# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

def load_and_preprocess(csv_file="House_prices (1).csv"):
    df = pd.read_csv(csv_file)

    # Fix furnishing status typo
    df['furnishing_status'] = df['furnishing_status'].replace('huuuhuhhhhhhh', 'Furnished')

    # Ensure numeric
    for col in ['bedrooms','bathrooms','area sqft','price']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Remove invalid rows & extreme outliers
    df = df.dropna(subset=['bedrooms','bathrooms','area sqft','price'])
    df = df[(df['area sqft'] > 100) & (df['price'] > 50000)]

    # Feature engineering
    df['total_rooms'] = df['bedrooms'] + df['bathrooms']
    df['price_per_sqft'] = df['price'] / df['area sqft']

    # One-hot encode categorical
    df_encoded = pd.get_dummies(df, columns=['property type','furnishing_status','location'], drop_first=True)

    # Scale numeric
    scaler = StandardScaler()
    num_features = ['bedrooms','bathrooms','area sqft','total_rooms','price_per_sqft']
    df_encoded[num_features] = scaler.fit_transform(df_encoded[num_features])

    df_encoded = df_encoded.replace([np.inf,-np.inf],0).fillna(0)

    return df_encoded, scaler, df

def train_models(df_encoded, cv_folds=5):
    # Split target
    y = df_encoded['price']
    X = df_encoded.drop(columns=['price'])

    X = X.replace([np.inf,-np.inf],0).fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Models
    lr = LinearRegression()
    dt = DecisionTreeRegressor(random_state=42)
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    gb = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, max_depth=5, random_state=42)

    # Fit models
    lr.fit(X_train, y_train)
    dt.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    gb.fit(X_train, y_train)

    # Cross-validation R²
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    acc_lr = np.mean(cross_val_score(LinearRegression(), X_train, y_train, cv=kf, scoring='r2'))
    acc_dt = np.mean(cross_val_score(DecisionTreeRegressor(random_state=42), X_train, y_train, cv=kf, scoring='r2'))
    acc_rf = np.mean(cross_val_score(RandomForestRegressor(n_estimators=200, random_state=42), X_train, y_train, cv=kf, scoring='r2'))
    acc_gb = np.mean(cross_val_score(HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, max_depth=5, random_state=42),
                                     X_train, y_train, cv=kf, scoring='r2'))

    return lr, dt, rf, gb, X.columns, (acc_lr, acc_dt, acc_rf, acc_gb)
