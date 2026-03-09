# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.preprocessing import RobustScaler, PolynomialFeatures
from sklearn.metrics import r2_score

def load_and_preprocess(csv_file="House_prices (1).csv"):
    df = pd.read_csv(csv_file)

    # Fix furnishing typo
    df['furnishing_status'] = df['furnishing_status'].replace('huuuhuhhhhhhh', 'Furnished')

    # Numeric conversion
    for col in ['bedrooms','bathrooms','area sqft','price']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop NaNs and extreme outliers
    df = df.dropna(subset=['bedrooms','bathrooms','area sqft','price'])
    df = df[(df['area sqft'] > 100) & (df['price'] > 50000) & (df['price'] < 15e7)]

    # Feature engineering
    df['total_rooms'] = df['bedrooms'] + df['bathrooms']
    df['price_per_sqft'] = df['price'] / df['area sqft']

    # --- LOCATION FEATURE: average price per location ---
    location_avg_price = df.groupby('location')['price'].mean().to_dict()
    df['location_avg_price'] = df['location'].map(location_avg_price)

    # One-hot encode categorical columns except location (already handled)
    df_encoded = pd.get_dummies(df, columns=['property type','furnishing_status'], drop_first=True)

    # Optionally: add polynomial features for numeric columns to capture non-linearities
    num_features = ['bedrooms','bathrooms','area sqft','total_rooms','price_per_sqft','location_avg_price']
    poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
    poly_features = poly.fit_transform(df_encoded[num_features])
    poly_feature_names = poly.get_feature_names_out(num_features)
    df_poly = pd.DataFrame(poly_features, columns=poly_feature_names, index=df_encoded.index)

    # Merge polynomial features with encoded categorical columns
    df_encoded = pd.concat([df_poly, df_encoded.drop(columns=num_features + ['location'])], axis=1)

    # Scale numeric features
    scaler = RobustScaler()
    df_encoded[poly_feature_names] = scaler.fit_transform(df_encoded[poly_feature_names])

    df_encoded = df_encoded.replace([np.inf,-np.inf],0).fillna(0)

    return df_encoded, scaler, df

def train_models(df_encoded, cv_folds=5):
    y = df_encoded['price']
    X = df_encoded.drop(columns=['price'])
    X = X.replace([np.inf,-np.inf],0).fillna(0)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize models
    lr = LinearRegression()
    dt = DecisionTreeRegressor(max_depth=15, random_state=42)
    rf = RandomForestRegressor(n_estimators=500, max_depth=15, random_state=42, n_jobs=-1)
    gb = HistGradientBoostingRegressor(max_iter=500, max_depth=5, learning_rate=0.05, random_state=42)

    # Cross-validation
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    acc_lr = np.mean(cross_val_score(lr, X_train, y_train, cv=kf, scoring='r2'))
    acc_dt = np.mean(cross_val_score(dt, X_train, y_train, cv=kf, scoring='r2'))
    acc_rf = np.mean(cross_val_score(rf, X_train, y_train, cv=kf, scoring='r2'))
    acc_gb = np.mean(cross_val_score(gb, X_train, y_train, cv=kf, scoring='r2'))

    # Fit on full training set
    lr.fit(X_train, y_train)
    dt.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    gb.fit(X_train, y_train)

    # Holdout R²
    y_pred_lr = lr.predict(X_test)
    y_pred_dt = dt.predict(X_test)
    y_pred_rf = rf.predict(X_test)
    y_pred_gb = gb.predict(X_test)

    holdout_acc_lr = r2_score(y_test, y_pred_lr)
    holdout_acc_dt = r2_score(y_test, y_pred_dt)
    holdout_acc_rf = r2_score(y_test, y_pred_rf)
    holdout_acc_gb = r2_score(y_test, y_pred_gb)

    print(f"[Holdout R²] LR: {holdout_acc_lr:.2f}, DT: {holdout_acc_dt:.2f}, RF: {holdout_acc_rf:.2f}, GB: {holdout_acc_gb:.2f}")

    return lr, dt, rf, gb, X.columns, (acc_lr, acc_dt, acc_rf, acc_gb)
