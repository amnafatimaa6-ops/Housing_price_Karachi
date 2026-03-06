# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

def load_and_preprocess(csv_file="House_prices (1).csv"):
    df = pd.read_csv(csv_file)

    # Fix furnishing status
    df['furnishing_status'] = df['furnishing_status'].replace('huuuhuhhhhhhh', 'Furnished')

    # Convert numeric columns
    for col in ['bedrooms','bathrooms','area sqft','price']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop invalid rows
    df = df.dropna(subset=['bedrooms','bathrooms','area sqft','price'])
    df = df[(df['area sqft'] > 100) & (df['price'] > 50000)]  # remove extreme outliers

    # Log-transform price
    df['log_price'] = np.log1p(df['price'])

    # Feature engineering
    df['total_rooms'] = df['bedrooms'] + df['bathrooms']
    df['price_per_sqft'] = df['price'] / df['area sqft']

    # One-hot encode categorical
    df_encoded = pd.get_dummies(df, columns=['property type','furnishing_status','location'], drop_first=True)

    # Scale numeric only
    scaler = StandardScaler()
    num_features = ['bedrooms','bathrooms','area sqft','total_rooms','price_per_sqft']
    df_encoded[num_features] = scaler.fit_transform(df_encoded[num_features])

    df_encoded = df_encoded.replace([np.inf,-np.inf],0).fillna(0)

    return df_encoded, scaler, df

def train_models(df_encoded, cv_folds=5):
    y = df_encoded['log_price']  # use log for LR
    X = df_encoded.drop(columns=['price','log_price'])

    X = X.replace([np.inf,-np.inf],0).fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

    # Initialize models
    lr = LinearRegression()
    dt = DecisionTreeRegressor(random_state=42)
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)

    # Cross-validation
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    acc_lr = np.mean(cross_val_score(lr, X_train, y_train, cv=kf, scoring='r2'))
    acc_dt = np.mean(cross_val_score(dt, X_train, y_train, cv=kf, scoring='r2'))
    acc_rf = np.mean(cross_val_score(rf, X_train, y_train, cv=kf, scoring='r2'))

    # Fit on full training set
    lr.fit(X_train, y_train)
    dt.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    # Convert predictions back to original scale
    y_pred_lr = np.expm1(lr.predict(X_test))
    y_pred_dt = np.expm1(dt.predict(X_test))
    y_pred_rf = np.expm1(rf.predict(X_test))

    holdout_acc_lr = r2_score(np.expm1(y_test), y_pred_lr)
    holdout_acc_dt = r2_score(np.expm1(y_test), y_pred_dt)
    holdout_acc_rf = r2_score(np.expm1(y_test), y_pred_rf)

    print(f"[Holdout R²] LR: {holdout_acc_lr:.2f}, DT: {holdout_acc_dt:.2f}, RF: {holdout_acc_rf:.2f}")

    return lr, dt, rf, X.columns, (acc_lr, acc_dt, acc_rf)
