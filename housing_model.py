# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.preprocessing import RobustScaler
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
    df['bed_bath_ratio'] = df['bedrooms'] / df['bathrooms'].replace(0,1)
    df['area_per_room'] = df['area sqft'] / df['total_rooms']

    # One-hot encoding
    df_encoded = pd.get_dummies(df, columns=['property type','furnishing_status','location'], drop_first=True)

    # Scale numeric features
    numeric_cols = ['bedrooms','bathrooms','area sqft','total_rooms','price_per_sqft',
                    'bed_bath_ratio','area_per_room']
    scaler = RobustScaler()
    df_encoded[numeric_cols] = scaler.fit_transform(df_encoded[numeric_cols])

    df_encoded = df_encoded.replace([np.inf,-np.inf],0).fillna(0)
    return df_encoded, scaler, df

def train_models(df_encoded, cv_folds=5):
    y = df_encoded['price']
    X = df_encoded.drop(columns=['price'])
    X = X.replace([np.inf,-np.inf],0).fillna(0)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize models safely
    lr = LinearRegression()
    lasso = LassoCV(alphas=np.logspace(-2,3,20), cv=cv_folds, max_iter=5000)
    dt = DecisionTreeRegressor(max_depth=15, random_state=42)
    rf = RandomForestRegressor(n_estimators=300, max_depth=15, random_state=42, n_jobs=-1)
    gb = HistGradientBoostingRegressor(max_iter=300, max_depth=5, learning_rate=0.05, random_state=42)

    # Wrap training in try/except to prevent crashes
    models = [lr, lasso, dt, rf, gb]
    holdout_scores = {}
    accuracies = {}

    for model in models:
        try:
            # Cross-validation R²
            kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            acc = np.mean(cross_val_score(model, X_train, y_train, cv=kf, scoring='r2'))
            accuracies[model.__class__.__name__] = acc

            # Fit full model
            model.fit(X_train, y_train)

            # Holdout R²
            y_pred = model.predict(X_test)
            holdout_scores[model.__class__.__name__] = r2_score(y_test, y_pred)
        except Exception as e:
            print(f"Model {model.__class__.__name__} failed: {e}")
            accuracies[model.__class__.__name__] = 0
            holdout_scores[model.__class__.__name__] = 0

    print("[Holdout R²]", holdout_scores)

    return lr, lasso, dt, rf, gb, X.columns, (accuracies.get('LinearRegression',0),
                                             accuracies.get('LassoCV',0),
                                             accuracies.get('DecisionTreeRegressor',0),
                                             accuracies.get('RandomForestRegressor',0),
                                             accuracies.get('HistGradientBoostingRegressor',0))
