import pandas as pd
from sklearn.ensemble import RandomForestRegressor

def train_model():
    # Load data
    df = pd.read_csv("House_prices.csv")

    # Drop empty columns
    df = df.dropna(axis=1, how='all')

    # Ensure new columns exist
    if 'property_type' not in df.columns:
        df['property_type'] = 'House'  # default
    if 'furnishing_status' not in df.columns:
        df['furnishing_status'] = 'Furnished'  # default

    # Keep only main features
    df = df[['bedrooms', 'bathrooms', 'area sqft', 'location', 'price', 'property_type', 'furnishing_status']]

    # Location average price
    location_avg = df.groupby('location')['price'].mean()
    df['location_avg_price'] = df['location'].map(location_avg)

    # Encode categorical features
    df['property_type_House'] = (df['property_type'] == 'House').astype(int)
    df['furnishing_status_Unfurnished'] = (df['furnishing_status'] == 'Unfurnished').astype(int)

    # Features & target
    X = df[['bedrooms', 'bathrooms', 'area sqft', 'location_avg_price', 'property_type_House', 'furnishing_status_Unfurnished']]
    y = df['price']

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Model confidence (R²)
    model_r2 = model.score(X, y)

    # RETURN 3 THINGS
    return model, location_avg, model_r2
