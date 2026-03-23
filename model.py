import pandas as pd
from sklearn.ensemble import RandomForestRegressor

def train_model():
    # Load data
    df = pd.read_csv("House_prices.csv")

    # Clean
    df = df.dropna(axis=1, how='all')

    # Ensure columns exist
    if 'property_type' not in df.columns:
        df['property_type'] = 'House'
    if 'furnishing_status' not in df.columns:
        df['furnishing_status'] = 'Furnished'

    # Keep relevant columns
    df = df[['bedrooms', 'bathrooms', 'area sqft', 'location', 'price', 'property_type', 'furnishing_status']]

    # Location average
    location_avg = df.groupby('location')['price'].mean()
    df['location_avg_price'] = df['location'].map(location_avg)

    # Encode categorical
    df['property_type_House'] = (df['property_type'] == 'House').astype(int)
    df['furnishing_status_Unfurnished'] = (df['furnishing_status'] == 'Unfurnished').astype(int)

    # NEW: price per sqft (this is the missing intelligence)
    df['price_per_sqft'] = df['price'] / df['area sqft']

    # Target: price_per_sqft (NOT raw price, NOT relative)
    y = df['price_per_sqft']

    # Features
    X = df[['bedrooms', 'bathrooms', 'area sqft',
            'location_avg_price',
            'property_type_House',
            'furnishing_status_Unfurnished']]

    # Model
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42
    )
    model.fit(X, y)

    model_r2 = model.score(X, y)

    return model, location_avg, model_r2
