import pandas as pd
from sklearn.ensemble import RandomForestRegressor

def train_model():
    # Load data
    df = pd.read_csv("House_prices.csv")

    # Clean
    df = df.dropna(axis=1, how='all')

    # Keep only main features
    df = df[['bedrooms', 'bathrooms', 'area sqft', 'location', 'price']]

    # Location average price
    location_avg = df.groupby('location')['price'].mean()
    df['location_avg_price'] = df['location'].map(location_avg)

    # Features & target
    X = df[['bedrooms', 'bathrooms', 'area sqft', 'location_avg_price']]
    y = df['price']

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    return model, location_avg
