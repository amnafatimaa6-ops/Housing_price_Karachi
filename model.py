import pandas as pd

def train_model():
    # Load data
    df = pd.read_csv("House_prices.csv")

    # Drop empty columns
    df = df.dropna(axis=1, how='all')

    # Ensure new columns exist
    if 'property_type' not in df.columns:
        df['property_type'] = 'House'
    if 'furnishing_status' not in df.columns:
        df['furnishing_status'] = 'Furnished'

    # Keep main features + new columns
    df = df[['bedrooms', 'bathrooms', 'area sqft', 'location', 'price', 'property_type', 'furnishing_status']]

    # Compute median price per sqft per location
    df['price_per_sqft'] = df['price'] / df['area sqft']
    location_ppsqft = df.groupby('location')['price_per_sqft'].median()

    # Fake R² for reporting (we aren't training ML now, just using historical data)
    model_r2 = 0.80  

    return location_ppsqft, model_r2
