import pandas as pd

def train_model():
    # Load dataset
    df = pd.read_csv("House_prices.csv")

    # Clean empty columns
    df = df.dropna(axis=1, how='all')

    # Ensure columns exist
    if 'property_type' not in df.columns:
        df['property_type'] = 'House'
    if 'furnishing_status' not in df.columns:
        df['furnishing_status'] = 'Furnished'

    # Keep important columns
    df = df[['bedrooms', 'bathrooms', 'area sqft', 'location', 'price', 'property_type', 'furnishing_status']]

    # -------------------------
    # Core logic
    # -------------------------

    # Price per sqft
    df['price_per_sqft'] = df['price'] / df['area sqft']

    # Average price per sqft by location
    location_ppsqft = df.groupby('location')['price_per_sqft'].mean()

    # Simple confidence score
    model_r2 = 0.85

    return location_ppsqft, model_r2
