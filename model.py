import pandas as pd

def train_model():
    # Load dataset
    df = pd.read_csv("House_prices.csv")

    # Drop fully empty columns
    df = df.dropna(axis=1, how='all')

    # Ensure property_type and furnishing_status exist
    if 'property_type' not in df.columns:
        df['property_type'] = 'House'
    if 'furnishing_status' not in df.columns:
        df['furnishing_status'] = 'Furnished'

    # Keep only necessary columns
    df = df[['bedrooms', 'bathrooms', 'area sqft', 'location', 'price', 'property_type', 'furnishing_status']]

    # Median price per location & property type
    location_type_median = df.groupby(['location','property_type'])['price'].median()

    # Median area per location & property type
    location_type_area = df.groupby(['location','property_type'])['area sqft'].median()

    # Store for app usage
    return location_type_median, location_type_area
