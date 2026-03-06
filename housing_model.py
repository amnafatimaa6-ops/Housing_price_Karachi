import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

def clean_and_encode(df):
    # Strip spaces from column names
    df.columns = df.columns.str.strip()
    
    # Convert numeric columns properly
    numeric_cols = ['bedrooms', 'bathrooms', 'area sqft', 'price']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Strip string columns
    for col in ['property type', 'furnishing_status', 'location']:
        df[col] = df[col].astype(str).str.strip()
    
    # One-hot encode categorical
    df_encoded = pd.get_dummies(df, columns=['property type', 'furnishing_status', 'location'], drop_first=True)
    
    return df_encoded

def run_models(df):
    # Clean and encode
    df_encoded = clean_and_encode(df)
    
    # Create target variable
    df_encoded['log_price'] = np.log1p(df_encoded['price'])
    
    # Features and target
    X = df_encoded.drop(['price', 'log_price'], axis=1)
    y = df_encoded['log_price']
    
    # Impute just in case
    X = pd.DataFrame(SimpleImputer(strategy='mean').fit_transform(X), columns=X.columns)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Models
    lr = LinearRegression().fit(X_train, y_train)
    tree = DecisionTreeRegressor(random_state=42).fit(X_train, y_train)
    rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
    
    return lr, tree, rf, df_encoded, X_train.columns

# Example usage:
# df = pd.read_csv("your_housing_data.csv")
# lr, tree, rf, df_encoded, features = run_models(df)
# print("Training complete. Feature columns:", features)
