import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

def run_models(df):
    """
    Takes the raw housing dataframe and returns trained ML models
    along with the one-hot encoded dataframe for predictions.
    """

    # ------------------------------
    # Step 1: Clean the dataframe
    # ------------------------------

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    
    # Strip whitespace from string/object columns
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.strip()
    
    # Convert numeric columns to proper numeric type
    numeric_cols = ['bedrooms', 'bathrooms', 'area sqft', 'price']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Fill any accidental NaNs with 0 (should be rare)
    df[numeric_cols] = df[numeric_cols].fillna(0)

    # ------------------------------
    # Step 2: One-hot encode categories
    # ------------------------------
    cat_cols = ['property type', 'furnishing_status', 'location']
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # ------------------------------
    # Step 3: Create target variable
    # ------------------------------
    df_encoded['log_price'] = np.log1p(df_encoded['price'])

    # Features and target
    X = df_encoded.drop(['price', 'log_price'], axis=1)
    y = df_encoded['log_price']

    # ------------------------------
    # Step 4: Impute missing values if any
    # ------------------------------
    imputer = SimpleImputer(strategy='mean')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # ------------------------------
    # Step 5: Split dataset
    # ------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=0.2, random_state=42
    )

    # ------------------------------
    # Step 6: Initialize and train models
    # ------------------------------
    lr_model = LinearRegression()
    tree_model = DecisionTreeRegressor(random_state=42)
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

    lr_model.fit(X_train, y_train)
    tree_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)

    return lr_model, tree_model, rf_model, df_encoded

# ------------------------------
# Example usage:
# df = pd.read_csv("your_housing_data.csv")
# lr, tree, rf, df_encoded = run_models(df)
# ------------------------------
