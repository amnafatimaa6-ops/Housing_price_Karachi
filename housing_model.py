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
    along with the one-hot encoded dataframe for predictions in Streamlit.
    """

    # Strip whitespace from categorical columns
    cat_cols = ['property type', 'furnishing_status', 'location']
    for col in cat_cols:
        df[col] = df[col].astype(str).str.strip()

    # One-hot encode categorical variables
    df_encoded = pd.get_dummies(df, drop_first=True)  # drop_first avoids multicollinearity

    # Add log_price for stability
    df_encoded['log_price'] = np.log1p(df['price'])

    # Prepare features and target
    X = df_encoded.drop(['price', 'log_price'], axis=1)
    y = df_encoded['log_price']

    # Ensure no NaNs sneak in
    imputer = SimpleImputer(strategy='mean')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

    # Initialize models
    lr_model = LinearRegression()
    tree_model = DecisionTreeRegressor(random_state=42)
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Train models
    lr_model.fit(X_train, y_train)
    tree_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)

    return lr_model, tree_model, rf_model, df_encoded
