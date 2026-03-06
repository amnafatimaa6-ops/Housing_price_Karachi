import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def run_models(df):
    """
    Takes the housing dataframe and returns trained models
    along with the encoded dataframe.
    """

    # Clean column names (removes hidden spaces that often cause errors)
    df.columns = df.columns.str.strip()

    # Clean categorical columns
    categorical_cols = ['property type', 'furnishing_status', 'location']
    for col in categorical_cols:
        df[col] = df[col].astype(str).str.strip()

    # One-hot encode categorical features
    df_encoded = pd.get_dummies(df, columns=categorical_cols)

    # Log transform price for stability
    df_encoded['log_price'] = np.log1p(df_encoded['price'])

    # Features and target
    X = df_encoded.drop(columns=['price', 'log_price'])
    y = df_encoded['log_price']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Models
    lr_model = LinearRegression()
    tree_model = DecisionTreeRegressor(random_state=42)
    rf_model = RandomForestRegressor(n_estimators=200, random_state=42)

    # Train models
    lr_model.fit(X_train, y_train)
    tree_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)

    # Return trained models + feature columns for prediction later
    return lr_model, tree_model, rf_model, X.columns.tolist()
