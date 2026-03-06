# housing_model.py
# Model training file
# This file should NOT load the dataset itself.
# The dataset (df) will be passed from app.py

# ------------------------
# IMPORTS
# ------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# ------------------------
# MODEL FUNCTION
# ------------------------
def run_models(df):
    # =========================
    # Data inspection
    # =========================
    print(df.isnull().sum())
    print("locations:", df['location'].unique())

    # =========================
    # Data preprocessing
    # =========================
    df_encoded = pd.get_dummies(df)

    # log transform
    df_encoded['log_price'] = np.log1p(df['price'])

    X = df_encoded.drop(['price','log_price'], axis=1)
    y = df_encoded['log_price']

    # =========================
    # Train test split
    # =========================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # =========================
    # Models
    # =========================
    lr_model = LinearRegression()
    tree_model = DecisionTreeRegressor()
    rf_model = RandomForestRegressor()

    # =========================
    # Training
    # =========================
    lr_model.fit(X_train, y_train)
    tree_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)

    # =========================
    # Plots
    # =========================
    plt.figure()
    sns.histplot(df['price'], kde=True)
    plt.title("Price Distribution")
    plt.show()

    plt.figure()
    sns.scatterplot(x=df['area sqft'], y=df['price'])
    plt.title("Area vs Price")
    plt.show()

    # =========================
    # Return objects for app
    # =========================
    return lr_model, tree_model, rf_model, df_encoded, df
