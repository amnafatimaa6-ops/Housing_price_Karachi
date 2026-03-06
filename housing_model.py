# housing_model.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def run_models(df):
    df_encoded = pd.get_dummies(df)
    df_encoded['log_price'] = np.log1p(df['price'])

    X = df_encoded.drop(['price', 'log_price'], axis=1)
    y = df_encoded['log_price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lr_model = LinearRegression()
    tree_model = DecisionTreeRegressor()
    rf_model = RandomForestRegressor()

    lr_model.fit(X_train, y_train)
    tree_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)

    return lr_model, tree_model, rf_model, df_encoded
