# app.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# ---------------- Dataset ----------------
BASE_DIR = os.getcwd()
dataset_path = os.path.join(BASE_DIR, "House_prices (1).csv")

try:
    df = pd.read_csv(dataset_path, encoding="latin1", on_bad_lines="skip")
except Exception as e:
    st.error(f"Error loading CSV: {e}")
    st.stop()

st.write("Dataset loaded successfully!")
st.dataframe(df.head())

# ---------------- Train models ----------------
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

lr_model, tree_model, rf_model, df_encoded = run_models(df)

# ---------------- Streamlit UI ----------------
st.title("Karachi Housing Price Predictor (ML Models)")

st.sidebar.header("Enter Property Details")
area = st.sidebar.number_input("Area (sqft)", 100, 100000, 1000, 50)
bedrooms = st.sidebar.number_input("Bedrooms", 1, 10, 3)
bathrooms = st.sidebar.number_input("Bathrooms", 1, 7, 2)
furnishing = st.sidebar.selectbox("Furnishing Status", ["Furnished", "Unfurnished"])
location = st.sidebar.selectbox("Location", df['location'].unique())
property_type = st.sidebar.selectbox("Property Type", df['property type'].unique())

# ---------- Preprocess user input ----------
def preprocess_input(area, bedrooms, bathrooms, furnishing, location, property_type, df_encoded):
    # Start with zeros for all columns
    input_dict = dict.fromkeys(df_encoded.drop(['price','log_price'], axis=1).columns, 0)

    # Set numeric values
    input_dict['area sqft'] = area
    input_dict['bedrooms'] = bedrooms
    input_dict['bathrooms'] = bathrooms

    # Set 1 for selected categorical features
    for col_name in [
        f'property type_{property_type}',
        f'furnishing_status_{furnishing}',
        f'location_{location}'
    ]:
        if col_name in input_dict:
            input_dict[col_name] = 1

    return pd.DataFrame([input_dict])

input_df = preprocess_input(area, bedrooms, bathrooms, furnishing, location, property_type, df_encoded)

# ---------- Predictions ----------
pred_lr = lr_model.predict(input_df)[0]
pred_tree = tree_model.predict(input_df)[0]
pred_rf = rf_model.predict(input_df)[0]

st.subheader("Predicted Prices")
st.write(f"Linear Regression: PKR {np.expm1(pred_lr):,.0f}")
st.write(f"Decision Tree: PKR {np.expm1(pred_tree):,.0f}")
st.write(f"Random Forest: PKR {np.expm1(pred_rf):,.0f}")

# ---------- Random Forest: Actual vs Predicted plot ----------
st.subheader("Random Forest: Actual vs Predicted (Sample from Dataset)")
y_actual = df['price']
X_rf = df_encoded.drop(['price', 'log_price'], axis=1)
rf_pred_full = np.expm1(rf_model.predict(X_rf))

plt.figure(figsize=(8,6))
plt.scatter(y_actual, rf_pred_full, alpha=0.7, color='royalblue', edgecolor='k')
plt.plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], 'r--')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Random Forest Predictions")
st.pyplot(plt)
