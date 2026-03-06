import streamlit as st
import pandas as pd
import numpy as np

from housing_model import run_models

st.set_page_config(page_title="Karachi Housing Price Predictor", layout="centered")

st.title("🏠 Karachi Housing Price Predictor")

st.write(
"This app predicts property prices in Karachi using Machine Learning models "
"(Linear Regression, Decision Tree, Random Forest)."
)

# Load dataset

df = pd.read_excel("House_prices.xlsx")

# Train models

lr_model, tree_model, rf_model, feature_cols = run_models(df)

st.sidebar.header("Property Details")

# User Inputs

property_type = st.sidebar.selectbox(
"Property Type",
df["property type"].unique()
)

location = st.sidebar.selectbox(
"Location",
df["location"].unique()
)

furnishing = st.sidebar.selectbox(
"Furnishing Status",
df["furnishing_status"].unique()
)

area = st.sidebar.number_input(
"Area (sqft)",
min_value=100,
max_value=100000,
value=1000
)

bedrooms = st.sidebar.number_input(
"Bedrooms",
min_value=1,
max_value=10,
value=3
)

bathrooms = st.sidebar.number_input(
"Bathrooms",
min_value=1,
max_value=10,
value=3
)

# Create input dataframe

input_data = pd.DataFrame({
"property type": [property_type],
"location": [location],
"furnishing_status": [furnishing],
"area sqft": [area],
"bedrooms": [bedrooms],
"bathrooms": [bathrooms]
})

# One-hot encode like training data

input_encoded = pd.get_dummies(input_data)

# Align columns with training data

input_encoded = input_encoded.reindex(columns=feature_cols, fill_value=0)

# Prediction button

if st.button("Predict Price"):

```
lr_pred = np.expm1(lr_model.predict(input_encoded))[0]
tree_pred = np.expm1(tree_model.predict(input_encoded))[0]
rf_pred = np.expm1(rf_model.predict(input_encoded))[0]

st.subheader("Predicted Prices")

st.write(f"Linear Regression: PKR {lr_pred:,.0f}")
st.write(f"Decision Tree: PKR {tree_pred:,.0f}")
st.write(f"Random Forest: PKR {rf_pred:,.0f}")

avg_price = (lr_pred + tree_pred + rf_pred) / 3

st.success(f"Estimated Market Price: PKR {avg_price:,.0f}")
```
