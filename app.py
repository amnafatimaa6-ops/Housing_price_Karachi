
try:
    df = pd.read_excel("House_prices.xlsx")  # relative path
except FileNotFoundError:
    st.error("Dataset not found! Make sure House_prices.xlsx is in the same folder as app.py.")
    st.stop()
from housing_model import lr_model, tree_model, rf_model, df_encoded, df

st.title(" Karachi Housing Price Predictor (ML Models)")

st.sidebar.header("Enter Property Details")
area = st.sidebar.number_input("Area (sqft)", 100, 100000, 1000, 50)
bedrooms = st.sidebar.number_input("Bedrooms", 1, 10, 3)
bathrooms = st.sidebar.number_input("Bathrooms", 1, 7, 2)
furnishing = st.sidebar.selectbox("Furnishing Status", ["Furnished", "Unfurnished"])
location = st.sidebar.selectbox("Location", df['location'].unique())
property_type = st.sidebar.selectbox("Property Type", df['property type'].unique())

# ---------- Preprocess user input ----------
def preprocess_input(area, bedrooms, bathrooms, furnishing, location, property_type):
    input_dict = {'area sqft': area, 'bedrooms': bedrooms, 'bathrooms': bathrooms}

    # One-hot columns from df_encoded
    for col in df_encoded.columns:
        if 'property type_' in col or 'furnishing_status_' in col or 'location_' in col:
            input_dict[col] = 0

    # Set 1 for selected input
    if f'property type_{property_type}' in df_encoded.columns:
        input_dict[f'property type_{property_type}'] = 1
    if f'furnishing_status_{furnishing}' in df_encoded.columns:
        input_dict[f'furnishing_status_{furnishing}'] = 1
    if f'location_{location}' in df_encoded.columns:
        input_dict[f'location_{location}'] = 1

    input_df = pd.DataFrame([input_dict])
    return input_df

input_df = preprocess_input(area, bedrooms, bathrooms, furnishing, location, property_type)

# ---------- Predictions ----------
pred_lr = lr_model.predict(input_df)[0]
pred_tree = tree_model.predict(input_df)[0]
pred_rf = rf_model.predict(input_df)[0]

st.subheader(" Predicted Prices")
st.write(f"Linear Regression: PKR {pred_lr:,.0f}")
st.write(f"Decision Tree: PKR {pred_tree:,.0f}")
st.write(f"Random Forest: PKR {pred_rf:,.0f}")

# ---------- Random Forest: Actual vs Predicted plot ----------
st.subheader("Random Forest: Actual vs Predicted (Sample from Dataset)")
y_tree = df['price']
X_tree = df_encoded.drop(columns=['price', 'log_price'], errors='ignore')
rf_pred_full = rf_model.predict(X_tree)

plt.figure(figsize=(8,6))
plt.scatter(y_tree, rf_pred_full, alpha=0.7, color='royalblue', edgecolor='k')
plt.plot([y_tree.min(), y_tree.max()], [y_tree.min(), y_tree.max()], 'r--')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Random Forest Predictions")
st.pyplot(plt)
