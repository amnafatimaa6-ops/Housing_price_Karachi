import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="🏡 House Price Estimator", page_icon="🏠")

st.title("🏡 Karachi House & Flat Price Estimator")

# -------------------------
# Load Data & Compute Averages
# -------------------------
@st.cache_resource
def load_data():
    df = pd.read_csv("House_prices.csv").dropna()
    # Ensure property_type and furnishing columns exist
    if 'property_type' not in df.columns:
        df['property_type'] = 'House'
    if 'furnishing_status' not in df.columns:
        df['furnishing_status'] = 'Furnished'

    # Compute average price per location + property_type
    avg_prices = df.groupby(['location', 'property_type'])['price'].median()
    
    # Compute median area per location + property_type
    median_area = df.groupby(['location', 'property_type'])['area sqft'].median()
    
    return df, avg_prices, median_area

df, location_type_price, location_type_area = load_data()

# -------------------------
# Sidebar Inputs
# -------------------------
st.sidebar.header("Property Details")
bedrooms = st.sidebar.slider("Bedrooms", 1, 10, 3)
bathrooms = st.sidebar.slider("Bathrooms", 1, 7, 2)
area = st.sidebar.slider("Area (sqft)", 300, 10000, 1500)
location = st.sidebar.selectbox("Location", sorted(df['location'].unique()))
property_type = st.sidebar.selectbox("Property Type", ["House", "Flat"])
furnishing = st.sidebar.selectbox("Furnishing", ["Furnished", "Unfurnished"])

# -------------------------
# Helper: format price in Cr/Lakh
# -------------------------
def format_price_lakh(amount):
    if amount >= 100:
        crore = amount // 100
        lakh = amount % 100
        return f"{int(crore)} Cr {int(lakh)} Lakh"
    else:
        return f"{int(amount)} Lakh"

# -------------------------
# Prediction
# -------------------------
if st.button("Estimate Price 💰"):

    # Base price: median price for location + property type
    base_price = location_type_price.get((location, property_type), df['price'].median())
    
    # Convert to Lakh for calculations
    base_price_lakh = base_price / 100_000
    
    # Median area for scaling
    median_area = location_type_area.get((location, property_type), df['area sqft'].median())
    
    # Adjust for area
    area_factor = area / median_area
    adjusted_price_lakh = base_price_lakh * (0.9 + 0.2 * min(area_factor, 2))  # small scaling
    
    # Adjust for bedrooms & bathrooms
    adjusted_price_lakh *= 1 + 0.03*(bedrooms-3)
    adjusted_price_lakh *= 1 + 0.02*(bathrooms-2)
    
    # Furnishing adjustment
    if furnishing == "Unfurnished":
        adjusted_price_lakh *= 0.95

    # Price range ±5%
    low_price = adjusted_price_lakh * 0.95
    high_price = adjusted_price_lakh * 1.05
    
    # Format prices
    formatted_low = format_price_lakh(low_price)
    formatted_high = format_price_lakh(high_price)

    # -------------------------
    # Display Results
    # -------------------------
    st.success(f"🏷 Estimated Price Range: {formatted_low} – {formatted_high}")
    st.info(f"⚠️ This estimate is based on historical average prices in {location} for {property_type}s.")
    
    # -------------------------
    # Interactive Plotly Graph
    # -------------------------
    fig = go.Figure(go.Bar(
        x=[f"Min Estimate", f"Max Estimate"],
        y=[low_price, high_price],
        text=[formatted_low, formatted_high],
        textposition='auto',
        marker_color=['lightskyblue', 'lightgreen']
    ))
    fig.update_layout(title=f"📊 Price Range for {property_type} in {location}",
                      yaxis_title="Price (Lakh PKR)")
    
    st.plotly_chart(fig)
