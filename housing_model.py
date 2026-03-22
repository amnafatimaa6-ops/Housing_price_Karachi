import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load data
df = pd.read_csv("House_prices.csv")

# Clean
df = df.dropna(axis=1, how='all')

# -------------------------
# Keep ONLY main features
# -------------------------
df = df[['bedrooms', 'bathrooms', 'area sqft', 'location', 'price']]

# -------------------------
# Location average price
# -------------------------
location_avg = df.groupby('location')['price'].mean()
df['location_avg_price'] = df['location'].map(location_avg)

# -------------------------
# Final features
# -------------------------
X = df[['bedrooms', 'bathrooms', 'area sqft', 'location_avg_price']]
y = df['price']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save everything
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(location_avg, open("location_avg.pkl", "wb"))
