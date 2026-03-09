if st.button("Predict Price"):
    # 1️⃣ Calculate location_avg_price
    location_avg_price_value = df[df['location']==location]['price'].mean()
    
    # 2️⃣ Prepare numeric dataframe
    input_numeric = pd.DataFrame({
        'bedrooms':[bedrooms],
        'bathrooms':[bathrooms],
        'area sqft':[area],
        'total_rooms':[bedrooms+bathrooms],
        'price_per_sqft':[0],
        'location_avg_price':[location_avg_price_value]
    })

    # 3️⃣ Polynomial transformation (same as training)
    from sklearn.preprocessing import PolynomialFeatures
    num_features = ['bedrooms','bathrooms','area sqft','total_rooms','price_per_sqft','location_avg_price']
    poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
    input_poly = poly.fit_transform(input_numeric[num_features])
    poly_feature_names = poly.get_feature_names_out(num_features)
    input_poly_df = pd.DataFrame(input_poly, columns=poly_feature_names)

    # 4️⃣ Scale numeric features
    input_poly_df[poly_feature_names] = scaler.transform(input_poly_df[poly_feature_names])

    # 5️⃣ Prepare categorical features
    cat_features = pd.DataFrame({
        'property type_House':[1 if prop_type=="House" else 0],
        'property type_Apartment':[1 if prop_type=="Apartment" else 0],
        'furnishing_status_Furnished':[1 if furnishing=="Furnished" else 0],
        'furnishing_status_Unfurnished':[1 if furnishing=="Unfurnished" else 0]
    })

    # Merge all features
    input_df = pd.concat([input_poly_df, cat_features], axis=1)

    # 6️⃣ Align columns to training features
    input_df = input_df.reindex(columns=feature_cols, fill_value=0)

    # 7️⃣ Predict
    pred_lr = lr_model.predict(input_df)[0]
    pred_dt = dt_model.predict(input_df)[0]
    pred_rf = rf_model.predict(input_df)[0]
    pred_gb = gb_model.predict(input_df)[0]
    avg_pred = (pred_lr + pred_dt + pred_rf + pred_gb)/4

    # Display
    st.subheader("Predicted Price")
    st.write(f"**Linear Regression:** {format_price(pred_lr)}")
    st.write(f"**Decision Tree:** {format_price(pred_dt)}")
    st.write(f"**Random Forest:** {format_price(pred_rf)}")
    st.write(f"**Gradient Boosting:** {format_price(pred_gb)}")
    st.write(f"**Average Prediction:** {format_price(avg_pred)}")
