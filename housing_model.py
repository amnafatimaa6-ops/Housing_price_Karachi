# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

def train_models(df_encoded, cv_folds=5):
    # Log for LR
    df_encoded['log_price'] = np.log1p(df_encoded['price'])
    X = df_encoded.drop(columns=['price','log_price'])
    y_lr = df_encoded['log_price']
    y_tree = df_encoded['price']

    # Split
    X_train, X_test, y_lr_train, y_lr_test = train_test_split(X, y_lr, test_size=0.2, random_state=42)
    _, _, y_tree_train, y_tree_test = train_test_split(X, y_tree, test_size=0.2, random_state=42)

    # Models
    lr = LinearRegression().fit(X_train, y_lr_train)
    dt = DecisionTreeRegressor(random_state=42).fit(X_train, y_tree_train)
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1).fit(X_train, y_tree_train)
    
    # Gradient boosting
    gb = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, max_depth=5, random_state=42)
    gb.fit(X_train, y_tree_train)

    # Cross-validation R²
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    acc_lr = np.mean(cross_val_score(LinearRegression(), X_train, y_lr_train, cv=kf, scoring='r2'))
    acc_dt = np.mean(cross_val_score(DecisionTreeRegressor(random_state=42), X_train, y_tree_train, cv=kf, scoring='r2'))
    acc_rf = np.mean(cross_val_score(RandomForestRegressor(n_estimators=200, random_state=42), X_train, y_tree_train, cv=kf, scoring='r2'))
    acc_gb = np.mean(cross_val_score(HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, max_depth=5, random_state=42), X_train, y_tree_train, cv=kf, scoring='r2'))

    # Print holdout R² for reference
    y_pred_lr = np.expm1(lr.predict(X_test))
    y_pred_dt = dt.predict(X_test)
    y_pred_rf = rf.predict(X_test)
    y_pred_gb = gb.predict(X_test)

    holdout_acc_lr = r2_score(np.expm1(y_lr_test), y_pred_lr)
    holdout_acc_dt = r2_score(y_tree_test, y_pred_dt)
    holdout_acc_rf = r2_score(y_tree_test, y_pred_rf)
    holdout_acc_gb = r2_score(y_tree_test, y_pred_gb)

    print(f"[Holdout R²] LR: {holdout_acc_lr:.2f}, DT: {holdout_acc_dt:.2f}, RF: {holdout_acc_rf:.2f}, GB: {holdout_acc_gb:.2f}")

    return lr, dt, rf, gb, X.columns, (acc_lr, acc_dt, acc_rf, acc_gb)
