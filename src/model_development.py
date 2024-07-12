# model_development.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
from sklearn.preprocessing import StandardScaler

# Load data with features
data = pd.read_csv('../data/combined_flight_weather_data_with_features.csv')

# Define features and target variable
X = data[['temperature_celsius', 'humidity', 'day_of_week', 'month']]
y = data['price']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)

# Evaluate Linear Regression model
lr_mse = mean_squared_error(y_test, lr_predictions)
lr_mae = mean_absolute_error(y_test, lr_predictions)
print(f"Linear Regression MSE: {lr_mse:.2f}")
print(f"Linear Regression MAE: {lr_mae:.2f}")

# Train a Random Forest Regressor model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

# Evaluate Random Forest Regressor model
rf_mse = mean_squared_error(y_test, rf_predictions)
rf_mae = mean_absolute_error(y_test, rf_predictions)
print(f"Random Forest Regressor MSE: {rf_mse:.2f}")
print(f"Random Forest Regressor MAE: {rf_mae:.2f}")

# Create models directory if it doesn't exist
import os
if not os.path.exists('models'):
    os.makedirs('models')

# Save the trained models
joblib.dump(lr_model, 'models/linear_regression_model.pkl')
joblib.dump(rf_model, 'models/random_forest_model.pkl')

# Save the scaler
scaler = StandardScaler()
scaler.fit(X)
joblib.dump(scaler, 'models/scaler.pkl')
