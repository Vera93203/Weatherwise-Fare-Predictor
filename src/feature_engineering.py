# feature_engineering.py

import pandas as pd
from datetime import datetime

# Load combined data from CSV file
combined_data = pd.read_csv('../data/combined_flight_weather_data.csv')

# Create new features based on existing data
combined_data['temperature_celsius'] = combined_data['temperature'] - 273.15
combined_data['day_of_week'] = pd.to_datetime(combined_data['date']).dt.dayofweek
combined_data['month'] = pd.to_datetime(combined_data['date']).dt.month

# Drop unnecessary columns
combined_data = combined_data.drop(columns=['callsign', 'squawk'])

# Save the data with new features
combined_data.to_csv('../data/combined_flight_weather_data_with_features.csv', index=False)

print("Feature engineering complete. Data with new features saved to 'data/combined_flight_weather_data_with_features.csv'.")
