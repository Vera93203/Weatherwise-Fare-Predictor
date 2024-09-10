# preprocess_data.py

import json
import os
import pandas as pd
from datetime import datetime, timezone
import numpy as np

# Load flight data
with open('../data/flight_data.json', 'r') as f:
    flight_data = json.load(f)

# Load weather data for each city
weather_data = {}
cities = ["London", "New York", "Tokyo"]
for city in cities:
    with open(f'../data/weather_{city}.json', 'r') as f:
        weather_data[city] = json.load(f)

# Function to extract relevant weather info
def extract_weather_info(data):
    return {
        'city': data['name'],
        'temperature': data['main']['temp'],
        'humidity': data['main']['humidity'],
        'weather': data['weather'][0]['description'],
        'date': datetime.fromtimestamp(data['dt'], tz=timezone.utc).strftime('%Y-%m-%d')
    }

# Process weather data
processed_weather_data = []
for city, data in weather_data.items():
    processed_weather_data.append(extract_weather_info(data))

weather_df = pd.DataFrame(processed_weather_data)

# Extract relevant flight info
def extract_flight_info(data):
    flights = []
    for flight in data['states']:
        if flight[3] is not None:  # Check if time_position is not None
            flight_info = {
                'icao24': flight[0],
                'callsign': flight[1],
                'origin_country': flight[2],
                'time_position': flight[3],
                'last_contact': flight[4],
                'longitude': flight[5],
                'latitude': flight[6],
                'baro_altitude': flight[7],
                'on_ground': flight[8],
                'velocity': flight[9],
                'heading': flight[10],
                'vertical_rate': flight[11],
                'geo_altitude': flight[13],
                'squawk': flight[14],
                'spi': flight[15],
                'date': datetime.fromtimestamp(flight[3], tz=timezone.utc).strftime('%Y-%m-%d')
            }
            flights.append(flight_info)
    return flights

processed_flight_data = extract_flight_info(flight_data)
flight_df = pd.DataFrame(processed_flight_data)

# Add a synthetic 'price' column (for illustration purposes)
np.random.seed(42)  # For reproducibility
flight_df['price'] = np.random.randint(100, 1000, size=len(flight_df))

# Combine flight and weather data
combined_data = pd.merge(flight_df, weather_df, on='date', how='inner')

# Save combined data to a CSV file
if not os.path.exists('../data'):
    os.makedirs('../data')
combined_data.to_csv('../data/combined_flight_weather_data.csv', index=False)

print("Data preprocessing complete. Combined data saved to 'data/combined_flight_weather_data.csv'.")


from sklearn.model_selection import train_test_split

# Split data into training and testing sets
X = combined_data.drop(columns=['price', 'icao24', 'date'])
y = combined_data['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



