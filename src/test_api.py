# test_api.py

import requests

url = 'http://127.0.0.1:5000/predict'

# Example input data for prediction
data = {
    'temperature_celsius': 20.0,
    'humidity': 50,
    'day_of_week': 2,  # e.g., 0 for Monday, 1 for Tuesday, etc.
    'month': 7
}

response = requests.post(url, json=data)

if response.status_code == 200:
    print("Prediction response:", response.json())
else:
    print("Failed to get a response:", response.status_code, response.text)
