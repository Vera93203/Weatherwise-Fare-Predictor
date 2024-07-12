import requests
from flask import Flask, render_template, request, jsonify, url_for
import joblib
import pandas as pd
from datetime import datetime, timedelta
import random

app = Flask(__name__)

# Load the trained models
lr_model = joblib.load('models/linear_regression_model.pkl')
rf_model = joblib.load('models/random_forest_model.pkl')

# Load the scaler
scaler = joblib.load('models/scaler.pkl')

#  OpenWeatherMap API key
API_KEY = '9faef2eafc692677b242ef170a9e7491' 

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Extract features from request
    features = [data['temperature_celsius'], data['humidity'], data['day_of_week'], data['month']]

    # Scale the features
    features_scaled = scaler.transform([features])

    # Make predictions
    lr_prediction = lr_model.predict(features_scaled)[0]
    rf_prediction = rf_model.predict(features_scaled)[0]

    return jsonify({
        'linear_regression_prediction': lr_prediction,
        'random_forest_prediction': rf_prediction
    })

@app.route('/best_days', methods=['POST'])
def best_days():
    data = request.get_json()
    start_date = datetime.strptime(data['start_date'], '%Y-%m-%d')
    end_date = datetime.strptime(data['end_date'], '%Y-%m-%d')
    departure_city = data['departure_city']
    arrival_city = data['arrival_city']
    
    best_days = []
    
    for n in range((end_date - start_date).days + 1):
        current_date = start_date + timedelta(n)
        day_of_week = current_date.weekday()
        month = current_date.month
        
        # Fetch or simulate weather data for the departure and arrival cities and date
        departure_weather_data = get_weather_data(departure_city, current_date)
        arrival_weather_data = get_weather_data(arrival_city, current_date)
        temperature_celsius = (departure_weather_data['temperature'] + arrival_weather_data['temperature']) / 2
        humidity = (departure_weather_data['humidity'] + arrival_weather_data['humidity']) / 2
        
        # Prepare features
        features = [temperature_celsius, humidity, day_of_week, month]
        features_scaled = scaler.transform([features])
        
        # Make predictions
        lr_prediction = lr_model.predict(features_scaled)[0]
        rf_prediction = rf_model.predict(features_scaled)[0]
        
        avg_prediction = (lr_prediction + rf_prediction) / 2
        
        # Simulate realistic prices
        price = simulate_price(departure_city, arrival_city, current_date)
        
        best_days.append({
            'date': current_date.strftime('%Y-%m-%d'),
            'temperature': temperature_celsius,
            'humidity': humidity,
            'airline': get_best_airline(departure_city, arrival_city),
            'price': price
        })
    
    # Sort by price to find the best days
    best_days.sort(key=lambda x: x['price'])
    
    return jsonify(best_days[:5])  # Return top 5 best days

def get_weather_data(city, date):
    # Fetch weather data for the given city and date using OpenWeatherMap API
    url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    data = response.json()
    
    # Find the weather data corresponding to the requested date
    for item in data['list']:
        item_date = datetime.strptime(item['dt_txt'], '%Y-%m-%d %H:%M:%S')
        if item_date.date() == date.date():
            return {
                'temperature': item['main']['temp'],
                'humidity': item['main']['humidity']
            }
    
    # Default to some value if not found (or handle as an error)
    return {
        'temperature': 20,  # Default temperature
        'humidity': 50  # Default humidity
    }

def get_best_airline(departure_city, arrival_city):
    # Simulate airline data based on the route
    airlines = ['Emirate Airline', 'Etihad', 'Qatar Airways']
    return random.choice(airlines)

def simulate_price(departure_city, arrival_city, date):
    # Simulate a more realistic price based on distance, demand, and date
    base_price = random.randint(300, 1000)  # Base price range
    # Adjust price based on departure and arrival cities
    if (departure_city in ["New York", "London", "Tokyo"] or arrival_city in ["New York", "London", "Tokyo"]):
        base_price += 200  # Increase for popular cities
    
    # Adjust price based on date (higher prices during peak seasons)
    if date.month in [6, 7, 8, 12]:  # Summer and winter holidays
        base_price += 150
    
    return base_price

if __name__ == '__main__':
    app.run(debug=True)
