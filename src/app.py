import requests
from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
from datetime import datetime, timedelta
import random

app = Flask(__name__)

# Load the trained Linear Regression model
lr_model = joblib.load('models/linear_regression_model.pkl')

# Load the scaler
scaler = joblib.load('models/scaler.pkl')

# My OpenWeatherMap API key
API_KEY = '9faef2eafc692677b242ef170a9e7491'

@app.route('/')
def home():
    return render_template('index.html')

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
        
        # Make predictions using Linear Regression
        lr_prediction = lr_model.predict(features_scaled)[0]
        
        best_days.append({
            'date': current_date.strftime('%Y-%m-%d'),
            'departure_temperature': departure_weather_data['temperature'],
            'departure_humidity': departure_weather_data['humidity'],
            'arrival_temperature': arrival_weather_data['temperature'],
            'arrival_humidity': arrival_weather_data['humidity'],
            'airline': get_best_airline(departure_city, arrival_city),
            'price': lr_prediction
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
    airlines = ['Emirate Airline', 'MAI', 'Singapore', 'NorkAir', 'Etihad', 'Qatar Airways']
    return random.choice(airlines)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
