import requests
import json
import os

# Replace 'your_openweather_api_key' with your actual API key
API_KEY = '9faef2eafc692677b242ef170a9e7491'
BASE_URL = 'http://api.openweathermap.org/data/2.5/weather?'

def fetch_weather_data(city):
    request_url = f"{BASE_URL}appid={API_KEY}&q={city}"
    response = requests.get(request_url)
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        print(f"Failed to fetch data: {response.status_code}")
        return None

if __name__ == "__main__":
    cities = ["London", "New York", "Tokyo"]  # Add more cities as needed
    for city in cities:
        weather_data = fetch_weather_data(city)
        if weather_data:
            if not os.path.exists('../data'):
                os.makedirs('../data')
            with open(f'../data/weather_{city}.json', 'w') as f:
                json.dump(weather_data, f, indent=4)
            print(f"Weather data for {city} saved successfully.")
        else:
            print(f"Failed to fetch weather data for {city}.")
