import requests
import json
import os


BASE_URL = 'https://opensky-network.org/api/states/all'

def fetch_flight_data():
    response = requests.get(BASE_URL)
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        print(f"Failed to fetch data: {response.status_code}")
        return None

if __name__ == "__main__":
    flight_data = fetch_flight_data()
    if flight_data:
        if not os.path.exists('../data'):
            os.makedirs('../data')
        with open('../data/flight_data.json', 'w') as f:
            json.dump(flight_data, f, indent=4)
        print("Flight data saved successfully.")
    else:
        print("Failed to fetch flight data.")


    
