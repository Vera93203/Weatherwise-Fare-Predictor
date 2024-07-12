# store_data.py

import sqlite3
import pandas as pd

# Load combined data from CSV file
combined_data = pd.read_csv('../data/combined_flight_weather_data.csv')

# Connect to SQLite database (or create it if it doesn't exist)
conn = sqlite3.connect('../data/weather_fare_data.db')
cursor = conn.cursor()

# Create tables
cursor.execute('''
CREATE TABLE IF NOT EXISTS flights (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    icao24 TEXT,
    callsign TEXT,
    origin_country TEXT,
    time_position INTEGER,
    last_contact INTEGER,
    longitude REAL,
    latitude REAL,
    baro_altitude REAL,
    on_ground BOOLEAN,
    velocity REAL,
    heading REAL,
    vertical_rate REAL,
    geo_altitude REAL,
    squawk TEXT,
    spi BOOLEAN,
    date TEXT
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS weather (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    city TEXT,
    temperature REAL,
    humidity REAL,
    weather TEXT,
    date TEXT
)
''')

# Insert combined data into tables
for _, row in combined_data.iterrows():
    cursor.execute('''
    INSERT INTO flights (icao24, callsign, origin_country, time_position, last_contact, longitude, latitude, baro_altitude, on_ground, velocity, heading, vertical_rate, geo_altitude, squawk, spi, date)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (row['icao24'], row['callsign'], row['origin_country'], row['time_position'], row['last_contact'], row['longitude'], row['latitude'], row['baro_altitude'], row['on_ground'], row['velocity'], row['heading'], row['vertical_rate'], row['geo_altitude'], row['squawk'], row['spi'], row['date']))

    cursor.execute('''
    INSERT INTO weather (city, temperature, humidity, weather, date)
    VALUES (?, ?, ?, ?, ?)
    ''', (row['city'], row['temperature'], row['humidity'], row['weather'], row['date']))

conn.commit()
conn.close()

print("Data storage complete. Data saved to 'data/weather_fare_data.db'.")
