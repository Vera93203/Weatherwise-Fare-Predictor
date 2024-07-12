# eda.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load combined data from CSV file
combined_data = pd.read_csv('../data/combined_flight_weather_data.csv')

# Display basic information about the dataset
print("Basic Information")
print(combined_data.info())

# Display basic statistics about the dataset
print("\nBasic Statistics")
print(combined_data.describe())

# Display the first few rows of the dataset
print("\nFirst Few Rows of the Dataset")
print(combined_data.head())

# Check for missing values
print("\nMissing Values")
print(combined_data.isnull().sum())

# Filter out non-numeric columns for correlation matrix
numeric_data = combined_data.select_dtypes(include=['float64', 'int64'])

# Visualize correlations using a heatmap
plt.figure(figsize=(12, 8))
correlation_matrix = numeric_data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# Scatter plot of temperature vs. altitude
plt.figure(figsize=(10, 6))
sns.scatterplot(x='temperature', y='baro_altitude', data=combined_data)
plt.title('Temperature vs. Barometric Altitude')
plt.xlabel('Temperature (K)')
plt.ylabel('Barometric Altitude')
plt.show()

# Boxplot of barometric altitude by city
plt.figure(figsize=(10, 6))
sns.boxplot(x='city', y='baro_altitude', data=combined_data)
plt.title('Barometric Altitude by City')
plt.xlabel('City')
plt.ylabel('Barometric Altitude')
plt.show()

# Pair plot for selected features
plt.figure(figsize=(12, 10))
selected_features = ['temperature', 'humidity', 'baro_altitude']
sns.pairplot(combined_data[selected_features])
plt.suptitle('Pair Plot for Selected Features')
plt.show()

print("EDA complete.")
