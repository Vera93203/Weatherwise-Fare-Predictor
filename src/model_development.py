import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os

# Load data
data = pd.read_csv('../data/combined_flight_weather_data_with_features.csv')

# Check for non-numeric columns
non_numeric_columns = data.select_dtypes(include=['object']).columns
print(f"Non-numeric columns: {non_numeric_columns}")

# Handle non-numeric columns appropriately
categorical_features = ['origin_country', 'city', 'weather']
numeric_features = ['temperature_celsius', 'humidity', 'day_of_week', 'month']

# Preprocessing for numerical data
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Define the target variable
y = data['price']

# Apply the transformations and split the data
X = data.drop(columns=['price', 'icao24', 'date'])  # Dropping 'icao24' and 'date' columns

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to evaluate and return results for a given model
def evaluate_model(model, model_name):
    # Create a pipeline for the model
    model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                     ('model', model)])
    
    # Train the model
    model_pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model_pipeline.predict(X_test)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    accuracy = r2 * 100  # Converting R^2 to accuracy percentage
    
    print(f"{model_name} MSE: {mse:.2f}")
    print(f"{model_name} MAE: {mae:.2f}")
    print(f"{model_name} Accuracy: {accuracy:.2f}%")
    
    return model_pipeline, model_name, mse, mae, accuracy

# Evaluate models
results = []
models = {
    'Linear Regression': LinearRegression(),
    'Gradient Boosting Regressor': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'Support Vector Regressor': SVR(kernel='rbf')
}

for model_name, model in models.items():
    model_pipeline, name, mse, mae, accuracy = evaluate_model(model, model_name)
    results.append((name, mse, mae, accuracy))
    
    # Save the model
    joblib.dump(model_pipeline, f'models/{model_name.replace(" ", "_").lower()}_pipeline.pkl')

# Summarize the results in a table
results_df = pd.DataFrame(results, columns=['Model', 'MSE', 'MAE', 'Accuracy'])
print("\nComparison Table:")
print(results_df)

# Find the best two models based on Accuracy
best_two_models = results_df.nlargest(2, 'Accuracy')
print("\nBest Two Models based on Accuracy:")
print(best_two_models)

# Determine the best model
best_model = best_two_models.iloc[0]['Model']
best_model_name = best_model

print(f"\nThe best model is: {best_model_name}")

# Save the best model separately
best_model_pipeline = joblib.load(f'models/{best_model_name.replace(" ", "_").lower()}_pipeline.pkl')
joblib.dump(best_model_pipeline, 'models/best_model.pkl')

# Filter numeric columns for correlation matrix
numeric_data = data.select_dtypes(include=['number'])

# Visualize Correlation Matrix
corr_matrix = numeric_data.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()

# Visualize Pairplot
sns.pairplot(data[['temperature_celsius', 'humidity', 'day_of_week', 'month', 'price']])
plt.show()
