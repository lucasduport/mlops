import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

np.random.seed(42)
n_samples = 1000

square_feet = np.random.normal(2000, 500, n_samples)
bedrooms = np.random.randint(1, 6, n_samples)
bathrooms = np.random.randint(1, 4, n_samples)
age = np.random.randint(0, 50, n_samples)

# Simulate a price with some noise
price = (
    square_feet * 100 +
    bedrooms * 10000 +
    bathrooms * 15000 -
    age * 1000 +
    np.random.normal(0, 20000, n_samples)
)

data = pd.DataFrame({
    'square_feet': square_feet,
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'age': age,
    'price': price
})

print("Dataset created:")
print(data.head())
print(f"Dataset shape: {data.shape}")

X = data[['square_feet', 'bedrooms', 'bathrooms', 'age']]
y = data['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R^2 Score: {r2:.4f}")

model_filename = 'house_price_model.joblib'
joblib.dump(model, model_filename)
print(f"\nModel saved as {model_filename}")

# sample_data = X_test.head().to_dict('records')
# with open('sample_house_data.json', 'w') as f:
#     import json
#     json.dump(sample_data, f, indent=2)
# print("Sample data saved as sample_house_data.json")