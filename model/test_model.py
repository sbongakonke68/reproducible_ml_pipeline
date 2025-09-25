# test_model.py
import joblib
import pandas as pd

# 1. Load the model from the file
print("Loading the model...")
loaded_model = joblib.load('model/rf_model.joblib')
print("Model loaded successfully!")

# 2. Create some fake data that looks like the original
# Let's create one sample of data with the same features
# (This is like one row from the original dataset, but without the 'quality' column)
new_wine_data = pd.DataFrame([[
    7.4, 0.70, 0.00, 1.90, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4
]], columns=[
    'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
    'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
    'pH', 'sulphates', 'alcohol'
])

print("\nSample data for prediction:")
print(new_wine_data)

# 3. Use the loaded model to make a prediction!
prediction = loaded_model.predict(new_wine_data)

print(f"\nThe model predicts the quality of this wine is: {prediction[0]:.2f}")