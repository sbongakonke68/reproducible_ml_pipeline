# predict.py
import joblib
import pandas as pd
import numpy as np
from preprocess_data import preprocess_data

def predict_wine_quality(model_path, params_path, new_data):
    """
    Make predictions on new wine data using the trained model.
    """
    # Load the model and parameters
    model = joblib.load(model_path)
    training_params = joblib.load(params_path)
    
    print("✅ Model and parameters loaded successfully.")
    
    # Preprocess the new data using the same parameters from training
    X_processed, _ = preprocess_data(
        new_data, 
        is_training=False, 
        training_params=training_params
    )
    
    print(f"✅ New data processed. Shape: {X_processed.shape}")
    
    # Make predictions
    predictions = model.predict(X_processed)
    
    return predictions

def main():
    # Example: Create some new wine data for prediction
    new_wines = pd.DataFrame([{
        'fixed acidity': 7.4,
        'volatile acidity': 0.70,
        'citric acid': 0.00,
        'residual sugar': 1.9,
        'chlorides': 0.076,
        'free sulfur dioxide': 11.0,
        'total sulfur dioxide': 34.0,
        'density': 0.9978,
        'pH': 3.51,
        'sulphates': 0.56,
        'alcohol': 9.4
    }, {
        'fixed acidity': 8.1,
        'volatile acidity': 0.56,
        'citric acid': 0.28,
        'residual sugar': 1.7,
        'chlorides': 0.068,
        'free sulfur dioxide': 15.0,
        'total sulfur dioxide': 38.0,
        'density': 0.9958,
        'pH': 3.44,
        'sulphates': 0.68,
        'alcohol': 10.5
    }])
    
    print("New wine data for prediction:")
    print(new_wines)
    
    # Make predictions
    try:
        predictions = predict_wine_quality(
            'model/best_model_random_forest.joblib',  # Update this based on your best model
            'model/training_params.joblib',
            new_wines
        )
        
        print("\n🎯 Prediction Results:")
        for i, pred in enumerate(predictions):
            print(f"Wine {i+1}: Predicted Quality = {pred:.2f}")
            
    except FileNotFoundError:
        print("\n❌ Model files not found. Please run train_enhanced.py first.")
    except Exception as e:
        print(f"\n❌ Error during prediction: {e}")

if __name__ == "__main__":
    main()