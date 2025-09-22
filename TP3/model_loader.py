import joblib
import os
import json
import numpy as np

def load_house_price_model(model_path='house_price_model.joblib'):
    """
    Load the house price prediction model from disk.
    
    Args:
        model_path (str): Path to the saved model file
        
    Returns:
        sklearn model: The loaded machine learning model
        
    Raises:
        FileNotFoundError: If the model file doesn't exist
        Exception: If there's an error loading the model
    """
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        model = joblib.load(model_path)
        print(f"Model successfully loaded from {model_path}")
        return model
        
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def get_model_info(model):
    """
    Get basic information about the loaded model.
    
    Args:
        model: The loaded sklearn model
        
    Returns:
        dict: Model information
    """
    try:
        info = {
            "model_type": type(model).__name__,
            "n_features": model.n_features_in_,
            "coefficients": model.coef_.tolist() if hasattr(model, 'coef_') else None,
            "intercept": float(model.intercept_) if hasattr(model, 'intercept_') else None
        }
        return info
    except Exception as e:
        print(f"Error getting model info: {e}")
        return {}

def test_model_predictions():
    """Test the model with sample data"""
    
    # Load the model
    model = load_house_price_model()
    
    # Load sample data
    with open('sample_house_data.json', 'r') as f:
        sample_data = json.load(f)
    
    print("Testing model predictions:")
    print("-" * 50)
    
    # Test with each sample
    for i, house_data in enumerate(sample_data):
        # Convert to the format expected by the model
        features = np.array([[
            house_data['square_feet'],
            house_data['bedrooms'], 
            house_data['bathrooms'],
            house_data['age']
        ]])
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        print(f"House {i+1}:")
        print(f"  Features: {house_data}")
        print(f"  Predicted Price: ${prediction:,.2f}")
        print()
    
    # Test with custom data
    print("Testing with custom house data:")
    print("-" * 50)
    
    custom_houses = [
        {"square_feet": 2000, "bedrooms": 3, "bathrooms": 2, "age": 10},
        {"square_feet": 1500, "bedrooms": 2, "bathrooms": 1, "age": 25},
        {"square_feet": 3000, "bedrooms": 4, "bathrooms": 3, "age": 5}
    ]
    
    for i, house_data in enumerate(custom_houses):
        features = np.array([[
            house_data['square_feet'],
            house_data['bedrooms'], 
            house_data['bathrooms'],
            house_data['age']
        ]])
        
        prediction = model.predict(features)[0]
        
        print(f"Custom House {i+1}:")
        print(f"  Features: {house_data}")
        print(f"  Predicted Price: ${prediction:,.2f}")
        print()
        
if __name__ == "__main__":
    # Test the function
    try:
        model = load_house_price_model()
        info = get_model_info(model)
        print("Model information:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        test_model_predictions()
    except Exception as e:
        print(f"Failed to load model: {e}")

