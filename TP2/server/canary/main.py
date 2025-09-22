import os
import random
from fastapi import FastAPI, HTTPException
import mlflow.pyfunc
from pydantic import BaseModel
import numpy as np

mlflow.set_tracking_uri(uri=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
app = FastAPI()

# Canary deployment models
current_model = None
next_model = None

# Canary deployment probability of using current model
canary_probability = float(os.getenv("CANARY_PROBABILITY", "0.9"))

def load_model(model_name: str, model_version: str = "latest", target: str = "both"):
    """
    Load a model for canary deployment.
    
    Args:
        model_name: Name of the model to load
        model_version: Version of the model to load (default: "latest")
        target: Which model to update - "current", "next", or "both" (default: "both")
    """
    global current_model, next_model
    try:
        model_uri = f"models:/{model_name}/{model_version}"
        loaded_model = mlflow.pyfunc.load_model(model_uri)
        
        if target == "current" or target == "both":
            current_model = loaded_model
            print(f"Current model loaded from MLflow: {model_uri}")
        
        if target == "next" or target == "both":
            next_model = loaded_model
            print(f"Next model loaded from MLflow: {model_uri}")
            
    except Exception as e:
        print(f"Failed to load model from MLflow: {e}")
        raise RuntimeError("No model available.")

# Initialize both current and next models with the same model at startup
load_model(model_name="tracking-iris-model", target="both")

class PredictionRequest(BaseModel):
    features: list[list[float]]

class UpdateModelRequest(BaseModel):
    model_name: str
    model_version: str = "latest"

@app.post("/predict")
def predict(request: PredictionRequest):
    if not current_model or not next_model:
        raise HTTPException(status_code=500, detail="Models not loaded.")

    # Canary deployment: choose model based on probability
    # p probability for current model and (1-p) probability for next model
    use_current = random.random() < canary_probability
    selected_model = current_model if use_current else next_model
    model_type = "current" if use_current else "next"
    
    features_array = np.array(request.features, dtype=np.float64)
    predictions = selected_model.predict(features_array)
    
    return {
        "predictions": predictions.tolist(),
        "model_used": model_type,
        "canary_probability": canary_probability
    }

@app.post("/update-model")
def update_model(request: UpdateModelRequest):
    """Update the next model for canary deployment."""
    try:
        load_model(request.model_name, request.model_version, target="next")
        return {
            "status": f"Next model updated to {request.model_name} (version: {request.model_version}).",
            "message": "Use /accept-next-model to promote this model to current."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/accept-next-model")
def accept_next_model():
    """Promote the next model to become the current model."""
    global current_model
    if not next_model:
        raise HTTPException(status_code=500, detail="No next model available to promote.")
    
    current_model = next_model
    return {
        "status": "Next model has been promoted to current model.",
        "message": "Both current and next models are now the same. Update next model to continue canary deployment."
    }

class CanaryConfigRequest(BaseModel):
    probability: float

@app.post("/configure-canary")
def configure_canary(request: CanaryConfigRequest):
    """Configure the canary deployment probability."""
    global canary_probability
    if not 0 <= request.probability <= 1:
        raise HTTPException(status_code=400, detail="Probability must be between 0 and 1.")
    
    canary_probability = request.probability
    return {
        "status": f"Canary probability updated to {canary_probability}",
        "current_probability": canary_probability,
        "message": f"Current model will be used {canary_probability*100:.1f}% of the time, next model {(1-canary_probability)*100:.1f}% of the time."
    }

@app.get("/canary-status")
def canary_status():
    """Get current canary deployment status."""
    return {
        "canary_probability": canary_probability,
        "current_model_loaded": current_model is not None,
        "next_model_loaded": next_model is not None,
        "models_are_same": current_model is next_model,
        "current_usage_percentage": f"{canary_probability*100:.1f}%",
        "next_usage_percentage": f"{(1-canary_probability)*100:.1f}%"
    }

@app.get("/")
def read_root():
    return {"message": "Welcome to the MLflow Model Serving API!"}