import os
from fastapi import FastAPI, HTTPException
import mlflow.pyfunc
from pydantic import BaseModel
import numpy as np

mlflow.set_tracking_uri(uri=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
app = FastAPI()
model = None

def load_model(model_name: str, model_version: str = "latest"):
    global model
    try:
        model_uri = f"models:/{model_name}/{model_version}"
        model = mlflow.pyfunc.load_model(model_uri)
        print(f"Model loaded from MLflow: {model_uri}")
    except Exception as e:
        print(f"Failed to load model from MLflow: {e}")
        raise RuntimeError("No model available.")

load_model(model_name="tracking-iris-model")

class PredictionRequest(BaseModel):
    features: list[list[float]]

class UpdateModelRequest(BaseModel):
    model_name: str
    model_version: str = "latest"

@app.post("/predict")
def predict(request: PredictionRequest):
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    features_array = np.array(request.features, dtype=np.float64)
    predictions = model.predict(features_array)
    return {"predictions": predictions.tolist()}


@app.post("/update-model")
def update_model(request: UpdateModelRequest):
    try:
        load_model(request.model_name, request.model_version)
        return {
            "status": f"Model updated to {request.model_name} (version: {request.model_version})."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "Welcome to the MLflow Model Serving API!"}