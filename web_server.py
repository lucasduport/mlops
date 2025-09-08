from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

model = joblib.load('regression.joblib')

class PredictionRequest(BaseModel):
    size: float
    bedrooms: int
    garden: int

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/predict")
async def predict_price(request: PredictionRequest):
    input_data = [[request.size, request.bedrooms, request.garden]]
    prediction = model.predict(input_data)
    return {"predicted_price": round(prediction[0], 2)}