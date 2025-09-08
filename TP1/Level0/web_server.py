from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os

app = FastAPI()

MODEL_PATH = os.getenv("MODEL_PATH", "regression.joblib")
model = joblib.load(MODEL_PATH)

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

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload_flag = os.getenv("RELOAD", "false").lower() == "true"
    uvicorn.run("web_server:app", host=host, port=port, reload=reload_flag)