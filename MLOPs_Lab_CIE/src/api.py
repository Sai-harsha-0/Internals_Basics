from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib

app = FastAPI()

model = joblib.load("models/best_model.pkl")

class InputData(BaseModel):
    slope_degrees: float = Field(..., ge=5, le=45)
    rainfall_mm: float = Field(..., ge=50, le=500)
    soil_depth_m: float = Field(..., ge=0.5, le=5)
    vegetation_index: float = Field(..., ge=0.1, le=0.9)

@app.get("/ping")
def health():
    return {"status": "healthy", "model_loaded": True}

@app.post("/score")
def predict(data: InputData):
    values = [[
        data.slope_degrees,
        data.rainfall_mm,
        data.soil_depth_m,
        data.vegetation_index
    ]]
    pred = model.predict(values)[0]
    return {"prediction": pred}