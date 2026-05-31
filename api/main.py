"""
FASTAPI — HOUSE PRICE PREDICTION API
With custom beautiful frontend UI
"""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, validator
import numpy as np
import joblib
import json
import os
from datetime import datetime

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")
META_PATH  = os.path.join(BASE_DIR, "models", "model_meta.json")
STATIC_DIR = os.path.join(BASE_DIR, "api", "static")

model = joblib.load(MODEL_PATH)
with open(META_PATH) as f:
    meta = json.load(f)

FEATURES       = meta["features"]
CITY_PRICE_MAP = meta["city_price_map"]
CONDITION_MAP  = meta["condition_map"]

app = FastAPI(
    title="House Price Prediction API",
    description="ML-powered property valuation — LightGBM model with R²=0.96",
    version="1.0.0"
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

class PropertyInput(BaseModel):
    Location: str = Field(..., example="CityB")
    Size: float = Field(..., ge=100, le=10000, example=1500)
    Bedrooms: int = Field(..., ge=0, le=10, example=3)
    Bathrooms: int = Field(..., ge=0, le=10, example=2)
    Year_Built: int = Field(..., ge=1800, le=2024, example=1995)
    Condition: str = Field(..., example="Good")
    Sale_Year: int = Field(default_factory=lambda: datetime.now().year, ge=2000, le=2030, example=2024)
    Sale_Month: int = Field(default_factory=lambda: datetime.now().month, ge=1, le=12, example=6)

    @validator('Location')
    def validate_location(cls, v):
        valid = list(CITY_PRICE_MAP.keys())
        if v not in valid:
            raise ValueError(f"Location must be one of: {valid}")
        return v

    @validator('Condition')
    def validate_condition(cls, v):
        valid = list(CONDITION_MAP.keys())
        if v not in valid:
            raise ValueError(f"Condition must be one of: {valid}")
        return v

class PredictionResponse(BaseModel):
    predicted_price: float
    predicted_price_text: str
    input_summary: dict
    model_used: str
    confidence_note: str

def build_features(prop: PropertyInput) -> list:
    property_age     = max(0, prop.Sale_Year - prop.Year_Built)
    condition_score  = CONDITION_MAP[prop.Condition]
    location_encoded = CITY_PRICE_MAP.get(prop.Location, 490000)
    total_rooms      = prop.Bedrooms + prop.Bathrooms
    is_new_condition = 1.0 if prop.Condition == 'New' else 0.0
    is_small         = 1.0 if prop.Size < 1200 else 0.0
    return [prop.Size, prop.Bedrooms, prop.Bathrooms, property_age,
            condition_score, location_encoded, total_rooms,
            prop.Sale_Year, prop.Sale_Month, is_new_condition, is_small]

@app.get("/", include_in_schema=False)
def root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

@app.get("/health", tags=["General"])
def health_check():
    return {"status": "ok", "model_loaded": model is not None, "model_name": meta["model_name"]}

@app.get("/model-info", tags=["Model"])
def model_info():
    return {
        "model_name": meta["model_name"],
        "r2_validation": meta["r2_validation"],
        "mae_validation": f"${meta['mae_validation']:,.0f}",
        "cv_r2_mean": meta["cv_r2_mean"],
        "cv_r2_std": meta["cv_r2_std"],
        "features_used": meta["features"],
        "target_transform": meta["target_transform"]
    }

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_price(prop: PropertyInput):
    try:
        features = build_features(prop)
        features_array = np.array(features).reshape(1, -1)
        log_pred = model.predict(features_array)[0]
        predicted_price = float(np.expm1(log_pred))
        price_text = f"${predicted_price:,.0f}"
        return PredictionResponse(
            predicted_price=round(predicted_price, 2),
            predicted_price_text=price_text,
            input_summary={
                "Location": prop.Location,
                "Size": f"{prop.Size:,} sq ft",
                "Bedrooms": prop.Bedrooms,
                "Bathrooms": prop.Bathrooms,
                "Year_Built": prop.Year_Built,
                "Property_Age": f"{max(0, prop.Sale_Year - prop.Year_Built)} years",
                "Condition": prop.Condition,
                "Sale_Date": f"{prop.Sale_Month}/{prop.Sale_Year}",
            },
            model_used=meta["model_name"],
            confidence_note=f"Model R²={meta['r2_validation']}, avg error ≈ ${meta['mae_validation']:,.0f}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(properties: list[PropertyInput]):
    if len(properties) > 50:
        raise HTTPException(status_code=400, detail="Max 50 properties per batch request")
    results = []
    for i, prop in enumerate(properties):
        try:
            features = build_features(prop)
            features_array = np.array(features).reshape(1, -1)
            log_pred = model.predict(features_array)[0]
            predicted_price = float(np.expm1(log_pred))
            results.append({"index": i, "location": prop.Location,
                            "predicted_price": round(predicted_price, 2),
                            "price_text": f"${predicted_price:,.0f}", "status": "success"})
        except Exception as e:
            results.append({"index": i, "status": "error", "detail": str(e)})
    return {"total_requested": len(properties),
            "total_succeeded": sum(1 for r in results if r["status"] == "success"),
            "predictions": results}
