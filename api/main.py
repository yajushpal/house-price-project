"""
STEP 3: FASTAPI — HOUSE PRICE PREDICTION API
=============================================
Run this file to start the API server:
    uvicorn api.main:app --reload --port 8000

Then open in browser:
    http://localhost:8000/docs   ← Interactive Swagger UI (use this to demo!)
    http://localhost:8000        ← Welcome message
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
import numpy as np
import joblib
import json
import os
from datetime import datetime

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")
META_PATH  = os.path.join(BASE_DIR, "models", "model_meta.json")

# ── Load model and metadata on startup ────────────────────────────────────
model = joblib.load(MODEL_PATH)
with open(META_PATH) as f:
    meta = json.load(f)

FEATURES       = meta["features"]
CITY_PRICE_MAP = meta["city_price_map"]
CONDITION_MAP  = meta["condition_map"]

# ══════════════════════════════════════════════════════════════════════════
# FastAPI App Setup
# ══════════════════════════════════════════════════════════════════════════
app = FastAPI(
    title="🏠 House Price Prediction API",
    description="""
    Predict the sale price of a residential property using a Machine Learning model.
    
    ## How to use
    Send a POST request to `/predict` with property details.
    The model will return the **predicted sale price** in USD.
    
    ## Model Info
    - Algorithm: Gradient Boosting (XGBoost / LightGBM / Random Forest)
    - Trained on: ~1000 real estate transactions
    - Features: Size, Bedrooms, Bathrooms, Condition, Location, Age, Sale Date
    """,
    version="1.0.0"
)

# ══════════════════════════════════════════════════════════════════════════
# Request Schema — what the user must send
# ══════════════════════════════════════════════════════════════════════════
class PropertyInput(BaseModel):
    """
    Details of a property to predict the price for.
    """
    Location: str = Field(
        ...,
        description="City where the property is located",
        example="CityA"
    )
    Size: float = Field(
        ...,
        description="Size of the property in square feet",
        ge=100, le=10000,
        example=2000
    )
    Bedrooms: int = Field(
        ...,
        description="Number of bedrooms",
        ge=0, le=10,
        example=3
    )
    Bathrooms: int = Field(
        ...,
        description="Number of bathrooms",
        ge=0, le=10,
        example=2
    )
    Year_Built: int = Field(
        ...,
        description="Year the property was built",
        ge=1800, le=2024,
        example=1995
    )
    Condition: str = Field(
        ...,
        description="Condition of the property: Poor, Fair, Good, or New",
        example="Good"
    )
    Sale_Year: int = Field(
        default_factory=lambda: datetime.now().year,
        description="Year of the sale (defaults to current year)",
        ge=2000, le=2030,
        example=2024
    )
    Sale_Month: int = Field(
        default_factory=lambda: datetime.now().month,
        description="Month of the sale (1–12, defaults to current month)",
        ge=1, le=12,
        example=6
    )

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

# ══════════════════════════════════════════════════════════════════════════
# Response Schema — what the API returns
# ══════════════════════════════════════════════════════════════════════════
class PredictionResponse(BaseModel):
    predicted_price:      float
    predicted_price_text: str
    input_summary:        dict
    model_used:           str
    confidence_note:      str

# ══════════════════════════════════════════════════════════════════════════
# Helper: Build Feature Vector
# ══════════════════════════════════════════════════════════════════════════
def build_features(prop: PropertyInput) -> list:
    """Convert input to the exact feature vector the model expects."""
    property_age     = max(0, prop.Sale_Year - prop.Year_Built)
    condition_score  = CONDITION_MAP[prop.Condition]
    location_encoded = CITY_PRICE_MAP.get(prop.Location, 490000)
    total_rooms      = prop.Bedrooms + prop.Bathrooms
    is_new_condition = 1.0 if prop.Condition == 'New' else 0.0
    is_small         = 1.0 if prop.Size < 1200 else 0.0

    feature_vector = [
        prop.Size,
        prop.Bedrooms,
        prop.Bathrooms,
        property_age,
        condition_score,
        location_encoded,
        total_rooms,
        prop.Sale_Year,
        prop.Sale_Month,
        is_new_condition,
        is_small
    ]
    return feature_vector

# ══════════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════════

@app.get("/", tags=["General"])
def root():
    """Welcome message. Confirms the API is running."""
    return {
        "message"     : "🏠 House Price Prediction API is running!",
        "docs"        : "Visit /docs to use the interactive interface",
        "model"       : meta["model_name"],
        "model_r2"    : meta["r2_validation"],
        "model_mae"   : f"${meta['mae_validation']:,.0f}",
        "valid_cities": list(CITY_PRICE_MAP.keys()),
        "valid_conditions": list(CONDITION_MAP.keys())
    }


@app.get("/health", tags=["General"])
def health_check():
    """Health check endpoint. Returns status: ok if the model is loaded."""
    return {
        "status"      : "ok",
        "model_loaded": model is not None,
        "model_name"  : meta["model_name"]
    }


@app.get("/model-info", tags=["Model"])
def model_info():
    """Returns model performance metrics and training details."""
    return {
        "model_name"       : meta["model_name"],
        "r2_validation"    : meta["r2_validation"],
        "mae_validation"   : f"${meta['mae_validation']:,.0f}",
        "cv_r2_mean"       : meta["cv_r2_mean"],
        "cv_r2_std"        : meta["cv_r2_std"],
        "features_used"    : meta["features"],
        "target_transform" : meta["target_transform"],
        "description"      : "R² close to 1.0 = very accurate. MAE = average prediction error."
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_price(prop: PropertyInput):
    """
    ## Predict House Price
    
    Send property details and get back a predicted sale price.
    
    ### Example Request:
    ```json
    {
      "Location": "CityB",
      "Size": 1500,
      "Bedrooms": 3,
      "Bathrooms": 2,
      "Year_Built": 1990,
      "Condition": "Good",
      "Sale_Year": 2024,
      "Sale_Month": 6
    }
    ```
    """
    try:
        features = build_features(prop)
        features_array = np.array(features).reshape(1, -1)

        # Model predicts log(price), so we convert back with exp()
        log_pred = model.predict(features_array)[0]
        predicted_price = float(np.expm1(log_pred))

        # Format nicely
        price_text = f"${predicted_price:,.0f}"

        return PredictionResponse(
            predicted_price      = round(predicted_price, 2),
            predicted_price_text = price_text,
            input_summary        = {
                "Location"    : prop.Location,
                "Size"        : f"{prop.Size:,} sq ft",
                "Bedrooms"    : prop.Bedrooms,
                "Bathrooms"   : prop.Bathrooms,
                "Year_Built"  : prop.Year_Built,
                "Property_Age": f"{max(0, prop.Sale_Year - prop.Year_Built)} years",
                "Condition"   : prop.Condition,
                "Sale_Date"   : f"{prop.Sale_Month}/{prop.Sale_Year}",
            },
            model_used      = meta["model_name"],
            confidence_note = f"Model R²={meta['r2_validation']}, avg error ≈ ${meta['mae_validation']:,.0f}"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(properties: list[PropertyInput]):
    """
    ## Batch Predict — Multiple Properties at Once
    Send a list of properties and get predictions for all of them.
    Max 50 properties per request.
    """
    if len(properties) > 50:
        raise HTTPException(status_code=400, detail="Max 50 properties per batch request")

    results = []
    for i, prop in enumerate(properties):
        try:
            features = build_features(prop)
            features_array = np.array(features).reshape(1, -1)
            log_pred = model.predict(features_array)[0]
            predicted_price = float(np.expm1(log_pred))
            results.append({
                "index"           : i,
                "location"        : prop.Location,
                "predicted_price" : round(predicted_price, 2),
                "price_text"      : f"${predicted_price:,.0f}",
                "status"          : "success"
            })
        except Exception as e:
            results.append({"index": i, "status": "error", "detail": str(e)})

    return {
        "total_requested": len(properties),
        "total_succeeded": sum(1 for r in results if r["status"] == "success"),
        "predictions"    : results
    }
