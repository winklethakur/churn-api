import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal
 
app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predicts churn probability and customer segment for Telco customers.",
    version="1.0.0"
)
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
 
# ── Load model ─────────────────────────────────────────────────────────────────
with open("churn_model.pkl", "rb") as f:
    model = pickle.load(f)
 
print("Model loaded successfully!")
 
# ── Request schema ─────────────────────────────────────────────────────────────
class CustomerData(BaseModel):
    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(..., ge=0, le=72)
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ]
    MonthlyCharges: float = Field(..., ge=0)
    TotalCharges: float = Field(..., ge=0)
 
    class Config:
        json_schema_extra = {
            "example": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 2,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 70.70,
                "TotalCharges": 151.65
            }
        }
 
# ── Segment & strategy ─────────────────────────────────────────────────────────
def get_segment(tenure: int) -> str:
    if tenure <= 12:
        return "New customer"
    elif tenure <= 36:
        return "Mid-tenure"
    else:
        return "Loyal customer"
 
def get_retention_strategy(segment: str) -> str:
    strategies = {
        "New customer": "Offer onboarding discount or switch-to-annual-contract incentive. High priority if churn risk > 50%.",
        "Mid-tenure": "Loyalty rewards and upsell add-ons (OnlineSecurity, TechSupport). Check-in survey recommended.",
        "Loyal customer": "VIP treatment, referral programme, early access to new products. Low intervention needed."
    }
    return strategies[segment]
 
# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "Churn Prediction API is running."}
 
@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy"}
 
@app.post("/predict", tags=["Prediction"])
def predict(customer: CustomerData):
    try:
        df = pd.DataFrame([customer.model_dump()])
        churn_prob = float(model.predict_proba(df)[0][1])
        churn_pred = int(churn_prob >= 0.5)
        segment = get_segment(customer.tenure)
        risk_level = (
            "High"   if churn_prob >= 0.7 else
            "Medium" if churn_prob >= 0.4 else
            "Low"
        )
        return {
            "churn_prediction": churn_pred,
            "churn_label": "Will churn" if churn_pred else "Will not churn",
            "churn_probability": round(churn_prob, 4),
            "risk_level": risk_level,
            "segment": segment,
            "retention_strategy": get_retention_strategy(segment)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
 
@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(customers: list[CustomerData]):
    if len(customers) > 100:
        raise HTTPException(status_code=400, detail="Max 100 customers per batch.")
    return [predict(c) for c in customers]
 