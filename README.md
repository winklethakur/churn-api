# Churn Prediction API

FastAPI service for predicting customer churn and returning segment + retention strategy.

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/health` | Health check |
| POST | `/predict` | Predict churn for one customer |
| POST | `/predict/batch` | Predict churn for up to 100 customers |
| GET | `/docs` | Swagger UI (interactive docs) |

## Run locally

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Open http://localhost:8000/docs to test via Swagger UI.

## Deploy to Render (free tier)

1. Push this folder to a GitHub repository
2. Go to https://render.com → New → Web Service
3. Connect your GitHub repo
4. Render auto-detects `render.yaml` — just click **Deploy**
5. Your API will be live at `https://your-app-name.onrender.com`

**Important:** Upload `churn_model.pkl` to the repo root alongside `main.py`.

## Deploy to Railway

1. Push this folder to a GitHub repository
2. Go to https://railway.app → New Project → Deploy from GitHub
3. Railway auto-detects `Procfile` — click **Deploy**
4. Your API will be live at the URL Railway provides

## Example request

```bash
curl -X POST "https://your-app.onrender.com/predict" \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

## Example response

```json
{
  "churn_prediction": 1,
  "churn_label": "Will churn",
  "churn_probability": 0.8243,
  "risk_level": "High",
  "segment": "New customer",
  "retention_strategy": "Offer onboarding discount or switch-to-annual-contract incentive. High priority if churn risk > 50%."
}
```

## Project structure

```
churn_api/
├── main.py              # FastAPI app
├── churn_model.pkl      # Trained model (add this yourself)
├── requirements.txt     # Python dependencies
├── render.yaml          # Render deployment config
├── Procfile             # Railway deployment config
├── .gitignore
└── README.md
```
