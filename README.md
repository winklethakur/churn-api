# 📉 Customer Segmentation & Retention Analysis
### Telecom Churn Prediction · End-to-End ML Project
> Predict which telecom customers will churn, segment them into actionable groups, and serve real-time predictions via a production-ready REST API — all in one project.

---

## Project Structure

```
├── notebook/
│   ├── Customer_Churn_Prediction.ipynb   # Full ML pipeline & analysis
│   └── churn_model.pkl                   # Serialised best model
│
└── api/
    ├── main.py                           # FastAPI application
    ├── requirements.txt                  # Python dependencies
    ├── Dockerfile                        # Container definition
    └── render.yaml                       # Render deployment config
```

---

## Problem Statement

Customer churn is one of the most costly problems in the telecom industry — acquiring a new customer costs **5–7× more** than retaining an existing one. This project builds a **machine learning system** that:

1. Predicts the probability of a customer churning
2. Segments customers by behaviour (New / Mid-tenure / Loyal)
3. Recommends a targeted retention strategy per customer
4. Exposes all of this through a live REST API

**Dataset:** IBM Telco Customer Churn — 7,043 customers, 20 features

---

## ML Pipeline (Notebook)

### 1. Exploratory Data Analysis
- Class imbalance: ~26% churners vs ~74% non-churners
- Key churn drivers visualised: Contract type, Internet service, Payment method
- Churn rate by tenure, monthly charges, and contract length

### 2. Preprocessing
- Fixed `TotalCharges` type mismatch (raw string → float)
- `OneHotEncoder` for 16 categorical features
- `StandardScaler` for 3 numerical features
- Sklearn `Pipeline` + `ColumnTransformer` to prevent data leakage

### 3. Class Imbalance Handling
| Technique | Applied To |
|-----------|-----------|
| `class_weight='balanced'` | Logistic Regression, Random Forest |
| No native weight param — handled via balanced training data ratio | Gradient Boosting |

> Note: `scale_pos_weight` is an XGBoost parameter. Sklearn's `GradientBoostingClassifier` does not support it — class imbalance is addressed upstream via the balanced split strategy.

### 4. Models Trained & Compared

| Model | ROC-AUC | Churn Recall |
|-------|---------|-------------|
| Logistic Regression | ~0.835 | ~0.80 |
| Random Forest | ~0.813 | ~0.47 |
| **Gradient Boosting** | **~0.840** | **~0.54** |

> **Gradient Boosting selected** — best AUC overall. Logistic Regression had the highest recall but Gradient Boosting gave the best generalisation across all metrics.

### 5. Feature Importance
Top churn predictors (from Gradient Boosting):
- **Contract type** — single biggest driver; month-to-month customers churn at 3× the rate of annual contract customers
- **Tenure** — shorter tenure = higher churn risk
- **Monthly charges** — higher bills correlate with churn
- **Internet service** — Fiber optic customers churn significantly more than DSL
- **OnlineSecurity / TechSupport** — customers without these churn more (upsell opportunities)

### 6. Customer Segmentation (KMeans Clustering)
Clustered customers into **3 behavioural segments** using tenure, monthly charges, and total charges. Elbow method used to select k=3.

| Segment | Avg Tenure | Churn Risk | Characteristics |
|---------|-----------|-----------|----------------|
| New Customer | ≤ 12 months | 🔴 Very High (~47%) | Short tenure, high monthly charges, often month-to-month |
| Mid-tenure | 13–36 months | 🟡 Low (~15%) | Moderate tenure, mixed contract types |
| Loyal Customer | 37+ months | 🟢 Very Low (~12%) | Long tenure, lower charges, often on annual contracts |

---

## API — Live Deployment

The trained model is served via **FastAPI**, containerised with **Docker**, and deployed on **Render**.

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `GET` | `/health` | Service status |
| `POST` | `/predict` | Predict churn for a single customer |
| `POST` | `/predict/batch` | Batch predictions (up to 100 customers) |

### Sample Request

```bash
curl -X POST "https://<your-render-url>/predict" \
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

### Sample Response

```json
{
  "churn_prediction": 1,
  "churn_label": "Will churn",
  "churn_probability": 0.8241,
  "risk_level": "High",
  "segment": "New customer",
  "retention_strategy": "Offer onboarding discount or switch-to-annual-contract incentive. High priority if churn risk > 50%."
}
```

### Risk Levels

| Probability | Risk Level |
|-------------|-----------|
| ≥ 0.70 | 🔴 High |
| 0.40 – 0.69 | 🟡 Medium |
| < 0.40 | 🟢 Low |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Data & ML | Python, Pandas, NumPy, scikit-learn, XGBoost |
| Visualisation | Matplotlib, Seaborn |
| API Framework | FastAPI + Pydantic v2 |
| Server | Uvicorn (ASGI) |
| Containerisation | Docker |
| Deployment | Render |

---

## Run Locally

### Option A — Python

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>/api

pip install -r requirements.txt
uvicorn main:app --reload
```

Visit `http://127.0.0.1:8000/docs` for the interactive Swagger UI.

### Option B — Docker

```bash
cd api
docker build -t churn-api .
docker run -p 8000:8000 -e PORT=8000 churn-api
```

---

## Key ML Concepts Demonstrated

- **End-to-end pipeline** — raw data → trained model → deployed API
- **Handling class imbalance** — without resampling (SMOTE); instead using cost-sensitive learning
- **Sklearn Pipelines** — preprocessor + model in one object, preventing data leakage
- **Model evaluation** — AUC-ROC, precision, recall, F1 (not just accuracy)
- **Unsupervised segmentation** — KMeans clustering with elbow method for k selection
- **Production deployment** — Dockerised FastAPI with batch prediction support

---

## Notes

- Model is a `scikit-learn` Pipeline serialised with `pickle` — the same preprocessing steps used during training are automatically applied at inference time.
- Batch endpoint accepts up to **100 customers** per request.
- CORS is enabled for all origins — suitable for frontend integration.

---