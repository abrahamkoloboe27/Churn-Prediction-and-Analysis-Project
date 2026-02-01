# 🚀 Guide de Déploiement et Recommandations

## Table des Matières
- [Introduction](#introduction)
- [Du Notebook à la Production](#du-notebook-à-la-production)
- [Sauvegarder et Versionner les Modèles](#sauvegarder-et-versionner-les-modèles)
- [Créer une API de Prédiction](#créer-une-api-de-prédiction)
- [Déploiement avec Docker](#déploiement-avec-docker)
- [Monitoring en Production](#monitoring-en-production)
- [MLOps Best Practices](#mlops-best-practices)
- [Système de Recommandations](#système-de-recommandations)

## Introduction

Le déploiement d'un modèle ML en production est une étape critique qui transforme un notebook expérimental en un système fiable et utilisable. Ce guide couvre les meilleures pratiques pour déployer et utiliser efficacement des modèles ML.

---

## Du Notebook à la Production

### Workflow Typique

```
Notebook Jupyter → Code Modulaire → API → Containerisation → Déploiement → Monitoring
```

### Étape 1 : Refactorer le Code du Notebook

#### ❌ Code Notebook (Expérimental)

```python
# Dans un notebook
df = pd.read_csv('data.csv')
df = df.fillna(0)
X = df.drop('target', axis=1)
y = df['target']
model = RandomForestClassifier()
model.fit(X, y)
```

#### ✅ Code Production (Modulaire)

```python
# src/data_processing.py
class DataProcessor:
    """Traitement des données pour le modèle"""
    
    def __init__(self, config):
        self.config = config
        self.scaler = None
        self.encoder = None
    
    def fit(self, df):
        """Apprendre les transformations"""
        # Fit scaler sur colonnes numériques
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        self.scaler = StandardScaler()
        self.scaler.fit(df[numeric_cols])
        
        # Fit encoder sur colonnes catégorielles
        cat_cols = df.select_dtypes(include=['object']).columns
        self.encoder = OneHotEncoder(handle_unknown='ignore')
        self.encoder.fit(df[cat_cols])
        
        return self
    
    def transform(self, df):
        """Appliquer les transformations"""
        df = df.copy()
        
        # Handle missing values
        df = self._handle_missing(df)
        
        # Scale numerical
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = self.scaler.transform(df[numeric_cols])
        
        # Encode categorical
        cat_cols = df.select_dtypes(include=['object']).columns
        encoded = self.encoder.transform(df[cat_cols])
        encoded_df = pd.DataFrame(
            encoded.toarray(),
            columns=self.encoder.get_feature_names_out(),
            index=df.index
        )
        
        # Combine
        df = pd.concat([df[numeric_cols], encoded_df], axis=1)
        
        return df
    
    def _handle_missing(self, df):
        """Gérer les valeurs manquantes"""
        for col in df.columns:
            if df[col].isnull().any():
                if df[col].dtype in ['int64', 'float64']:
                    df[col].fillna(df[col].median(), inplace=True)
                else:
                    df[col].fillna('Missing', inplace=True)
        return df

# src/model.py
class ChurnPredictor:
    """Modèle de prédiction de churn"""
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.preprocessor = DataProcessor(config)
    
    def train(self, X_train, y_train):
        """Entraîner le modèle"""
        # Fit preprocessor
        self.preprocessor.fit(X_train)
        
        # Transform data
        X_transformed = self.preprocessor.transform(X_train)
        
        # Train model
        self.model = RandomForestClassifier(**self.config['model_params'])
        self.model.fit(X_transformed, y_train)
        
        return self
    
    def predict(self, X):
        """Prédire"""
        X_transformed = self.preprocessor.transform(X)
        return self.model.predict(X_transformed)
    
    def predict_proba(self, X):
        """Prédire probabilités"""
        X_transformed = self.preprocessor.transform(X)
        return self.model.predict_proba(X_transformed)

# src/config.py
config = {
    'model_params': {
        'n_estimators': 100,
        'max_depth': 20,
        'random_state': 42,
        'class_weight': 'balanced'
    },
    'data_path': 'data/churn.csv',
    'model_path': 'models/churn_model.pkl'
}
```

---

## Sauvegarder et Versionner les Modèles

### Sauvegarde avec Joblib/Pickle

```python
import joblib
from datetime import datetime

# Sauvegarder le modèle
def save_model(model, model_path, metadata=None):
    """Sauvegarder un modèle avec métadonnées"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Créer un objet avec modèle et métadonnées
    model_artifact = {
        'model': model,
        'preprocessor': model.preprocessor,
        'metadata': {
            'timestamp': timestamp,
            'model_type': type(model.model).__name__,
            'version': '1.0.0',
            **(metadata or {})
        }
    }
    
    # Sauvegarder
    joblib.dump(model_artifact, f"{model_path}_{timestamp}.pkl")
    # Aussi sauvegarder comme 'latest'
    joblib.dump(model_artifact, f"{model_path}_latest.pkl")
    
    print(f"✓ Model saved: {model_path}_{timestamp}.pkl")
    return model_artifact

# Charger le modèle
def load_model(model_path):
    """Charger un modèle"""
    model_artifact = joblib.load(model_path)
    print(f"✓ Model loaded from {model_path}")
    print(f"  Version: {model_artifact['metadata']['version']}")
    print(f"  Timestamp: {model_artifact['metadata']['timestamp']}")
    return model_artifact

# Utilisation
model = ChurnPredictor(config)
model.train(X_train, y_train)

metadata = {
    'accuracy': 0.85,
    'recall': 0.78,
    'f1_score': 0.81,
    'training_samples': len(X_train)
}

save_model(model, 'models/churn_model', metadata)
```

### Versioning avec MLflow

```python
import mlflow
import mlflow.sklearn

# Configuration MLflow
mlflow.set_experiment("churn_prediction")

# Training avec logging
with mlflow.start_run(run_name="random_forest_v1"):
    
    # Log parameters
    mlflow.log_params(config['model_params'])
    
    # Train model
    model = ChurnPredictor(config)
    model.train(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Log metrics
    from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    mlflow.log_metrics(metrics)
    
    # Log model
    mlflow.sklearn.log_model(model.model, "model")
    
    # Log artifacts
    mlflow.log_artifact("config.py")
    
    print(f"✓ Run ID: {mlflow.active_run().info.run_id}")
```

---

## Créer une API de Prédiction

### Option 1 : FastAPI (Recommandé)

```python
# api/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from typing import List, Dict

# Charger le modèle au démarrage
model_artifact = joblib.load('models/churn_model_latest.pkl')
model = model_artifact['model']

app = FastAPI(
    title="Churn Prediction API",
    description="API pour prédire le churn des clients",
    version="1.0.0"
)

# Schéma de données d'entrée
class Customer(BaseModel):
    customerID: str
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float
    
    class Config:
        schema_extra = {
            "example": {
                "customerID": "1234-ABCD",
                "gender": "Male",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 12,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 70.0,
                "TotalCharges": 840.0
            }
        }

class PredictionResponse(BaseModel):
    customerID: str
    churn_probability: float
    churn_prediction: str
    risk_level: str
    recommended_actions: List[str]

@app.get("/")
async def root():
    """Health check"""
    return {
        "status": "healthy",
        "model_version": model_artifact['metadata']['version'],
        "model_timestamp": model_artifact['metadata']['timestamp']
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_churn(customer: Customer):
    """Prédire le churn pour un client"""
    
    try:
        # Convertir en DataFrame
        customer_data = pd.DataFrame([customer.dict()])
        customer_id = customer_data['customerID'].values[0]
        customer_data = customer_data.drop('customerID', axis=1)
        
        # Prédire
        churn_proba = model.predict_proba(customer_data)[0, 1]
        churn_pred = "Yes" if churn_proba >= 0.5 else "No"
        
        # Déterminer le niveau de risque
        if churn_proba >= 0.7:
            risk_level = "High"
        elif churn_proba >= 0.4:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        # Générer recommandations
        actions = generate_recommendations(customer, churn_proba)
        
        return PredictionResponse(
            customerID=customer_id,
            churn_probability=round(churn_proba, 3),
            churn_prediction=churn_pred,
            risk_level=risk_level,
            recommended_actions=actions
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch")
async def predict_batch(customers: List[Customer]):
    """Prédiction en lot"""
    
    try:
        results = []
        for customer in customers:
            pred = await predict_churn(customer)
            results.append(pred)
        
        return {
            "total_customers": len(results),
            "high_risk": sum(1 for r in results if r.risk_level == "High"),
            "medium_risk": sum(1 for r in results if r.risk_level == "Medium"),
            "low_risk": sum(1 for r in results if r.risk_level == "Low"),
            "predictions": results
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def generate_recommendations(customer: Customer, churn_proba: float) -> List[str]:
    """Générer des recommandations basées sur le profil"""
    
    actions = []
    
    if customer.Contract == "Month-to-month":
        actions.append("Proposer un contrat annuel avec réduction")
    
    if customer.tenure < 12:
        actions.append("Programme d'onboarding renforcé")
    
    if customer.TechSupport == "No" and customer.InternetService != "No":
        actions.append("Offrir support technique gratuit pendant 3 mois")
    
    if customer.MonthlyCharges > 70:
        actions.append("Analyser possibilités d'optimisation du forfait")
    
    if churn_proba >= 0.7:
        actions.append("URGENT: Contact prioritaire par l'équipe de rétention")
    
    return actions if actions else ["Continuer le suivi standard"]

# Pour lancer: uvicorn api.main:app --reload
```

### Option 2 : Flask (Alternative)

```python
# api/app.py
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Charger le modèle
model_artifact = joblib.load('models/churn_model_latest.pkl')
model = model_artifact['model']

@app.route('/')
def home():
    return jsonify({
        'status': 'healthy',
        'model_version': model_artifact['metadata']['version']
    })

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    # Convertir en DataFrame
    df = pd.DataFrame([data])
    customer_id = df['customerID'].values[0]
    df = df.drop('customerID', axis=1)
    
    # Prédire
    churn_proba = model.predict_proba(df)[0, 1]
    
    return jsonify({
        'customerID': customer_id,
        'churn_probability': float(churn_proba),
        'churn_prediction': 'Yes' if churn_proba >= 0.5 else 'No'
    })

if __name__ == '__main__':
    # Note: Ne jamais utiliser debug=True en production pour des raisons de sécurité
    app.run(host='0.0.0.0', port=5000)
```

---

## Déploiement avec Docker

### Dockerfile

```dockerfile
# Dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY api/ ./api/
COPY models/ ./models/
COPY config.py .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Run the application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### docker-compose.yml

```yaml
version: '3.8'

services:
  churn-api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs
    environment:
      - MODEL_PATH=/app/models/churn_model_latest.pkl
      - LOG_LEVEL=INFO
    restart: unless-stopped
    
  # Optional: Add monitoring
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    
  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-storage:/var/lib/grafana

volumes:
  grafana-storage:
```

### .dockerignore

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv

# Jupyter
.ipynb_checkpoints

# IDE
.vscode/
.idea/

# Data
data/raw/
*.csv
*.xlsx

# Models (except latest)
models/*
!models/*_latest.pkl

# Logs
logs/
*.log

# Git
.git
.gitignore

# Docker
.dockerignore
Dockerfile
docker-compose.yml

# Tests
tests/
.pytest_cache/

# Documentation
docs/
*.md
```

### Commandes Docker

```bash
# Build image
docker build -t churn-prediction-api:latest .

# Run container
docker run -d -p 8000:8000 --name churn-api churn-prediction-api:latest

# Check logs
docker logs -f churn-api

# Test API
curl http://localhost:8000/

# Stop container
docker stop churn-api

# Remove container
docker rm churn-api

# Using docker-compose
docker-compose up -d
docker-compose logs -f
docker-compose down
```

---

## Monitoring en Production

### 1. Logging

```python
# src/utils/logger.py
import logging
import sys
from datetime import datetime

def setup_logger(name, log_file=None, level=logging.INFO):
    """Configurer un logger"""
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# Dans l'API
from src.utils.logger import setup_logger

logger = setup_logger('churn_api', 'logs/api.log')

@app.post("/predict")
async def predict_churn(customer: Customer):
    logger.info(f"Prediction request for customer: {customer.customerID}")
    
    try:
        # ... prediction logic ...
        logger.info(f"Prediction successful: {churn_pred} (prob: {churn_proba:.3f})")
        return response
    
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        raise
```

### 2. Métriques de Performance

```python
# api/monitoring.py
from prometheus_client import Counter, Histogram, Gauge
import time

# Métriques
prediction_counter = Counter(
    'predictions_total',
    'Total number of predictions',
    ['model_version', 'risk_level']
)

prediction_latency = Histogram(
    'prediction_latency_seconds',
    'Time spent processing prediction'
)

model_score_distribution = Histogram(
    'churn_probability_distribution',
    'Distribution of churn probabilities'
)

active_customers = Gauge(
    'active_customers_total',
    'Total number of active customers'
)

# Dans l'API
@app.post("/predict")
async def predict_churn(customer: Customer):
    start_time = time.time()
    
    try:
        # ... prediction ...
        
        # Enregistrer métriques
        prediction_counter.labels(
            model_version=model_artifact['metadata']['version'],
            risk_level=risk_level
        ).inc()
        
        model_score_distribution.observe(churn_proba)
        prediction_latency.observe(time.time() - start_time)
        
        return response
    
    except Exception as e:
        logger.error(f"Error: {e}")
        raise
```

### 3. Model Drift Detection

```python
# src/monitoring/drift_detector.py
from scipy import stats
import numpy as np

class DriftDetector:
    """Détecter le drift des données et du modèle"""
    
    def __init__(self, reference_data, reference_predictions, reference_accuracy=None, reference_recall=None):
        self.reference_data = reference_data
        self.reference_predictions = reference_predictions
        self.reference_accuracy = reference_accuracy
        self.reference_recall = reference_recall
    
    def detect_data_drift(self, new_data, threshold=0.05):
        """Détecter le drift des features"""
        
        drift_detected = {}
        
        for col in self.reference_data.columns:
            if self.reference_data[col].dtype in ['int64', 'float64']:
                # Test de Kolmogorov-Smirnov
                stat, pvalue = stats.ks_2samp(
                    self.reference_data[col].dropna(),
                    new_data[col].dropna()
                )
                
                drift_detected[col] = {
                    'drifted': pvalue < threshold,
                    'pvalue': pvalue,
                    'statistic': stat
                }
        
        return drift_detected
    
    def detect_prediction_drift(self, new_predictions, threshold=0.05):
        """Détecter le drift des prédictions"""
        
        stat, pvalue = stats.ks_2samp(
            self.reference_predictions,
            new_predictions
        )
        
        return {
            'drifted': pvalue < threshold,
            'pvalue': pvalue,
            'statistic': stat
        }
    
    def monitor_performance(self, y_true, y_pred, alert_threshold=0.1):
        """Monitorer la dégradation des performances"""
        
        from sklearn.metrics import accuracy_score, recall_score
        
        current_accuracy = accuracy_score(y_true, y_pred)
        current_recall = recall_score(y_true, y_pred)
        
        # Comparer avec performance de référence
        accuracy_drop = self.reference_accuracy - current_accuracy
        recall_drop = self.reference_recall - current_recall
        
        alerts = []
        
        if accuracy_drop > alert_threshold:
            alerts.append(f"⚠️ Accuracy dropped by {accuracy_drop:.2%}")
        
        if recall_drop > alert_threshold:
            alerts.append(f"⚠️ Recall dropped by {recall_drop:.2%}")
        
        return alerts

# Utilisation
detector = DriftDetector(X_train, y_train_pred)

# Périodiquement
drift_report = detector.detect_data_drift(X_new)
for col, info in drift_report.items():
    if info['drifted']:
        logger.warning(f"Drift detected in {col}: p-value={info['pvalue']:.4f}")
```

---

## MLOps Best Practices

### 1. CI/CD Pipeline

```yaml
# .github/workflows/ml-pipeline.yml
name: ML Pipeline

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run tests
      run: |
        pytest tests/ --cov=src --cov-report=xml
    
    - name: Lint code
      run: |
        pip install flake8
        flake8 src/ api/
  
  train-model:
    needs: test
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
    
    - name: Train model
      run: |
        python scripts/train.py
    
    - name: Evaluate model
      run: |
        python scripts/evaluate.py
    
    - name: Upload model artifact
      uses: actions/upload-artifact@v2
      with:
        name: trained-model
        path: models/
  
  build-docker:
    needs: train-model
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Build Docker image
      run: |
        docker build -t churn-api:${{ github.sha }} .
    
    - name: Test Docker image
      run: |
        docker run -d -p 8000:8000 churn-api:${{ github.sha }}
        sleep 10
        curl http://localhost:8000/
```

### 2. Structure de Projet Complète

```
churn-prediction/
│
├── data/
│   ├── raw/                  # Données brutes
│   ├── processed/            # Données traitées
│   └── external/             # Données externes
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_loader.py
│   │   └── preprocessor.py
│   ├── features/
│   │   ├── __init__.py
│   │   └── feature_engineering.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py
│   │   └── predict.py
│   ├── monitoring/
│   │   ├── __init__.py
│   │   └── drift_detector.py
│   └── utils/
│       ├── __init__.py
│       └── logger.py
│
├── api/
│   ├── __init__.py
│   ├── main.py              # FastAPI app
│   └── schemas.py           # Pydantic models
│
├── tests/
│   ├── test_data.py
│   ├── test_models.py
│   └── test_api.py
│
├── models/                   # Modèles sauvegardés
│   └── .gitkeep
│
├── logs/                     # Logs
│   └── .gitkeep
│
├── scripts/
│   ├── train.py             # Script d'entraînement
│   ├── evaluate.py          # Script d'évaluation
│   └── deploy.py            # Script de déploiement
│
├── config/
│   ├── config.yaml
│   └── model_config.yaml
│
├── docs/                     # Documentation
│
├── .github/
│   └── workflows/
│       └── ml-pipeline.yml
│
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── requirements.txt
├── setup.py
├── README.md
└── .gitignore
```

---

## Système de Recommandations

### Engine de Recommandations

```python
# src/recommendations/recommendation_engine.py
from typing import List, Dict
import pandas as pd

class RecommendationEngine:
    """Moteur de recommandations pour la rétention client"""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.action_library = self._load_action_library()
    
    def _load_action_library(self) -> Dict:
        """Bibliothèque d'actions de rétention"""
        return {
            'contract': {
                'condition': lambda c: c['Contract'] == 'Month-to-month',
                'action': 'Proposer contrat annuel avec 15% réduction',
                'cost': 50,
                'success_rate': 0.3
            },
            'new_customer': {
                'condition': lambda c: c['tenure'] < 6,
                'action': 'Programme onboarding renforcé + call mensuel',
                'cost': 30,
                'success_rate': 0.4
            },
            'tech_support': {
                'condition': lambda c: c['TechSupport'] == 'No' and c['InternetService'] != 'No',
                'action': 'Offrir support technique gratuit 3 mois',
                'cost': 75,
                'success_rate': 0.35
            },
            'high_charges': {
                'condition': lambda c: c['MonthlyCharges'] > 70,
                'action': 'Audit forfait + optimisation personnalisée',
                'cost': 20,
                'success_rate': 0.25
            },
            'payment_method': {
                'condition': lambda c: c['PaymentMethod'] == 'Electronic check',
                'action': 'Promouvoir paiement automatique + 5€ crédit',
                'cost': 10,
                'success_rate': 0.2
            },
            'no_security': {
                'condition': lambda c: c['OnlineSecurity'] == 'No' and c['InternetService'] != 'No',
                'action': 'Offrir sécurité en ligne gratuit 2 mois',
                'cost': 40,
                'success_rate': 0.3
            }
        }
    
    def generate_recommendations(self, customer_df: pd.DataFrame) -> List[Dict]:
        """Générer des recommandations pour des clients"""
        
        # Prédire le churn
        churn_probas = self.model.predict_proba(customer_df)[:, 1]
        
        recommendations = []
        
        for idx, (_, customer) in enumerate(customer_df.iterrows()):
            churn_prob = churn_probas[idx]
            customer_dict = customer.to_dict()
            
            # Identifier les actions applicables
            applicable_actions = []
            for action_name, action_info in self.action_library.items():
                if action_info['condition'](customer_dict):
                    applicable_actions.append({
                        'action': action_info['action'],
                        'cost': action_info['cost'],
                        'success_rate': action_info['success_rate'],
                        'expected_value': self._calculate_expected_value(
                            churn_prob,
                            action_info['cost'],
                            action_info['success_rate'],
                            customer_dict.get('MonthlyCharges', 0) * 24  # CLV
                        )
                    })
            
            # Trier par expected value
            applicable_actions.sort(key=lambda x: x['expected_value'], reverse=True)
            
            # Sélectionner les meilleures actions
            selected_actions = self._select_optimal_actions(
                applicable_actions, 
                churn_prob,
                max_actions=3
            )
            
            recommendations.append({
                'customer_index': idx,
                'churn_probability': churn_prob,
                'risk_level': self._get_risk_level(churn_prob),
                'recommended_actions': selected_actions,
                'priority_score': churn_prob * customer_dict.get('MonthlyCharges', 0)
            })
        
        return recommendations
    
    def _calculate_expected_value(self, churn_prob, action_cost, success_rate, clv):
        """Calculer la valeur attendue d'une action"""
        # Expected value = (Probability of churn × Success rate × CLV) - Action cost
        return (churn_prob * success_rate * clv) - action_cost
    
    def _get_risk_level(self, churn_prob):
        """Déterminer le niveau de risque"""
        if churn_prob >= 0.7:
            return 'High'
        elif churn_prob >= 0.4:
            return 'Medium'
        else:
            return 'Low'
    
    def _select_optimal_actions(self, actions, churn_prob, max_actions=3):
        """Sélectionner les actions optimales"""
        
        if churn_prob < 0.3:
            # Low risk: pas d'action ou monitoring
            return [{'action': 'Monitoring standard', 'cost': 0}]
        
        elif churn_prob < 0.6:
            # Medium risk: 1-2 actions
            return actions[:min(2, len(actions))]
        
        else:
            # High risk: jusqu'à 3 actions + contact urgent
            selected = actions[:min(max_actions, len(actions))]
            selected.insert(0, {
                'action': 'URGENT: Contact immédiat équipe rétention',
                'cost': 100,
                'priority': 'HIGH'
            })
            return selected
    
    def batch_prioritization(self, customers_df: pd.DataFrame, budget: float = None):
        """Prioriser un lot de clients selon le budget"""
        
        recommendations = self.generate_recommendations(customers_df)
        
        # Trier par priority_score
        recommendations.sort(key=lambda x: x['priority_score'], reverse=True)
        
        if budget:
            # Sélectionner les clients dans le budget
            total_cost = 0
            prioritized = []
            
            for rec in recommendations:
                action_costs = sum(a.get('cost', 0) for a in rec['recommended_actions'])
                if total_cost + action_costs <= budget:
                    prioritized.append(rec)
                    total_cost += action_costs
                else:
                    break
            
            return {
                'total_customers': len(recommendations),
                'customers_in_budget': len(prioritized),
                'total_cost': total_cost,
                'remaining_budget': budget - total_cost,
                'prioritized_customers': prioritized
            }
        
        return recommendations

# Utilisation
engine = RecommendationEngine(model, config)

# Pour un batch de clients
customers = load_at_risk_customers()
recommendations = engine.batch_prioritization(customers, budget=10000)

print(f"Clients à contacter: {recommendations['customers_in_budget']}")
print(f"Budget utilisé: {recommendations['total_cost']}€")
```

---

## Bonnes Pratiques Finales

### ✅ Checklist de Déploiement

- [ ] Code refactoré et modulaire
- [ ] Tests unitaires et d'intégration
- [ ] Logging configuré
- [ ] Monitoring des métriques
- [ ] Gestion des erreurs robuste
- [ ] Documentation API (Swagger/OpenAPI)
- [ ] Dockerfile et docker-compose
- [ ] CI/CD configuré
- [ ] Détection de drift
- [ ] Plan de rollback
- [ ] Sécurité (authentification, rate limiting)
- [ ] Versioning des modèles
- [ ] Backups réguliers

### ⚠️ Points d'Attention

1. **Ne jamais** déployer sans tests
2. **Toujours** versionner les modèles
3. **Monitorer** continuellement en production
4. **Documenter** les décisions et changements
5. **Prévoir** un plan de rollback
6. **Tester** le système de A à Z avant production

---

**Navigation**
- [← Précédent : Meilleures Pratiques EDA](05_eda_best_practices.md)
- [Retour au README principal](../README.md)
