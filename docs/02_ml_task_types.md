# 📋 Types de Tâches en Machine Learning

## Table des Matières
- [Introduction](#introduction)
- [1. Régression (Regression)](#1-régression-regression)
- [2. Classification](#2-classification)
- [3. Clustering](#3-clustering)
- [4. Détection d'Anomalies (Anomaly Detection)](#4-détection-danomalies-anomaly-detection)
- [5. Séries Temporelles (Time Series)](#5-séries-temporelles-time-series)
- [6. Autres Types de Tâches](#6-autres-types-de-tâches)
- [Comment Choisir ?](#comment-choisir)

## Introduction

Le Machine Learning couvre différents types de tâches, chacune adaptée à des problèmes spécifiques. Ce guide détaille les principales catégories de tâches ML, leurs caractéristiques, algorithmes et cas d'usage.

## 1. Régression (Regression)

### Description
Prédire une **valeur numérique continue** à partir de features.

### Caractéristiques
- **Variable cible** : Continue (nombres réels)
- **Type d'apprentissage** : Supervisé
- **Output** : Un nombre (ex: 45.6, 1200.5)

### Algorithmes Courants
| Algorithme | Avantages | Inconvénients | Quand l'utiliser |
|------------|-----------|---------------|------------------|
| **Régression Linéaire** | Simple, interprétable | Assume linéarité | Relations linéaires |
| **Ridge/Lasso** | Régularisation | Tuning des hyperparamètres | Beaucoup de features |
| **Random Forest Regressor** | Gère non-linéarité | Moins interprétable | Relations complexes |
| **XGBoost Regressor** | Très performant | Temps d'entraînement | Compétitions, production |
| **SVR** | Efficace en haute dimension | Coûteux en calcul | Datasets petits/moyens |
| **Neural Networks** | Très flexible | Nécessite beaucoup de données | Données massives |

### Cas d'Usage
- 💰 **Prédiction de prix** : immobilier, actions
- 📈 **Prévisions de ventes**
- 🌡️ **Prédiction de température**
- ⏱️ **Estimation de temps de livraison**
- 💵 **Prédiction de revenus**

### Exemple de Code
```python
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Régression linéaire
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred = lr_model.predict(X_test)

# Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Évaluation
print(f"MSE: {mean_squared_error(y_test, y_pred)}")
print(f"R²: {r2_score(y_test, y_pred)}")
```

### Métriques Principales
- **MSE** (Mean Squared Error) : Pénalise fortement les grandes erreurs
- **RMSE** (Root Mean Squared Error) : Même unité que la variable cible
- **MAE** (Mean Absolute Error) : Moins sensible aux outliers
- **R²** (Coefficient de détermination) : Proportion de variance expliquée

---

## 2. Classification

### Description
Prédire une **catégorie** ou **classe** discrète à partir de features.

### 2.1 Classification Binaire
- **Variable cible** : 2 classes (0/1, Oui/Non, True/False)
- **Exemples** : Spam/Not Spam, Fraud/Not Fraud, Churn/No Churn

### 2.2 Classification Multi-classe
- **Variable cible** : > 2 classes mutuellement exclusives
- **Exemples** : Reconnaissance de chiffres (0-9), Catégories de produits

### 2.3 Classification Multi-label
- **Variable cible** : Plusieurs labels simultanés
- **Exemples** : Tags d'articles, Genres de films

### Algorithmes Courants
| Algorithme | Avantages | Inconvénients | Quand l'utiliser |
|------------|-----------|---------------|------------------|
| **Régression Logistique** | Simple, rapide, interprétable | Assume linéarité | Baseline, problèmes linéaires |
| **Decision Trees** | Interprétable | Overfitting facile | Règles métier claires |
| **Random Forest** | Robuste, performant | Boîte noire | Production, bonnes perfs |
| **XGBoost/LightGBM** | Excellentes performances | Hyperparamètres complexes | Compétitions |
| **SVM** | Efficace en haute dimension | Pas scalable | Petits datasets |
| **Neural Networks** | Très flexible | Beaucoup de données nécessaires | Images, texte, données complexes |
| **KNN** | Simple | Lent en prédiction | Petits datasets |
| **Naive Bayes** | Rapide, efficace | Assume indépendance | Classification de texte |

### Cas d'Usage
- 📧 **Détection de spam**
- 🏥 **Diagnostic médical**
- 💳 **Détection de fraude**
- 👤 **Prédiction de churn** (notre projet!)
- 🖼️ **Classification d'images**
- 📝 **Analyse de sentiment**
- 🔍 **Reconnaissance de caractères**

### Exemple de Code
```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Régression Logistique
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train, y_train)
y_pred = lr_model.predict(X_test)
y_pred_proba = lr_model.predict_proba(X_test)[:, 1]

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Évaluation
print(classification_report(y_test, y_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba)}")
```

### Métriques Principales
- **Accuracy** : % de prédictions correctes
- **Precision** : TP / (TP + FP)
- **Recall** : TP / (TP + FN)
- **F1-Score** : Moyenne harmonique de Precision et Recall
- **ROC-AUC** : Aire sous la courbe ROC
- **Confusion Matrix** : Matrice de confusion

---

## 3. Clustering

### Description
Regrouper des données similaires sans labels préexistants (**Apprentissage non supervisé**).

### Caractéristiques
- **Type d'apprentissage** : Non supervisé
- **Objectif** : Découvrir des groupes naturels
- **Pas de variable cible**

### Algorithmes Courants
| Algorithme | Avantages | Inconvénients | Quand l'utiliser |
|------------|-----------|---------------|------------------|
| **K-Means** | Rapide, simple | Nombre de clusters à définir | Clusters sphériques |
| **DBSCAN** | Trouve clusters de forme arbitraire | Sensible aux paramètres | Détection d'outliers |
| **Hierarchical Clustering** | Pas besoin de K | Pas scalable | Petit dataset, dendrogrammes |
| **Gaussian Mixture Models** | Soft clustering | Plus complexe | Clusters qui se chevauchent |

### Cas d'Usage
- 👥 **Segmentation de clients**
- 🗺️ **Analyse géographique**
- 🧬 **Analyse génétique**
- 📰 **Groupement de documents**
- 🎨 **Compression d'images**

### Exemple de Code
```python
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

# K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X)

# Évaluation
score = silhouette_score(X, clusters)
print(f"Silhouette Score: {score}")

# DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
clusters_db = dbscan.fit_predict(X)
```

### Métriques Principales
- **Silhouette Score** : Cohésion et séparation des clusters
- **Inertia** : Somme des distances au centre (K-Means)
- **Davies-Bouldin Index** : Ratio dispersion/séparation
- **Calinski-Harabasz Index** : Ratio variance inter/intra cluster

---

## 4. Détection d'Anomalies (Anomaly Detection)

### Description
Identifier les observations qui dévient significativement du comportement normal.

### Caractéristiques
- **Type d'apprentissage** : Généralement non supervisé ou semi-supervisé
- **Objectif** : Trouver les points inhabituels
- **Classes déséquilibrées** : Anomalies rares

### Algorithmes Courants
| Algorithme | Avantages | Inconvénients | Quand l'utiliser |
|------------|-----------|---------------|------------------|
| **Isolation Forest** | Efficace, rapide | Paramètres à tuner | Détection générale |
| **One-Class SVM** | Robuste | Pas scalable | Datasets petits |
| **Local Outlier Factor** | Détecte anomalies locales | Coûteux en calcul | Anomalies locales |
| **Autoencoders** | Capture patterns complexes | Nécessite beaucoup de données | Données complexes |
| **Statistical Methods** | Simple, interprétable | Assume distribution | Données simples |

### Cas d'Usage
- 💳 **Détection de fraude bancaire**
- 🏭 **Maintenance prédictive** (défaillances machines)
- 🔐 **Cybersécurité** (intrusions réseau)
- 🏥 **Diagnostic médical** (cas rares)
- 📊 **Contrôle qualité**

### Exemple de Code
```python
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# Isolation Forest
iso_forest = IsolationForest(contamination=0.1, random_state=42)
anomalies = iso_forest.fit_predict(X)

# LOF
lof = LocalOutlierFactor(contamination=0.1)
anomalies_lof = lof.fit_predict(X)

# -1 = anomalie, 1 = normal
print(f"Anomalies détectées: {(anomalies == -1).sum()}")
```

### Métriques Principales
- **Precision/Recall** : Si labels disponibles
- **F1-Score** : Équilibre entre précision et rappel
- **Contamination Rate** : Proportion d'anomalies attendues

---

## 5. Séries Temporelles (Time Series)

### Description
Analyser et prédire des données séquentielles dépendant du temps.

### Caractéristiques
- **Dépendance temporelle** : L'ordre des observations est crucial
- **Composantes** : Tendance, saisonnalité, cyclicité, bruit
- **Types** : Univarié (une variable) ou Multivarié (plusieurs variables)

### Approches Principales

#### 5.1 Méthodes Statistiques
- **ARIMA** : AutoRegressive Integrated Moving Average
- **SARIMA** : ARIMA avec saisonnalité
- **Prophet** : Développé par Facebook
- **Exponential Smoothing**

#### 5.2 Machine Learning
- **Regression Models** : Avec features temporelles
- **Random Forest/XGBoost** : Pour séries complexes

#### 5.3 Deep Learning
- **LSTM** : Long Short-Term Memory
- **GRU** : Gated Recurrent Unit
- **Transformer** : Architecture attention

### Cas d'Usage
- 📈 **Prévision de ventes**
- 💹 **Prédiction de cours boursiers**
- 🌤️ **Prévisions météorologiques**
- ⚡ **Prédiction de consommation énergétique**
- 🚦 **Prédiction de trafic**

### Exemple de Code
```python
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error

# ARIMA
model = ARIMA(train_data, order=(1, 1, 1))
model_fit = model.fit()
forecast = model_fit.forecast(steps=10)

# Avec sklearn (features temporelles)
from sklearn.ensemble import RandomForestRegressor

# Créer des features temporelles
df['lag_1'] = df['value'].shift(1)
df['lag_7'] = df['value'].shift(7)
df['rolling_mean_7'] = df['value'].rolling(7).mean()

# Entraîner le modèle
model = RandomForestRegressor()
model.fit(X_train, y_train)
```

### Métriques Principales
- **MAE** : Mean Absolute Error
- **RMSE** : Root Mean Squared Error
- **MAPE** : Mean Absolute Percentage Error
- **SMAPE** : Symmetric MAPE

---

## 6. Autres Types de Tâches

### 6.1 Réduction de Dimensionnalité
**Objectif** : Réduire le nombre de features tout en conservant l'information

**Algorithmes** :
- **PCA** (Principal Component Analysis)
- **t-SNE** (pour visualisation)
- **UMAP** (pour visualisation)
- **Autoencoders**

**Use Cases** :
- Visualisation de données haute dimension
- Réduction de bruit
- Feature extraction

### 6.2 Ranking
**Objectif** : Ordonner des items selon leur pertinence

**Use Cases** :
- Moteurs de recherche
- Systèmes de recommandation
- Publicité en ligne

### 6.3 Recommandation
**Objectif** : Suggérer des items pertinents à un utilisateur

**Approches** :
- **Collaborative Filtering** : Basé sur comportements similaires
- **Content-Based** : Basé sur caractéristiques des items
- **Hybrid** : Combinaison des deux

**Use Cases** :
- Netflix, YouTube (recommandations de vidéos)
- E-commerce (produits suggérés)
- Spotify (musique)

### 6.4 NLP (Natural Language Processing)
**Tâches** :
- Classification de texte
- Named Entity Recognition (NER)
- Machine Translation
- Question Answering
- Summarization

**Algorithmes** :
- BERT, GPT, T5 (Transformers)
- RNN, LSTM
- Bag of Words, TF-IDF

### 6.5 Computer Vision
**Tâches** :
- Classification d'images
- Détection d'objets
- Segmentation
- Face Recognition

**Algorithmes** :
- CNN (Convolutional Neural Networks)
- ResNet, VGG, EfficientNet
- YOLO, R-CNN (détection)

---

## Comment Choisir ?

### Diagramme de Décision

```
Avez-vous des labels (variable cible) ?
│
├─ OUI → Apprentissage Supervisé
│   │
│   ├─ Variable cible continue (nombres) ?
│   │   └─ OUI → RÉGRESSION
│   │
│   └─ Variable cible catégorielle (classes) ?
│       └─ OUI → CLASSIFICATION
│
└─ NON → Apprentissage Non Supervisé
    │
    ├─ Voulez-vous grouper des observations similaires ?
    │   └─ OUI → CLUSTERING
    │
    ├─ Voulez-vous trouver des observations anormales ?
    │   └─ OUI → DÉTECTION D'ANOMALIES
    │
    └─ Voulez-vous réduire le nombre de variables ?
        └─ OUI → RÉDUCTION DE DIMENSIONNALITÉ

Vos données ont-elles une dépendance temporelle ?
└─ OUI → SÉRIES TEMPORELLES
```

### Questions à se Poser

1. **Quel est mon objectif business ?**
   - Prédire une valeur → Régression
   - Classer dans des catégories → Classification
   - Découvrir des groupes → Clustering
   - Trouver des anomalies → Anomaly Detection

2. **Ai-je des labels ?**
   - Oui → Supervisé
   - Non → Non supervisé
   - Partiellement → Semi-supervisé

3. **Quel type de variable cible ?**
   - Continue → Régression
   - Catégorielle → Classification

4. **Mes données sont-elles temporelles ?**
   - Oui → Time Series
   - Non → Autres approches

5. **Ai-je des contraintes ?**
   - Interprétabilité → Modèles simples (LR, DT)
   - Performance → Ensemble methods, DL
   - Temps réel → Modèles rapides (LR, KNN)
   - Peu de données → Modèles simples, feature engineering

### Tableau Récapitulatif

| Type de Tâche | Supervisé ? | Variable Cible | Use Cases Principaux |
|---------------|-------------|----------------|---------------------|
| **Régression** | Oui | Continue | Prix, ventes, température |
| **Classification** | Oui | Catégorielle | Spam, churn, diagnostic |
| **Clustering** | Non | Aucune | Segmentation clients |
| **Anomaly Detection** | Non/Semi | Aucune/Binaire | Fraude, maintenance |
| **Time Series** | Oui/Non | Continue | Prévisions, prédictions |

---

## Ressources Complémentaires

### Documentation
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Choosing the right estimator](https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html)

### Livres
- "Pattern Recognition and Machine Learning" - Christopher Bishop
- "Hands-On Machine Learning" - Aurélien Géron

---

**Navigation**
- [← Précédent : Fondamentaux ML](01_machine_learning_fundamentals.md)
- [Suivant : Guide des Métriques →](03_metrics_guide.md)
- [Retour au README principal](../README.md)
