# 🧠 Fondamentaux du Machine Learning

## Table des Matières
- [Introduction](#introduction)
- [Qu'est-ce que le Machine Learning ?](#quest-ce-que-le-machine-learning)
- [Types d'Apprentissage](#types-dapprentissage)
- [Le Processus de Machine Learning](#le-processus-de-machine-learning)
- [Concepts Clés](#concepts-clés)
- [Références](#références)

## Introduction

Le Machine Learning (ML) est une branche de l'intelligence artificielle qui permet aux ordinateurs d'apprendre à partir de données sans être explicitement programmés. Ce guide présente les concepts fondamentaux nécessaires pour comprendre et appliquer le ML.

## Qu'est-ce que le Machine Learning ?

Le Machine Learning consiste à développer des algorithmes qui peuvent :
- **Apprendre** des patterns à partir de données
- **Généraliser** ces patterns à de nouvelles données
- **Faire des prédictions** ou **prendre des décisions** basées sur ces apprentissages

### Différence avec la Programmation Traditionnelle

**Programmation Traditionnelle :**
```
Données + Programme → Résultats
```

**Machine Learning :**
```
Données + Résultats → Programme (Modèle)
```

## Types d'Apprentissage

### 1. Apprentissage Supervisé (Supervised Learning)
- **Description** : Le modèle apprend à partir de données étiquetées (avec réponses)
- **Exemples** : Classification, Régression
- **Use Cases** : Prédiction de prix, détection de spam, diagnostic médical

### 2. Apprentissage Non Supervisé (Unsupervised Learning)
- **Description** : Le modèle découvre des patterns dans des données non étiquetées
- **Exemples** : Clustering, Réduction de dimensionnalité
- **Use Cases** : Segmentation de clients, détection d'anomalies

### 3. Apprentissage par Renforcement (Reinforcement Learning)
- **Description** : Le modèle apprend par essais-erreurs avec un système de récompenses
- **Exemples** : Jeux, Robotique
- **Use Cases** : AlphaGo, voitures autonomes

### 4. Apprentissage Semi-Supervisé
- **Description** : Combinaison de données étiquetées et non étiquetées
- **Use Cases** : Quand l'étiquetage est coûteux

## Le Processus de Machine Learning

### 1. Définition du Problème
- Identifier l'objectif business
- Définir la métrique de succès
- Déterminer le type de problème ML

### 2. Collection et Préparation des Données
- **Collecte** : Rassembler les données pertinentes
- **Nettoyage** : Gérer les valeurs manquantes, outliers
- **Exploration** : EDA (Exploratory Data Analysis)
- **Transformation** : Normalisation, encodage

### 3. Feature Engineering
- Création de nouvelles features
- Sélection des features pertinentes
- Transformation des variables

### 4. Séparation des Données
```python
# Exemple typique
Train Set (70-80%) : Pour entraîner le modèle
Validation Set (10-15%) : Pour ajuster les hyperparamètres
Test Set (10-15%) : Pour évaluer la performance finale
```

### 5. Entraînement du Modèle
- Choix de l'algorithme approprié
- Entraînement sur le training set
- Ajustement des hyperparamètres

### 6. Évaluation
- Test sur le test set
- Calcul des métriques de performance
- Analyse des erreurs

### 7. Déploiement et Monitoring
- Mise en production
- Surveillance des performances
- Réentraînement périodique

## Concepts Clés

### Overfitting (Surapprentissage)
**Définition** : Le modèle apprend trop bien les données d'entraînement, incluant le bruit
- **Symptômes** : Excellent sur train set, mauvais sur test set
- **Solutions** : Régularisation, plus de données, validation croisée

### Underfitting (Sous-apprentissage)
**Définition** : Le modèle est trop simple pour capturer les patterns
- **Symptômes** : Mauvais sur train et test set
- **Solutions** : Modèle plus complexe, plus de features

### Biais-Variance Tradeoff
- **Biais élevé** : Underfitting → Modèle trop simple
- **Variance élevée** : Overfitting → Modèle trop complexe
- **Objectif** : Trouver l'équilibre optimal

### Validation Croisée (Cross-Validation)
Technique pour évaluer la performance d'un modèle de manière robuste :
```
K-Fold Cross-Validation :
- Diviser les données en K parties
- Entraîner K fois (chaque fois avec K-1 parties)
- Moyenner les performances
```

### Hyperparamètres vs Paramètres
- **Paramètres** : Appris par le modèle (poids, biais)
- **Hyperparamètres** : Définis avant l'entraînement (learning rate, nombre d'arbres)

### Feature Scaling
Normaliser les features pour améliorer la performance :
- **Standardization** : (x - μ) / σ
- **Min-Max Scaling** : (x - min) / (max - min)
- **Quand l'utiliser** : KNN, SVM, Régression linéaire, Réseaux de neurones

### Curse of Dimensionality
Plus on a de features, plus il faut de données pour éviter l'overfitting
- **Solutions** : Feature selection, PCA, regularization

## Workflow Typique en Python

```python
# 1. Import des bibliothèques
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 2. Chargement des données
data = pd.read_csv('data.csv')

# 3. Préparation
X = data.drop('target', axis=1)
y = data['target']

# 4. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Preprocessing
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Entraînement
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# 7. Prédiction
y_pred = model.predict(X_test_scaled)

# 8. Évaluation
print(classification_report(y_test, y_pred))
```

## Bibliothèques Python Essentielles

### Pour le ML
- **Scikit-learn** : Algorithmes ML classiques
- **TensorFlow/PyTorch** : Deep Learning
- **XGBoost/LightGBM** : Gradient Boosting

### Pour la Data Manipulation
- **Pandas** : Manipulation de données
- **NumPy** : Calculs numériques
- **Polars** : Alternative rapide à Pandas

### Pour la Visualisation
- **Matplotlib** : Visualisations de base
- **Seaborn** : Visualisations statistiques
- **Plotly** : Visualisations interactives

## Bonnes Pratiques

1. **Toujours** séparer train/test avant toute transformation
2. **Fit** les transformations sur train, **transform** sur test
3. **Utiliser** la validation croisée pour une évaluation robuste
4. **Documenter** vos expériences et résultats
5. **Versionner** vos données et modèles
6. **Surveiller** les performances en production
7. **Réentraîner** régulièrement avec de nouvelles données

## Erreurs Courantes à Éviter

❌ **Data Leakage** : Utiliser des informations du test set pendant l'entraînement
❌ **Feature Scaling sur tout le dataset** : Scaler avant le split
❌ **Ignorer les valeurs manquantes** : Toujours les gérer explicitement
❌ **Ne pas valider les hypothèses** : Vérifier les assumptions des modèles
❌ **Optimiser uniquement sur accuracy** : Utiliser plusieurs métriques

## Ressources Additionnelles

### Livres
- "Hands-On Machine Learning" - Aurélien Géron
- "The Elements of Statistical Learning" - Hastie, Tibshirani, Friedman
- "Pattern Recognition and Machine Learning" - Christopher Bishop

### Cours en Ligne
- Coursera : Machine Learning by Andrew Ng
- Fast.ai : Practical Deep Learning
- DataCamp : Machine Learning Scientist Track

### Documentation
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [PyTorch Documentation](https://pytorch.org/)

## Références

- Bishop, C. M. (2006). Pattern Recognition and Machine Learning
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning
- Géron, A. (2019). Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow

---

**Navigation**
- [Retour au README principal](../README.md)
- [Suivant : Types de Tâches ML →](02_ml_task_types.md)
