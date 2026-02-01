# 📊 Guide Complet des Métriques en Machine Learning

## Table des Matières
- [Introduction](#introduction)
- [Métriques de Régression](#métriques-de-régression)
- [Métriques de Classification](#métriques-de-classification)
- [Métriques de Clustering](#métriques-de-clustering)
- [Comment Choisir ses Métriques](#comment-choisir-ses-métriques)
- [Combiner Plusieurs Métriques](#combiner-plusieurs-métriques)
- [Métriques Business vs Métriques ML](#métriques-business-vs-métriques-ml)

## Introduction

Les métriques sont essentielles pour évaluer la performance des modèles ML. Choisir les bonnes métriques dépend :
- Du **type de problème** (régression, classification, etc.)
- Du **contexte business**
- Des **coûts** associés aux erreurs
- De la **distribution** des classes

> ⚠️ **Règle d'Or** : Ne jamais se fier à une seule métrique. Toujours en combiner plusieurs pour une évaluation complète.

---

## Métriques de Régression

### 1. MAE (Mean Absolute Error)

**Formule** :
```
MAE = (1/n) × Σ|yᵢ - ŷᵢ|
```

**Caractéristiques** :
- ✅ Interprétable (même unité que y)
- ✅ Robuste aux outliers
- ❌ Ne pénalise pas fortement les grandes erreurs

**Quand l'utiliser** :
- Quand les outliers ne doivent pas avoir trop d'impact
- Quand on veut une métrique facile à expliquer
- Pour des prédictions de prix, distances

**Exemple** :
```python
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_true, y_pred)
print(f"MAE: {mae:.2f} euros")  # Ex: "MAE: 50.00 euros"
```

---

### 2. MSE (Mean Squared Error)

**Formule** :
```
MSE = (1/n) × Σ(yᵢ - ŷᵢ)²
```

**Caractéristiques** :
- ✅ Pénalise fortement les grandes erreurs
- ❌ Pas la même unité que y (unité²)
- ❌ Sensible aux outliers

**Quand l'utiliser** :
- Quand les grandes erreurs sont très coûteuses
- Pour l'optimisation (dérivable)
- En interne pour l'entraînement

**Exemple** :
```python
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_true, y_pred)
print(f"MSE: {mse:.2f}")
```

---

### 3. RMSE (Root Mean Squared Error)

**Formule** :
```
RMSE = √MSE = √[(1/n) × Σ(yᵢ - ŷᵢ)²]
```

**Caractéristiques** :
- ✅ Même unité que y
- ✅ Pénalise fortement les grandes erreurs
- ❌ Sensible aux outliers

**Quand l'utiliser** :
- Version "interprétable" du MSE
- Métrique standard pour comparer des modèles
- Quand on veut pénaliser les grandes erreurs

**Exemple** :
```python
import numpy as np
from sklearn.metrics import mean_squared_error

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
print(f"RMSE: {rmse:.2f} euros")
```

---

### 4. R² (Coefficient de Détermination)

**Formule** :
```
R² = 1 - (SS_res / SS_tot)
où SS_res = Σ(yᵢ - ŷᵢ)²
et SS_tot = Σ(yᵢ - ȳ)²
```

**Caractéristiques** :
- ✅ Sans unité (entre -∞ et 1)
- ✅ Facile à interpréter
- ❌ Peut être trompeur avec des modèles complexes

**Interprétation** :
- R² = 1 : Modèle parfait
- R² = 0 : Modèle équivalent à la moyenne
- R² < 0 : Modèle pire que la moyenne

**Quand l'utiliser** :
- Pour expliquer la variance capturée
- Comparer des modèles sur le même dataset
- Communication avec non-experts

**Exemple** :
```python
from sklearn.metrics import r2_score

r2 = r2_score(y_true, y_pred)
print(f"R²: {r2:.3f}")  # Ex: "R²: 0.856" → 85.6% de variance expliquée
```

---

### 5. MAPE (Mean Absolute Percentage Error)

**Formule** :
```
MAPE = (100/n) × Σ|((yᵢ - ŷᵢ) / yᵢ)|
```

**Caractéristiques** :
- ✅ Indépendante de l'échelle
- ✅ Facile à interpréter (%)
- ❌ Problème si yᵢ = 0
- ❌ Asymétrique

**Quand l'utiliser** :
- Pour comparer des modèles sur différents datasets
- Quand l'erreur relative est importante
- Séries temporelles

**Exemple** :
```python
def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mape_score = mape(y_true, y_pred)
print(f"MAPE: {mape_score:.2f}%")  # Ex: "MAPE: 5.23%"
```

---

### Tableau Comparatif : Métriques de Régression

| Métrique | Interprétabilité | Robuste aux Outliers | Quand privilégier |
|----------|------------------|----------------------|-------------------|
| **MAE** | ⭐⭐⭐ | ✅ | Outliers présents, interprétation simple |
| **MSE** | ⭐ | ❌ | Optimisation, grandes erreurs coûteuses |
| **RMSE** | ⭐⭐⭐ | ❌ | Standard, grandes erreurs importantes |
| **R²** | ⭐⭐⭐ | ⭐⭐ | Variance expliquée, communication |
| **MAPE** | ⭐⭐⭐ | ⭐⭐ | Erreur relative, comparaison multi-datasets |

---

## Métriques de Classification

### Matrice de Confusion

La base de toutes les métriques de classification :

```
                Prédiction
                Positive    Negative
Réalité  Pos    TP          FN
         Neg    FP          TN
```

- **TP** (True Positive) : Correctement prédit positif
- **TN** (True Negative) : Correctement prédit négatif
- **FP** (False Positive) : Incorrectement prédit positif (Erreur Type I)
- **FN** (False Negative) : Incorrectement prédit négatif (Erreur Type II)

---

### 1. Accuracy (Exactitude)

**Formule** :
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**Caractéristiques** :
- ✅ Simple, intuitif
- ❌ Trompeur avec classes déséquilibrées
- ❌ Ne différencie pas les types d'erreurs

**Quand l'utiliser** :
- Classes équilibrées
- Coûts d'erreur similaires
- Baseline simple

**⚠️ Exemple du Piège** :
```
Dataset : 95% classe 0, 5% classe 1
Modèle naïf prédisant toujours 0 → Accuracy = 95% !
```

**Exemple** :
```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy:.3f}")  # Ex: 0.856
```

---

### 2. Precision (Précision)

**Formule** :
```
Precision = TP / (TP + FP)
```

**Signification** : Parmi les prédictions positives, combien sont vraiment positives ?

**Quand l'utiliser** :
- Le coût d'un **False Positive est élevé**
- Spam detection (éviter de marquer un vrai email comme spam)
- Recommandations (éviter de recommander des mauvais items)

**Exemple** :
```python
from sklearn.metrics import precision_score

precision = precision_score(y_true, y_pred)
print(f"Precision: {precision:.3f}")
```

---

### 3. Recall (Rappel / Sensibilité)

**Formule** :
```
Recall = TP / (TP + FN)
```

**Signification** : Parmi les vraies positifs, combien sont détectés ?

**Quand l'utiliser** :
- Le coût d'un **False Negative est élevé**
- Détection de fraude (ne pas manquer une fraude)
- Diagnostic médical (ne pas manquer une maladie)
- Détection de churn (ne pas manquer un client à risque)

**Exemple** :
```python
from sklearn.metrics import recall_score

recall = recall_score(y_true, y_pred)
print(f"Recall: {recall:.3f}")
```

---

### 4. F1-Score

**Formule** :
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**Caractéristiques** :
- ✅ Équilibre entre Precision et Recall
- ✅ Bon pour classes déséquilibrées
- ❌ Traite Precision et Recall également

**Quand l'utiliser** :
- Classes déséquilibrées
- Besoin d'équilibre entre Precision et Recall
- Métrique unique pour comparer des modèles

**Variantes** :
- **F2-Score** : Favorise le Recall (β=2)
- **F0.5-Score** : Favorise la Precision (β=0.5)

**Exemple** :
```python
from sklearn.metrics import f1_score

f1 = f1_score(y_true, y_pred)
print(f"F1-Score: {f1:.3f}")
```

---

### 5. ROC-AUC (Area Under the ROC Curve)

**Description** :
- Courbe ROC : True Positive Rate vs False Positive Rate
- AUC : Aire sous cette courbe

**Caractéristiques** :
- ✅ Indépendant du seuil de décision
- ✅ Bon pour classes déséquilibrées
- ✅ Compare la capacité de discrimination

**Interprétation** :
- AUC = 1.0 : Modèle parfait
- AUC = 0.5 : Modèle aléatoire
- AUC < 0.5 : Pire qu'aléatoire (inversez les prédictions!)

**Quand l'utiliser** :
- Comparer des modèles
- Quand le seuil n'est pas fixé
- Classes déséquilibrées

**Exemple** :
```python
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Calcul AUC
auc = roc_auc_score(y_true, y_pred_proba)
print(f"ROC-AUC: {auc:.3f}")

# Courbe ROC
fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
```

---

### 6. Precision-Recall AUC

**Description** :
- Alternative à ROC-AUC
- Courbe Precision vs Recall

**Quand l'utiliser** :
- **Classes très déséquilibrées**
- Classe positive rare et importante
- Meilleure que ROC-AUC dans ces cas

**Exemple** :
```python
from sklearn.metrics import precision_recall_curve, auc

precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
pr_auc = auc(recall, precision)
print(f"PR-AUC: {pr_auc:.3f}")
```

---

### 7. Log Loss (Cross-Entropy)

**Formule** :
```
Log Loss = -(1/n) × Σ[yᵢ×log(ŷᵢ) + (1-yᵢ)×log(1-ŷᵢ)]
```

**Caractéristiques** :
- ✅ Prend en compte les probabilités (pas juste 0/1)
- ✅ Pénalise les prédictions confiantes mais fausses
- ❌ Moins interprétable

**Quand l'utiliser** :
- Probabilités calibrées importantes
- Optimisation de modèles
- Compétitions Kaggle

**Exemple** :
```python
from sklearn.metrics import log_loss

logloss = log_loss(y_true, y_pred_proba)
print(f"Log Loss: {logloss:.3f}")  # Plus bas = meilleur
```

---

### Choisir entre Precision et Recall

| Contexte | Privilégier | Raison |
|----------|-------------|--------|
| **Spam Detection** | Precision | Éviter de bloquer vrais emails |
| **Fraud Detection** | Recall | Ne pas manquer une fraude |
| **Medical Diagnosis (cancer)** | Recall | Ne pas manquer un malade |
| **Recommended Products** | Precision | Éviter mauvaises recommandations |
| **Churn Prediction** | Recall | Ne pas manquer clients à risque |
| **Content Moderation** | Recall | Ne pas manquer contenu inapproprié |

---

### Tableau Comparatif : Métriques de Classification

| Métrique | Classes Déséquilibrées | Interprétabilité | Usage Principal |
|----------|------------------------|------------------|-----------------|
| **Accuracy** | ❌ | ⭐⭐⭐ | Baseline, classes équilibrées |
| **Precision** | ✅ | ⭐⭐⭐ | FP coûteux |
| **Recall** | ✅ | ⭐⭐⭐ | FN coûteux |
| **F1-Score** | ✅ | ⭐⭐ | Équilibre Precision-Recall |
| **ROC-AUC** | ✅ | ⭐⭐ | Comparaison modèles |
| **PR-AUC** | ✅✅ | ⭐⭐ | Classes très déséquilibrées |
| **Log Loss** | ✅ | ⭐ | Probabilités calibrées |

---

## Métriques de Clustering

### 1. Silhouette Score

**Formule** :
```
s(i) = (b(i) - a(i)) / max(a(i), b(i))
où a(i) = distance moyenne intra-cluster
et b(i) = distance moyenne au cluster le plus proche
```

**Interprétation** :
- Score entre -1 et 1
- Proche de 1 : Bon clustering
- Proche de 0 : Sur la frontière
- Négatif : Probablement dans le mauvais cluster

**Exemple** :
```python
from sklearn.metrics import silhouette_score

score = silhouette_score(X, cluster_labels)
print(f"Silhouette Score: {score:.3f}")
```

---

### 2. Davies-Bouldin Index

**Caractéristiques** :
- Plus bas = meilleur
- Mesure le ratio dispersion/séparation

**Exemple** :
```python
from sklearn.metrics import davies_bouldin_score

db_score = davies_bouldin_score(X, cluster_labels)
print(f"Davies-Bouldin: {db_score:.3f}")  # Plus bas = meilleur
```

---

### 3. Calinski-Harabasz Index

**Caractéristiques** :
- Plus élevé = meilleur
- Variance inter-cluster vs intra-cluster

**Exemple** :
```python
from sklearn.metrics import calinski_harabasz_score

ch_score = calinski_harabasz_score(X, cluster_labels)
print(f"Calinski-Harabasz: {ch_score:.2f}")  # Plus élevé = meilleur
```

---

## Comment Choisir ses Métriques

### Étape 1 : Identifier le Type de Problème

```
Régression → MAE, RMSE, R²
Classification → Accuracy, F1, ROC-AUC
Clustering → Silhouette, Davies-Bouldin
```

### Étape 2 : Analyser le Contexte Business

**Questions à se poser** :
1. Les classes sont-elles équilibrées ?
2. Quel type d'erreur est le plus coûteux ?
3. Ai-je besoin de probabilités calibrées ?
4. Dois-je expliquer à des non-experts ?

### Étape 3 : Considérer les Contraintes

- **Interprétabilité** : Privilégier MAE, Accuracy, Precision, Recall
- **Performance pure** : ROC-AUC, RMSE, Log Loss
- **Classes déséquilibrées** : F1, ROC-AUC, PR-AUC
- **Communication** : R², Accuracy, MAPE

---

## Combiner Plusieurs Métriques

### Principe Fondamental

> ⚠️ **Ne JAMAIS se fier à une seule métrique !**

### Approches de Combinaison

#### 1. Ensemble Complémentaire
```python
# Classification déséquilibrée
metrics = {
    'accuracy': accuracy_score(y_true, y_pred),
    'precision': precision_score(y_true, y_pred),
    'recall': recall_score(y_true, y_pred),
    'f1': f1_score(y_true, y_pred),
    'roc_auc': roc_auc_score(y_true, y_pred_proba)
}
```

#### 2. Métrique Principale + Métriques Secondaires

**Exemple Churn Prediction** :
- **Principale** : Recall (ne pas manquer clients à risque)
- **Secondaires** : Precision (éviter trop de faux positifs), F1, ROC-AUC

#### 3. Seuil de Performance Multi-Métriques

```python
# Un modèle est acceptable si :
acceptable = (
    recall >= 0.75 and      # Capture 75% des churns
    precision >= 0.60 and   # 60% de vrais positifs
    f1 >= 0.65             # Bon équilibre général
)
```

---

### Stratégies par Type de Problème

#### Classification Binaire Déséquilibrée (ex: Churn)
```python
primary_metrics = ['recall', 'f1_score']
secondary_metrics = ['precision', 'roc_auc']
monitoring_metrics = ['confusion_matrix', 'classification_report']
```

#### Régression (ex: Prix)
```python
primary_metrics = ['rmse', 'mae']
secondary_metrics = ['r2', 'mape']
visual_metrics = ['residual_plots', 'prediction_vs_actual']
```

#### Multi-classe
```python
primary_metrics = ['macro_f1', 'weighted_f1']
secondary_metrics = ['accuracy', 'per_class_recall']
monitoring_metrics = ['confusion_matrix']
```

---

## Métriques Business vs Métriques ML

### Différence Fondamentale

| Aspect | Métriques ML | Métriques Business |
|--------|--------------|-------------------|
| **Focus** | Performance du modèle | Impact business |
| **Exemples** | Accuracy, F1, RMSE | ROI, Revenue, Cost savings |
| **Audience** | Data Scientists | Stakeholders, Management |
| **Temporalité** | Immédiate | Long terme |

### Relier ML et Business

#### Exemple : Churn Prediction

**Métriques ML** :
- Recall = 0.75
- Precision = 0.60
- F1 = 0.67

**Traduction Business** :
- Coût moyen d'acquisition client : 500€
- Valeur vie client (CLV) : 2000€
- Coût campagne rétention : 50€

**Calcul ROI** :
```python
# Sur 1000 clients à risque détectés
TP = 750   # Recall = 0.75
FP = 500   # Precision = 0.60

# Avec intervention
clients_saved = TP * 0.50  # 50% sauvés = 375 clients
revenue_saved = clients_saved * 2000  # 750,000€

# Coûts
campaign_cost = 1250 * 50  # 62,500€

# ROI
roi = (revenue_saved - campaign_cost) / campaign_cost
# ROI = 11x → Excellent !
```

### Dashboard de Métriques Complète

```python
# 1. Métriques ML (pour l'équipe DS)
ml_metrics = {
    'accuracy': 0.85,
    'precision': 0.60,
    'recall': 0.75,
    'f1': 0.67,
    'roc_auc': 0.82
}

# 2. Métriques Business (pour stakeholders)
business_metrics = {
    'clients_saved': 375,
    'revenue_saved': '750K€',
    'campaign_cost': '62.5K€',
    'roi': '11x',
    'cost_per_save': '167€'
}

# 3. Métriques Opérationnelles (pour la production)
operational_metrics = {
    'prediction_latency': '50ms',
    'model_uptime': '99.9%',
    'data_quality_score': 0.95,
    'predictions_per_day': 10000
}
```

---

## Guide de Décision Rapide

### Pour Classification

```python
# Classes équilibrées + erreurs similaires
→ Accuracy + Confusion Matrix

# Classes déséquilibrées + FN coûteux (ex: churn, fraude)
→ Recall + F1 + ROC-AUC

# Classes déséquilibrées + FP coûteux (ex: spam)
→ Precision + F1 + PR-AUC

# Probabilités importantes
→ Log Loss + ROC-AUC

# Comparaison de modèles
→ ROC-AUC (ou PR-AUC si très déséquilibré)
```

### Pour Régression

```python
# Standard, interprétation
→ RMSE + R² + MAE

# Outliers présents
→ MAE + R² + Median Absolute Error

# Erreur relative importante
→ MAPE + RMSE

# Comparaison multi-datasets
→ MAPE + R²
```

---

## Bonnes Pratiques

### ✅ À Faire

1. **Toujours** utiliser plusieurs métriques complémentaires
2. **Aligner** métriques ML avec objectifs business
3. **Documenter** le choix des métriques et pourquoi
4. **Visualiser** : Confusion matrix, courbes ROC/PR, residual plots
5. **Monitorer** en production : drift des métriques
6. **Communiquer** différemment selon l'audience

### ❌ À Éviter

1. ❌ Se fier uniquement à l'accuracy sur classes déséquilibrées
2. ❌ Optimiser une métrique sans considérer le business
3. ❌ Ignorer les distributions de classes
4. ❌ Utiliser métriques de train pour comparer modèles (utiliser validation/test)
5. ❌ Oublier de calibrer les probabilités si nécessaires

---

## Exemple Complet : Pipeline d'Évaluation

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_classifier(y_true, y_pred, y_pred_proba):
    """Évaluation complète d'un classificateur"""
    
    # 1. Métriques de base
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1-Score': f1_score(y_true, y_pred),
        'ROC-AUC': roc_auc_score(y_true, y_pred_proba)
    }
    
    # 2. Afficher les métriques
    print("=== MÉTRIQUES DE PERFORMANCE ===")
    for name, value in metrics.items():
        print(f"{name}: {value:.3f}")
    
    # 3. Classification report
    print("\n=== CLASSIFICATION REPORT ===")
    print(classification_report(y_true, y_pred))
    
    # 4. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    # 5. ROC Curve
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {metrics["ROC-AUC"]:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()
    
    return metrics

# Utilisation
metrics = evaluate_classifier(y_test, y_pred, y_pred_proba[:, 1])
```

---

## Ressources Complémentaires

### Documentation
- [Scikit-learn Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Classification Metrics Explained](https://towardsdatascience.com/comprehensive-guide-on-multiclass-classification-metrics-af94cfb83fbd)

### Outils
- **MLflow** : Tracking des métriques
- **Weights & Biases** : Monitoring et visualisation
- **TensorBoard** : Pour Deep Learning

---

**Navigation**
- [← Précédent : Types de Tâches ML](02_ml_task_types.md)
- [Suivant : Guide Churn Prediction →](04_churn_prediction_guide.md)
- [Retour au README principal](../README.md)
