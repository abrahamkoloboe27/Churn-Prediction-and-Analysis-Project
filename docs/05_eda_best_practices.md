# 📊 Meilleures Pratiques pour l'Analyse Exploratoire des Données (EDA)

## Table des Matières
- [Introduction](#introduction)
- [Objectifs de l'EDA](#objectifs-de-leda)
- [Workflow Complet d'EDA](#workflow-complet-deda)
- [Analyse Univariée](#analyse-univariée)
- [Analyse Bivariée](#analyse-bivariée)
- [Analyse Multivariée](#analyse-multivariée)
- [Visualisations Recommandées](#visualisations-recommandées)
- [Détection d'Anomalies](#détection-danomalies)
- [Feature Engineering](#feature-engineering)
- [Outils et Bibliothèques](#outils-et-bibliothèques)
- [Checklist EDA](#checklist-eda)

## Introduction

L'**Analyse Exploratoire des Données** (EDA) est une étape cruciale avant toute modélisation. Elle permet de :
- Comprendre la structure des données
- Identifier les patterns et tendances
- Détecter les anomalies
- Formuler des hypothèses
- Préparer le feature engineering

> 💡 **Citation** : "Torture the data, and it will confess to anything" - Ronald Coase
> 
> L'EDA bien faite révèle la vérité des données sans les forcer !

## Objectifs de l'EDA

### 1. Comprendre les Données
- Structure du dataset (dimensions, types)
- Distribution des variables
- Relations entre variables

### 2. Identifier les Problèmes
- Valeurs manquantes
- Outliers
- Erreurs de données
- Déséquilibres

### 3. Générer des Insights
- Patterns intéressants
- Corrélations
- Segmentations naturelles

### 4. Préparer la Modélisation
- Features pertinentes
- Transformations nécessaires
- Stratégie de preprocessing

---

## Workflow Complet d'EDA

### Phase 1 : Premier Regard

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Configuration visualisations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Charger les données
df = pd.read_csv('data.csv')

# 1. Dimensions
print("=" * 50)
print("DIMENSIONS")
print("=" * 50)
print(f"Nombre de lignes: {df.shape[0]:,}")
print(f"Nombre de colonnes: {df.shape[1]:,}")
print(f"Total cellules: {df.shape[0] * df.shape[1]:,}")

# 2. Aperçu
print("\n" + "=" * 50)
print("APERÇU DES DONNÉES")
print("=" * 50)
print(df.head(10))

# 3. Info générale
print("\n" + "=" * 50)
print("INFORMATIONS GÉNÉRALES")
print("=" * 50)
print(df.info())

# 4. Types de données
print("\n" + "=" * 50)
print("TYPES DE DONNÉES")
print("=" * 50)
print(df.dtypes.value_counts())

# 5. Statistiques descriptives
print("\n" + "=" * 50)
print("STATISTIQUES DESCRIPTIVES")
print("=" * 50)
print(df.describe(include='all').T)
```

### Phase 2 : Qualité des Données

```python
def data_quality_report(df):
    """Rapport complet de qualité des données"""
    
    quality_df = pd.DataFrame({
        'dtype': df.dtypes,
        'missing_count': df.isnull().sum(),
        'missing_pct': (df.isnull().sum() / len(df)) * 100,
        'unique_count': df.nunique(),
        'unique_pct': (df.nunique() / len(df)) * 100
    })
    
    # Ajouter min/max pour numériques
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    quality_df.loc[numeric_cols, 'min'] = df[numeric_cols].min().values
    quality_df.loc[numeric_cols, 'max'] = df[numeric_cols].max().values
    quality_df.loc[numeric_cols, 'mean'] = df[numeric_cols].mean().values
    quality_df.loc[numeric_cols, 'median'] = df[numeric_cols].median().values
    
    # Trier par % manquants
    quality_df = quality_df.sort_values('missing_pct', ascending=False)
    
    print("=" * 80)
    print("RAPPORT DE QUALITÉ DES DONNÉES")
    print("=" * 80)
    print(quality_df)
    
    # Alertes
    print("\n" + "=" * 80)
    print("ALERTES")
    print("=" * 80)
    
    # Valeurs manquantes élevées
    high_missing = quality_df[quality_df['missing_pct'] > 50]
    if not high_missing.empty:
        print(f"⚠️ {len(high_missing)} colonnes avec >50% valeurs manquantes:")
        print(high_missing[['missing_pct']].to_string())
    
    # Colonnes constantes
    constant_cols = quality_df[quality_df['unique_count'] == 1]
    if not constant_cols.empty:
        print(f"\n⚠️ {len(constant_cols)} colonnes constantes (à supprimer):")
        print(constant_cols.index.tolist())
    
    # Colonnes quasi-identifiantes
    high_cardinality = quality_df[quality_df['unique_pct'] > 95]
    if not high_cardinality.empty:
        print(f"\n⚠️ {len(high_cardinality)} colonnes quasi-unique (possibles IDs):")
        print(high_cardinality.index.tolist())
    
    return quality_df

# Utilisation
quality_report = data_quality_report(df)
```

### Phase 3 : Gestion des Valeurs Manquantes

```python
def analyze_missing_data(df):
    """Analyse approfondie des valeurs manquantes"""
    
    # Patterns de valeurs manquantes
    missing_df = df.isnull()
    
    # Visualisation
    plt.figure(figsize=(14, 6))
    
    # Heatmap des valeurs manquantes
    plt.subplot(1, 2, 1)
    sns.heatmap(missing_df, cbar=False, yticklabels=False, cmap='viridis')
    plt.title('Pattern des Valeurs Manquantes')
    
    # Bar plot
    plt.subplot(1, 2, 2)
    missing_counts = df.isnull().sum().sort_values(ascending=False)
    missing_counts = missing_counts[missing_counts > 0]
    if not missing_counts.empty:
        missing_counts.plot(kind='barh')
        plt.title('Nombre de Valeurs Manquantes par Colonne')
        plt.xlabel('Count')
    
    plt.tight_layout()
    plt.show()
    
    # Corrélation entre valeurs manquantes
    if missing_df.sum().sum() > 0:  # Si des valeurs manquantes
        missing_corr = missing_df.corr()
        
        # Ne garder que les corrélations significatives
        mask = np.abs(missing_corr) > 0.3
        if mask.sum().sum() > len(mask):  # Plus que la diagonale
            plt.figure(figsize=(10, 8))
            sns.heatmap(missing_corr, annot=True, cmap='coolwarm', center=0, 
                       mask=~mask, linewidths=1)
            plt.title('Corrélation entre Patterns de Valeurs Manquantes')
            plt.show()

# Utilisation
analyze_missing_data(df)

# Stratégies de gestion
def handle_missing_values(df, strategy='smart'):
    """Gérer les valeurs manquantes intelligemment"""
    
    df_clean = df.copy()
    
    for col in df.columns:
        missing_pct = df[col].isnull().mean()
        
        if missing_pct == 0:
            continue
        
        elif missing_pct > 0.5:
            print(f"⚠️ {col}: {missing_pct*100:.1f}% manquant → Suppression colonne")
            df_clean.drop(col, axis=1, inplace=True)
        
        elif df[col].dtype in ['int64', 'float64']:
            # Numérique
            if missing_pct < 0.05:
                # Peu manquant → médiane
                df_clean[col].fillna(df[col].median(), inplace=True)
                print(f"✓ {col}: Rempli avec médiane")
            else:
                # Plus manquant → créer indicateur
                df_clean[f'{col}_is_missing'] = df[col].isnull().astype(int)
                df_clean[col].fillna(df[col].median(), inplace=True)
                print(f"✓ {col}: Rempli avec médiane + indicateur")
        
        else:
            # Catégoriel
            if missing_pct < 0.05:
                df_clean[col].fillna(df[col].mode()[0], inplace=True)
                print(f"✓ {col}: Rempli avec mode")
            else:
                df_clean[col].fillna('Missing', inplace=True)
                print(f"✓ {col}: Catégorie 'Missing' créée")
    
    return df_clean

# Utilisation
df_clean = handle_missing_values(df)
```

---

## Analyse Univariée

### Variables Numériques

```python
def analyze_numerical(df, col):
    """Analyse complète d'une variable numérique"""
    
    print("=" * 60)
    print(f"ANALYSE: {col}")
    print("=" * 60)
    
    # Statistiques
    print("\n--- Statistiques ---")
    stats = df[col].describe()
    print(stats)
    
    # Skewness et Kurtosis
    print(f"\nSkewness: {df[col].skew():.3f}")
    print(f"Kurtosis: {df[col].kurtosis():.3f}")
    
    # Visualisations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Histogramme
    axes[0, 0].hist(df[col].dropna(), bins=50, edgecolor='black')
    axes[0, 0].axvline(df[col].mean(), color='red', linestyle='--', label=f'Mean: {df[col].mean():.2f}')
    axes[0, 0].axvline(df[col].median(), color='green', linestyle='--', label=f'Median: {df[col].median():.2f}')
    axes[0, 0].set_title(f'Distribution de {col}')
    axes[0, 0].legend()
    
    # Box plot
    axes[0, 1].boxplot(df[col].dropna())
    axes[0, 1].set_title(f'Box Plot de {col}')
    
    # QQ plot (normalité)
    from scipy import stats as sp_stats
    sp_stats.probplot(df[col].dropna(), dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot (Test de Normalité)')
    
    # Distribution cumulative
    sorted_data = np.sort(df[col].dropna())
    cumulative = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    axes[1, 1].plot(sorted_data, cumulative)
    axes[1, 1].set_title('Distribution Cumulative')
    axes[1, 1].set_xlabel(col)
    axes[1, 1].set_ylabel('Cumulative Probability')
    
    plt.tight_layout()
    plt.show()
    
    # Détection outliers
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    print(f"\n--- Outliers (méthode IQR) ---")
    print(f"Nombre d'outliers: {len(outliers)} ({len(outliers)/len(df)*100:.2f}%)")
    print(f"Lower bound: {lower_bound:.2f}")
    print(f"Upper bound: {upper_bound:.2f}")

# Analyser toutes les variables numériques
numerical_cols = df.select_dtypes(include=[np.number]).columns
for col in numerical_cols:
    analyze_numerical(df, col)
```

### Variables Catégorielles

```python
def analyze_categorical(df, col, top_n=10):
    """Analyse complète d'une variable catégorielle"""
    
    print("=" * 60)
    print(f"ANALYSE: {col}")
    print("=" * 60)
    
    # Comptages
    value_counts = df[col].value_counts()
    value_props = df[col].value_counts(normalize=True) * 100
    
    print(f"\n--- Distribution (Top {top_n}) ---")
    summary_df = pd.DataFrame({
        'Count': value_counts.head(top_n),
        'Percentage': value_props.head(top_n)
    })
    print(summary_df)
    
    print(f"\nNombre de catégories uniques: {df[col].nunique()}")
    print(f"Mode: {df[col].mode()[0]}")
    
    # Visualisation
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar plot
    value_counts.head(top_n).plot(kind='barh', ax=axes[0])
    axes[0].set_title(f'Top {top_n} Catégories - {col}')
    axes[0].set_xlabel('Count')
    
    # Pie chart (pour petit nombre de catégories)
    if df[col].nunique() <= 10:
        value_counts.plot(kind='pie', ax=axes[1], autopct='%1.1f%%')
        axes[1].set_title(f'Distribution - {col}')
        axes[1].set_ylabel('')
    else:
        # Cumulative bar plot
        cumsum_pct = value_props.cumsum()
        ax2 = axes[1].twinx()
        value_props.head(top_n).plot(kind='bar', ax=axes[1], color='steelblue')
        cumsum_pct.head(top_n).plot(ax=ax2, color='red', marker='o')
        axes[1].set_title(f'Distribution et Cumulative - {col}')
        axes[1].set_xlabel('Category')
        axes[1].set_ylabel('Percentage', color='steelblue')
        ax2.set_ylabel('Cumulative %', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
    
    plt.tight_layout()
    plt.show()

# Analyser toutes les variables catégorielles
categorical_cols = df.select_dtypes(include=['object', 'category']).columns
for col in categorical_cols:
    analyze_categorical(df, col)
```

---

## Analyse Bivariée

### Variable Cible Numérique (Régression)

```python
def bivariate_numeric_target(df, feature, target):
    """Analyse bivariée avec cible numérique"""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    if df[feature].dtype in ['object', 'category']:
        # Feature catégorielle
        
        # Box plot par catégorie
        df.boxplot(column=target, by=feature, ax=axes[0])
        axes[0].set_title(f'{target} by {feature}')
        plt.sca(axes[0])
        plt.xticks(rotation=45)
        
        # Moyennes par catégorie
        means = df.groupby(feature)[target].mean().sort_values(ascending=False)
        means.plot(kind='barh', ax=axes[1])
        axes[1].set_title(f'Mean {target} by {feature}')
        
        # Violin plot
        if df[feature].nunique() <= 10:
            sns.violinplot(data=df, x=feature, y=target, ax=axes[2])
            axes[2].set_title(f'{target} Distribution by {feature}')
            plt.sca(axes[2])
            plt.xticks(rotation=45)
        
    else:
        # Feature numérique
        
        # Scatter plot
        axes[0].scatter(df[feature], df[target], alpha=0.5)
        axes[0].set_xlabel(feature)
        axes[0].set_ylabel(target)
        axes[0].set_title(f'{target} vs {feature}')
        
        # Regression line
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            df[feature].dropna(), df.loc[df[feature].notna(), target]
        )
        line = slope * df[feature] + intercept
        axes[0].plot(df[feature], line, 'r', label=f'R²={r_value**2:.3f}')
        axes[0].legend()
        
        # Hexbin (pour beaucoup de points)
        axes[1].hexbin(df[feature], df[target], gridsize=30, cmap='Blues')
        axes[1].set_title(f'Density: {target} vs {feature}')
        
        # Corrélation avec bins
        df_temp = df[[feature, target]].copy()
        df_temp['feature_bins'] = pd.qcut(df_temp[feature], q=10, duplicates='drop')
        bin_means = df_temp.groupby('feature_bins')[target].mean()
        bin_means.plot(kind='line', marker='o', ax=axes[2])
        axes[2].set_title(f'Mean {target} by {feature} (binned)')
        plt.sca(axes[2])
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # Statistiques
    if df[feature].dtype not in ['object', 'category']:
        corr = df[[feature, target]].corr().iloc[0, 1]
        print(f"Corrélation {feature} vs {target}: {corr:.3f}")

# Utilisation
for feature in df.drop(target_column, axis=1).columns:
    bivariate_numeric_target(df, feature, target_column)
```

### Variable Cible Catégorielle (Classification)

```python
def bivariate_categorical_target(df, feature, target):
    """Analyse bivariée avec cible catégorielle"""
    
    print(f"\n{'='*60}")
    print(f"ANALYSE: {feature} vs {target}")
    print(f"{'='*60}")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    if df[feature].dtype in ['object', 'category']:
        # Feature catégorielle vs target catégorielle
        
        # Crosstab
        ct = pd.crosstab(df[feature], df[target], normalize='index') * 100
        print("\n--- Taux par catégorie (%) ---")
        print(ct)
        
        # Stacked bar
        ct.plot(kind='bar', stacked=True, ax=axes[0])
        axes[0].set_title(f'{target} Distribution by {feature}')
        axes[0].set_ylabel('Percentage')
        plt.sca(axes[0])
        plt.xticks(rotation=45)
        
        # Grouped bar
        ct.plot(kind='bar', ax=axes[1])
        axes[1].set_title(f'{target} Rate by {feature}')
        axes[1].set_ylabel('Percentage')
        plt.sca(axes[1])
        plt.xticks(rotation=45)
        
        # Heatmap
        ct_counts = pd.crosstab(df[feature], df[target])
        sns.heatmap(ct_counts, annot=True, fmt='d', cmap='Blues', ax=axes[2])
        axes[2].set_title(f'Counts: {feature} vs {target}')
        
    else:
        # Feature numérique vs target catégorielle
        
        # Distribution par classe
        for class_val in df[target].unique():
            df[df[target] == class_val][feature].hist(
                bins=30, alpha=0.5, label=f'{target}={class_val}', ax=axes[0]
            )
        axes[0].set_title(f'{feature} Distribution by {target}')
        axes[0].legend()
        
        # Box plot
        df.boxplot(column=feature, by=target, ax=axes[1])
        axes[1].set_title(f'{feature} by {target}')
        
        # Violin plot
        sns.violinplot(data=df, x=target, y=feature, ax=axes[2])
        axes[2].set_title(f'{feature} Distribution by {target}')
        
        # Test statistique
        from scipy import stats
        groups = [df[df[target] == val][feature].dropna() for val in df[target].unique()]
        if len(groups) == 2:
            stat, pval = stats.ttest_ind(*groups)
            print(f"\nT-test: statistic={stat:.3f}, p-value={pval:.4f}")
        else:
            stat, pval = stats.f_oneway(*groups)
            print(f"\nANOVA: F-statistic={stat:.3f}, p-value={pval:.4f}")
    
    plt.tight_layout()
    plt.show()

# Utilisation pour churn
for feature in df.drop('Churn', axis=1).columns:
    bivariate_categorical_target(df, feature, 'Churn')
```

---

## Analyse Multivariée

### Matrice de Corrélation

```python
def correlation_analysis(df):
    """Analyse des corrélations"""
    
    # Sélectionner uniquement numériques
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Calculer corrélations
    corr_matrix = numeric_df.corr()
    
    # Visualisation
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                fmt='.2f', square=True, linewidths=1)
    plt.title('Matrice de Corrélation')
    plt.tight_layout()
    plt.show()
    
    # Paires hautement corrélées
    print("\n" + "=" * 60)
    print("PAIRES HAUTEMENT CORRÉLÉES (|r| > 0.7)")
    print("=" * 60)
    
    # Extraire triangulaire supérieur
    upper_tri = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    # Trouver corrélations élevées
    high_corr = [
        (column, row, upper_tri[column][row])
        for column in upper_tri.columns
        for row in upper_tri.index
        if abs(upper_tri[column][row]) > 0.7
    ]
    
    if high_corr:
        high_corr_df = pd.DataFrame(high_corr, columns=['Feature 1', 'Feature 2', 'Correlation'])
        high_corr_df = high_corr_df.sort_values('Correlation', key=abs, ascending=False)
        print(high_corr_df.to_string(index=False))
    else:
        print("Aucune corrélation élevée trouvée.")
    
    return corr_matrix

# Utilisation
corr_matrix = correlation_analysis(df)
```

### PCA pour Visualisation

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def pca_visualization(df, target_col=None, n_components=2):
    """Visualisation PCA"""
    
    # Préparer les données
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if target_col and target_col in numeric_cols:
        numeric_cols = numeric_cols.drop(target_col)
    
    X = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # Standardiser
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    # Variance expliquée
    print(f"Variance expliquée par PC1: {pca.explained_variance_ratio_[0]*100:.2f}%")
    print(f"Variance expliquée par PC2: {pca.explained_variance_ratio_[1]*100:.2f}%")
    print(f"Variance totale expliquée: {sum(pca.explained_variance_ratio_)*100:.2f}%")
    
    # Visualisation
    plt.figure(figsize=(10, 6))
    
    if target_col:
        for label in df[target_col].unique():
            mask = df[target_col] == label
            plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=f'{target_col}={label}', alpha=0.6)
        plt.legend()
    else:
        plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6)
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    plt.title('PCA Projection')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return pca, X_pca

# Utilisation
pca, X_pca = pca_visualization(df, target_col='Churn')
```

---

## Visualisations Recommandées

### Pairplot pour Relations Multiples

```python
# Pour subset de variables
important_features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Churn']
sns.pairplot(df[important_features], hue='Churn', diag_kind='kde', corner=True)
plt.suptitle('Pairplot des Features Principales', y=1.02)
plt.show()
```

### Visualisations Interactives avec Plotly

```python
import plotly.express as px
import plotly.graph_objects as go

# Scatter interactif
fig = px.scatter(df, x='tenure', y='MonthlyCharges', color='Churn',
                 hover_data=['Contract', 'InternetService'],
                 title='Tenure vs Monthly Charges (Interactive)')
fig.show()

# Distribution interactive
fig = px.histogram(df, x='MonthlyCharges', color='Churn', 
                   marginal='box', nbins=50,
                   title='Distribution of Monthly Charges by Churn')
fig.show()

# Parallel coordinates
fig = px.parallel_coordinates(
    df.sample(500),  # Sample pour performance
    dimensions=['tenure', 'MonthlyCharges', 'TotalCharges'],
    color='Churn',
    title='Parallel Coordinates Plot'
)
fig.show()
```

---

## Détection d'Anomalies

```python
def detect_outliers(df, method='iqr'):
    """Détecter les outliers"""
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outliers_dict = {}
    
    for col in numeric_cols:
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower) | (df[col] > upper)]
            
        elif method == 'zscore':
            from scipy import stats
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            outliers = df[z_scores > 3]
        
        if len(outliers) > 0:
            outliers_dict[col] = {
                'count': len(outliers),
                'percentage': len(outliers) / len(df) * 100,
                'indices': outliers.index.tolist()
            }
    
    return outliers_dict

# Utilisation
outliers = detect_outliers(df, method='iqr')
for col, info in outliers.items():
    print(f"{col}: {info['count']} outliers ({info['percentage']:.2f}%)")
```

---

## Feature Engineering

### Créer de Nouvelles Features

```python
def feature_engineering_examples(df):
    """Exemples de feature engineering"""
    
    df_new = df.copy()
    
    # 1. Binning
    df_new['tenure_group'] = pd.cut(df['tenure'], 
                                      bins=[0, 12, 24, 48, 100],
                                      labels=['0-1y', '1-2y', '2-4y', '4y+'])
    
    # 2. Ratios
    df_new['charges_per_tenure'] = df['TotalCharges'] / (df['tenure'] + 1)
    
    # 3. Interactions
    df_new['is_new_expensive'] = ((df['tenure'] < 12) & 
                                   (df['MonthlyCharges'] > df['MonthlyCharges'].median())).astype(int)
    
    # 4. Agrégations
    service_cols = [col for col in df.columns if 'Service' in col or 'Support' in col]
    df_new['total_services'] = df[service_cols].apply(lambda x: (x == 'Yes').sum(), axis=1)
    
    # 5. Polynomial features (avec prudence)
    df_new['tenure_squared'] = df['tenure'] ** 2
    
    return df_new

df_engineered = feature_engineering_examples(df)
```

---

## Outils et Bibliothèques

### Pandas Profiling (ydata-profiling)

```python
# Installation: pip install ydata-profiling
from ydata_profiling import ProfileReport

profile = ProfileReport(df, title="Churn Dataset Report", explorative=True)
profile.to_file("eda_report.html")
```

### Sweetviz

```python
# Installation: pip install sweetviz
import sweetviz as sv

report = sv.analyze(df)
report.show_html("sweetviz_report.html")

# Comparaison train/test
report = sv.compare([train_df, "Train"], [test_df, "Test"])
report.show_html("comparison_report.html")
```

### AutoViz

```python
# Installation: pip install autoviz
from autoviz.AutoViz_Class import AutoViz_Class

AV = AutoViz_Class()
df_viz = AV.AutoViz("data.csv")
```

---

## Checklist EDA

### ✅ Phase 1 : Compréhension Initiale
- [ ] Charger et afficher les premières lignes
- [ ] Vérifier les dimensions (lignes × colonnes)
- [ ] Identifier les types de données
- [ ] Générer statistiques descriptives

### ✅ Phase 2 : Qualité des Données
- [ ] Vérifier les valeurs manquantes
- [ ] Détecter les doublons
- [ ] Identifier les outliers
- [ ] Vérifier la cohérence des données
- [ ] Analyser les valeurs constantes/quasi-constantes

### ✅ Phase 3 : Analyse Univariée
- [ ] Distribution de la variable cible
- [ ] Distributions des variables numériques
- [ ] Fréquences des variables catégorielles
- [ ] Tests de normalité si nécessaire

### ✅ Phase 4 : Analyse Bivariée
- [ ] Corrélations entre features numériques
- [ ] Relation chaque feature vs target
- [ ] Chi-square pour catégorielles
- [ ] Tests statistiques appropriés

### ✅ Phase 5 : Analyse Multivariée
- [ ] Matrice de corrélation complète
- [ ] PCA/t-SNE pour visualisation
- [ ] Interactions entre features
- [ ] Pairplots pour features importantes

### ✅ Phase 6 : Insights et Actions
- [ ] Documenter les patterns trouvés
- [ ] Lister les features importantes
- [ ] Planifier le feature engineering
- [ ] Définir la stratégie de preprocessing

---

## Bonnes Pratiques

### ✅ À Faire

1. **Documenter tout** : Notez vos observations dans des markdown cells
2. **Visualiser systématiquement** : Ne pas se fier qu'aux chiffres
3. **Vérifier les hypothèses** : Ne pas supposer, vérifier
4. **Itérer** : L'EDA n'est jamais "terminée"
5. **Partager** : Communiquer les insights à l'équipe

### ❌ À Éviter

1. ❌ Regarder uniquement les moyennes (ignorer distributions)
2. ❌ Négliger les valeurs manquantes
3. ❌ Faire des transformations avant de comprendre
4. ❌ Utiliser trop de visualisations similaires
5. ❌ Oublier le contexte business

---

## Template Complet d'EDA

```python
"""
TEMPLATE EDA COMPLET
"""

# 1. IMPORTS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 2. CONFIGURATION
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# 3. CHARGEMENT
df = pd.read_csv('data.csv')
print(f"Dataset: {df.shape[0]} rows × {df.shape[1]} columns")

# 4. OVERVIEW
print(df.head())
print(df.info())
print(df.describe())

# 5. QUALITÉ
quality_report = data_quality_report(df)

# 6. MISSING VALUES
analyze_missing_data(df)
df = handle_missing_values(df)

# 7. UNIVARIÉE
for col in df.select_dtypes(include=[np.number]).columns:
    analyze_numerical(df, col)

for col in df.select_dtypes(include=['object']).columns:
    analyze_categorical(df, col)

# 8. BIVARIÉE (vs target)
for col in df.drop('target', axis=1).columns:
    bivariate_categorical_target(df, col, 'target')

# 9. MULTIVARIÉE
corr_matrix = correlation_analysis(df)
pca, X_pca = pca_visualization(df, target_col='target')

# 10. FEATURE ENGINEERING
df_final = feature_engineering_examples(df)

# 11. SAVE
df_final.to_csv('data_processed.csv', index=False)
print("✓ EDA Complete!")
```

---

**Navigation**
- [← Précédent : Guide Churn Prediction](04_churn_prediction_guide.md)
- [Suivant : Déploiement et Recommandations →](06_model_deployment_recommendations.md)
- [Retour au README principal](../README.md)
