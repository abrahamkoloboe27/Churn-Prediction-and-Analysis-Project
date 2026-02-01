# 🤝 Guide de Contribution

Merci de votre intérêt pour contribuer au projet Churn Prediction and Analysis ! Ce document vous guide à travers le processus de contribution.

## 📋 Table des Matières

- [Code de Conduite](#code-de-conduite)
- [Comment Contribuer](#comment-contribuer)
- [Processus de Développement](#processus-de-développement)
- [Standards de Code](#standards-de-code)
- [Soumettre une Pull Request](#soumettre-une-pull-request)
- [Signaler un Bug](#signaler-un-bug)
- [Proposer une Fonctionnalité](#proposer-une-fonctionnalité)

## Code de Conduite

Ce projet adhère à un code de conduite. En participant, vous vous engagez à respecter ce code. Veuillez signaler tout comportement inacceptable.

### Nos Engagements

- Être respectueux et inclusif
- Accepter les critiques constructives
- Se concentrer sur ce qui est le mieux pour la communauté
- Faire preuve d'empathie envers les autres membres

## Comment Contribuer

Il existe plusieurs façons de contribuer à ce projet :

### 1. 📝 Documentation

- Corriger des fautes de frappe ou des erreurs
- Améliorer la clarté des explications
- Ajouter des exemples
- Traduire la documentation

### 2. 🐛 Correction de Bugs

- Signaler des bugs via les Issues
- Corriger des bugs existants
- Améliorer les tests

### 3. ✨ Nouvelles Fonctionnalités

- Proposer de nouvelles fonctionnalités
- Implémenter des fonctionnalités approuvées
- Améliorer les fonctionnalités existantes

### 4. 🧪 Tests

- Ajouter des tests unitaires
- Améliorer la couverture de tests
- Créer des tests d'intégration

### 5. 🎨 Interface Utilisateur

- Améliorer l'interface Streamlit
- Ajouter des visualisations
- Optimiser l'UX

## Processus de Développement

### 1. Fork et Clone

```bash
# Fork le projet sur GitHub, puis :
git clone https://github.com/VOTRE-USERNAME/Churn-Prediction-and-Analysis-Project.git
cd Churn-Prediction-and-Analysis-Project
```

### 2. Créer un Environnement

```bash
# Créer un environnement virtuel
python -m venv venv

# Activer l'environnement
# Sur Linux/Mac:
source venv/bin/activate
# Sur Windows:
venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt
```

### 3. Créer une Branche

```bash
# Créer une branche pour votre fonctionnalité ou correction
git checkout -b feature/ma-nouvelle-fonctionnalite
# ou
git checkout -b fix/correction-bug
```

### 4. Faire vos Changements

- Écrivez du code clair et documenté
- Suivez les standards de code (voir ci-dessous)
- Ajoutez des tests si nécessaire
- Mettez à jour la documentation

### 5. Tester

```bash
# Lancer l'application localement
streamlit run src/🏚️Accueil.py

# Tester le build Docker
docker build -t test-app .
docker run -p 8501:8501 test-app
```

### 6. Commit et Push

```bash
# Stager vos changements
git add .

# Commiter avec un message descriptif
git commit -m "feat: ajouter nouvelle visualisation pour le dashboard"

# Pousser vers votre fork
git push origin feature/ma-nouvelle-fonctionnalite
```

### 7. Créer une Pull Request

- Allez sur GitHub et créez une Pull Request
- Décrivez clairement vos changements
- Référencez les issues liées si applicable
- Attendez la revue de code

## Standards de Code

### Style Python

Nous suivons les conventions PEP 8 avec quelques adaptations :

```python
# ✅ Bon
def calculate_churn_probability(customer_data):
    """
    Calcule la probabilité de churn pour un client.
    
    Args:
        customer_data (pd.DataFrame): Données du client
        
    Returns:
        float: Probabilité de churn entre 0 et 1
    """
    # Implémentation
    pass

# ❌ Mauvais
def calc_prob(data):
    # Pas de docstring, nom peu clair
    pass
```

### Conventions de Nommage

- **Variables et fonctions** : `snake_case`
- **Classes** : `PascalCase`
- **Constants** : `UPPER_CASE`
- **Fichiers** : `snake_case.py`

### Documentation

Toutes les fonctions publiques doivent avoir une docstring :

```python
def predict_churn(model, features):
    """
    Prédit le churn pour un ensemble de features.
    
    Args:
        model: Modèle ML entraîné
        features (pd.DataFrame): Features du client
        
    Returns:
        tuple: (prediction, probability)
        
    Raises:
        ValueError: Si les features sont invalides
        
    Example:
        >>> features = pd.DataFrame({'tenure': [12], 'charges': [70.0]})
        >>> pred, proba = predict_churn(model, features)
        >>> print(f"Prediction: {pred}, Probability: {proba:.2f}")
    """
    # Implementation
    pass
```

### Commentaires

- Expliquez le **pourquoi**, pas le **quoi**
- Gardez les commentaires à jour
- Utilisez des commentaires pour les sections complexes

```python
# ✅ Bon
# Utiliser SMOTE pour équilibrer les classes car le dataset est très déséquilibré (20% churn)
X_balanced, y_balanced = smote.fit_resample(X, y)

# ❌ Mauvais
# Appliquer SMOTE
X_balanced, y_balanced = smote.fit_resample(X, y)
```

## Soumettre une Pull Request

### Checklist avant Soumission

- [ ] Le code suit les standards du projet
- [ ] Tous les tests passent
- [ ] La documentation est à jour
- [ ] Le commit message est clair
- [ ] Pas de fichiers inutiles (`.pyc`, `.DS_Store`, etc.)
- [ ] Le build Docker fonctionne

### Format du Message de Commit

Utilisez des commits conventionnels :

```
type(scope): description courte

Description détaillée si nécessaire

Fixes #123
```

**Types** :
- `feat`: Nouvelle fonctionnalité
- `fix`: Correction de bug
- `docs`: Documentation
- `style`: Formatage (pas de changement de code)
- `refactor`: Refactoring
- `test`: Ajout de tests
- `chore`: Tâches de maintenance

**Exemples** :
```
feat(dashboard): ajouter graphique de distribution par contrat
fix(prediction): corriger le calcul de probabilité pour nouveaux clients
docs(readme): mettre à jour les instructions d'installation
```

### Processus de Revue

1. **Soumission** : Créez votre PR avec une description claire
2. **CI/CD** : Les tests automatiques s'exécutent
3. **Revue** : Un mainteneur examine votre code
4. **Modifications** : Apportez les changements demandés
5. **Merge** : Votre PR est fusionnée !

## Signaler un Bug

### Avant de Signaler

- Vérifiez que le bug n'est pas déjà signalé
- Assurez-vous que c'est bien un bug (pas un comportement attendu)
- Testez avec la dernière version

### Template de Bug Report

```markdown
**Description du Bug**
Description claire et concise du bug.

**Reproduction**
Étapes pour reproduire le comportement :
1. Aller à '...'
2. Cliquer sur '...'
3. Faire défiler jusqu'à '...'
4. Voir l'erreur

**Comportement Attendu**
Ce qui devrait se passer.

**Comportement Actuel**
Ce qui se passe réellement.

**Screenshots**
Si applicable, ajoutez des screenshots.

**Environnement**
- OS: [ex: macOS 13.0]
- Python: [ex: 3.9.7]
- Navigateur: [ex: Chrome 120]

**Informations Additionnelles**
Tout autre contexte pertinent.
```

## Proposer une Fonctionnalité

### Avant de Proposer

- Vérifiez qu'elle n'est pas déjà proposée
- Assurez-vous qu'elle s'aligne avec les objectifs du projet
- Réfléchissez à l'implémentation

### Template de Feature Request

```markdown
**Est-ce lié à un problème ?**
Description claire du problème. Ex: Je suis frustré quand [...]

**Solution Proposée**
Description claire de ce que vous voulez.

**Alternatives Considérées**
Autres solutions envisagées.

**Contexte Additionnel**
Screenshots, exemples, etc.

**Implémentation**
Idées sur comment implémenter (optionnel).
```

## Questions ?

Si vous avez des questions sur le processus de contribution :

1. Consultez la [documentation](docs/)
2. Ouvrez une [issue](https://github.com/abrahamkoloboe27/Churn-Prediction-and-Analysis-Project/issues) avec le tag `question`
3. Contactez les mainteneurs

## Remerciements

Merci à tous nos contributeurs ! 🎉

Votre temps et vos efforts sont grandement appréciés.

---

**Happy Contributing! 🚀**
