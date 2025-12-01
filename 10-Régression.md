# Chapitre 10 : Apprentissage supervisé - Régression

- [Chapitre 10 : Apprentissage supervisé - Régression](#chapitre-10--apprentissage-supervisé---régression)
  - [Introduction à la régression](#introduction-à-la-régression)
  - [Méthodologie générale](#méthodologie-générale)
    - [Préparation de l’échantillon](#préparation-de-léchantillon)
  - [Méthodes de régression](#méthodes-de-régression)
    - [Régression linéaire simple](#régression-linéaire-simple)
    - [Régression linéaire multiple](#régression-linéaire-multiple)
    - [Régression polynomiale](#régression-polynomiale)
    - [Ridge (régression avec régularisation L2)](#ridge-régression-avec-régularisation-l2)
    - [Lasso (régression avec régularisation L1)](#lasso-régression-avec-régularisation-l1)
    - [Elastic Net](#elastic-net)
    - [Arbre de régression](#arbre-de-régression)
    - [Random Forest (régression)](#random-forest-régression)
    - [Support Vector Regressor (SVR)](#support-vector-regressor-svr)
  - [Evaluation des modèles](#evaluation-des-modèles)
    - [Erreur quadratique moyenne (MSE)](#erreur-quadratique-moyenne-mse)
    - [Root Mean Squared Error (RMSE)](#root-mean-squared-error-rmse)
    - [Mean Absolute Error (MAE)](#mean-absolute-error-mae)
    - [R² (coefficient de détermination)](#r-coefficient-de-détermination)
    - [Exemple illustratif](#exemple-illustratif)
  - [Compromis sous-apprentissage / sur-apprentissage](#compromis-sous-apprentissage--sur-apprentissage)
  - [Validation croisée](#validation-croisée)
  - [Points clés à retenir](#points-clés-à-retenir)
  - [Régression Linéaire](#régression-linéaire)
    - [Introduction](#introduction)
    - [Configuration et méthodologie](#configuration-et-méthodologie)
    - [Modèle : régression linéaire](#modèle--régression-linéaire)
      - [Création et entraînement du modèle](#création-et-entraînement-du-modèle)
      - [Coefficients et intercept](#coefficients-et-intercept)
      - [Prédictions sur le jeu de test](#prédictions-sur-le-jeu-de-test)
      - [Évaluation du modèle](#évaluation-du-modèle)
      - [Visualisation des valeurs observées vs prédites](#visualisation-des-valeurs-observées-vs-prédites)
      - [Notes sur la régression multiple](#notes-sur-la-régression-multiple)
  - [Régression Polynomiale](#régression-polynomiale-1)
    - [Introduction](#introduction-1)
    - [Transformation polynomiale des variables](#transformation-polynomiale-des-variables)
    - [Création et entraînement du modèle](#création-et-entraînement-du-modèle-1)
    - [Coefficients et intercept](#coefficients-et-intercept-1)
    - [Prédictions sur le jeu de test](#prédictions-sur-le-jeu-de-test-1)
    - [Évaluation du modèle](#évaluation-du-modèle-1)
    - [Visualisation des valeurs observées vs prédites](#visualisation-des-valeurs-observées-vs-prédites-1)
    - [Notes importantes](#notes-importantes)
  - [Régression Ridge](#régression-ridge)
    - [Introduction](#introduction-2)
    - [Création du modèle Ridge](#création-du-modèle-ridge)
    - [Grid Search pour optimiser l’hyperparamètre alpha](#grid-search-pour-optimiser-lhyperparamètre-alpha)
    - [Prédictions sur le jeu de test](#prédictions-sur-le-jeu-de-test-2)
    - [Évaluation du modèle](#évaluation-du-modèle-2)
    - [Coefficients et intercept](#coefficients-et-intercept-2)
    - [Visualisation des valeurs observées vs prédites](#visualisation-des-valeurs-observées-vs-prédites-2)
    - [Notes importantes](#notes-importantes-1)
  - [Régression Lasso](#régression-lasso)
    - [Introduction](#introduction-3)
    - [Création du modèle Lasso](#création-du-modèle-lasso)
    - [Grid Search pour optimiser l’hyperparamètre alpha](#grid-search-pour-optimiser-lhyperparamètre-alpha-1)
    - [Prédictions sur le jeu de test](#prédictions-sur-le-jeu-de-test-3)
    - [Évaluation du modèle](#évaluation-du-modèle-3)
    - [Coefficients et intercept](#coefficients-et-intercept-3)
    - [Visualisation des valeurs observées vs prédites](#visualisation-des-valeurs-observées-vs-prédites-3)
    - [Notes importantes](#notes-importantes-2)
  - [Régression ElasticNet](#régression-elasticnet)
    - [Introduction](#introduction-4)
    - [Création du modèle ElasticNet](#création-du-modèle-elasticnet)
    - [Grid Search pour optimiser les hyperparamètres](#grid-search-pour-optimiser-les-hyperparamètres)
    - [Prédictions sur le jeu de test](#prédictions-sur-le-jeu-de-test-4)
    - [Évaluation du modèle](#évaluation-du-modèle-4)
    - [Coefficients et intercept](#coefficients-et-intercept-4)
    - [Visualisation des valeurs observées vs prédites](#visualisation-des-valeurs-observées-vs-prédites-4)
    - [Notes importantes](#notes-importantes-3)
  - [Arbre de Régression — Projet DPE](#arbre-de-régression--projet-dpe)
    - [Introduction](#introduction-5)
    - [Création et entraînement du modèle](#création-et-entraînement-du-modèle-2)
    - [Prédictions sur le jeu de test](#prédictions-sur-le-jeu-de-test-5)
    - [Évaluation du modèle](#évaluation-du-modèle-5)
    - [Visualisation de l'arbre](#visualisation-de-larbre)
    - [Interprétation](#interprétation)
    - [Points clés](#points-clés)
  - [Random Forest Regressor](#random-forest-regressor)
    - [Introduction](#introduction-6)
    - [Création et entraînement du modèle](#création-et-entraînement-du-modèle-3)
    - [Prédictions sur le jeu de test](#prédictions-sur-le-jeu-de-test-6)
    - [Évaluation du modèle](#évaluation-du-modèle-6)
    - [Importance des variables](#importance-des-variables)
    - [Interprétation](#interprétation-1)
    - [Points clés](#points-clés-1)
  - [Support Vector Regressor (SVR)](#support-vector-regressor-svr-1)
    - [Introduction](#introduction-7)
    - [Configuration et Grid Search](#configuration-et-grid-search)
    - [Meilleur modèle et hyperparamètres](#meilleur-modèle-et-hyperparamètres)
    - [Prédictions et évaluation](#prédictions-et-évaluation)
    - [Interprétation](#interprétation-2)
    - [Points clés](#points-clés-2)
  - [XGBoost Regressor](#xgboost-regressor)
    - [Introduction](#introduction-8)
    - [Configuration et Grid Search](#configuration-et-grid-search-1)
    - [Meilleur modèle et hyperparamètres](#meilleur-modèle-et-hyperparamètres-1)
    - [Prédictions et évaluation](#prédictions-et-évaluation-1)
    - [Interprétation](#interprétation-3)
    - [Points clés](#points-clés-3)
  - [En bref](#en-bref)
  - [Récapitulatif](#récapitulatif)
- [Exercice : Prédiction de la consommation énergétique](#exercice--prédiction-de-la-consommation-énergétique)

## Introduction à la régression

La **régression** est une sous-catégorie de l’apprentissage supervisé. Elle consiste à prédire une **variable cible continue** à partir de variables explicatives. L'objectif est d’estimer une valeur numérique pour chaque observation.

**Exemple**

- Variables explicatives : surface habitable, étage, type de logement, année de construction, etc.
- Variable cible : `besoin_chauffage` (kWh/an), `conso_chauffage_ef` (kWh/an), ou consommation énergétique globale.


## Méthodologie générale

### Préparation de l’échantillon

1. **Sélection des variables explicatives et de la cible**  
   - Identifier les variables pertinentes pour prédire la cible.  
   - Exemple : `X = [surface, etage, type_logement, année_construction]`, `Y = besoin_chauffage`.

2. **Séparation train / test**  
   - Entraînement : pour apprendre le modèle.  
   - Test : pour évaluer la généralisation.  
   - Exemple de répartition : 70% train, 30% test.

| Ensemble | Nombre de logements | Moyenne consommation (kWh) |
|----------|-------------------|----------------------------|
| Train    | 700               | 18 500                     |
| Test     | 300               | 18 700                     |

3. **Prétraitement**  
   - Gestion des valeurs manquantes (imputation moyenne, médiane, etc.)  
   - Encodage des variables catégorielles  
   - Normalisation / standardisation pour certaines méthodes (SVM, Ridge, Lasso)  


## Méthodes de régression

### Régression linéaire simple
- Principe : modèle linéaire reliant une variable explicative à la cible.  
- Avantages : facile à comprendre, rapide à entraîner, interprétation directe des coefficients.  
- Inconvénients : suppose une relation linéaire, sensible aux valeurs aberrantes.

### Régression linéaire multiple
- Principe : extension de la régression simple à plusieurs variables explicatives.  
- Avantages : permet de prendre en compte plusieurs facteurs simultanément.  
- Inconvénients : multicolinéarité possible, linéarité supposée, sensible aux outliers.

### Régression polynomiale
- Principe : transformation des variables explicatives en polynômes pour modéliser des relations non linéaires.  
- Avantages : capture les relations courbées.  
- Inconvénients : risque de surapprentissage si degré trop élevé.

### Ridge (régression avec régularisation L2)
- Principe : pénalise la somme des carrés des coefficients pour réduire le surapprentissage.  
- Avantages : stabilise le modèle en présence de multicolinéarité.  
- Inconvénients : ne permet pas de supprimer des variables (coefficients proches de zéro mais pas exactement zéro).

### Lasso (régression avec régularisation L1)
- Principe : pénalise la somme des valeurs absolues des coefficients.  
- Avantages : peut supprimer certaines variables (coefficients à zéro), facilite l’interprétation.  
- Inconvénients : moins stable si variables fortement corrélées.

### Elastic Net
- Principe : combinaison de L1 et L2 pour régulariser le modèle.  
- Avantages : profite des avantages de Ridge et Lasso, stable en présence de multicolinéarité.  
- Inconvénients : hyperparamètres à régler (ratio L1/L2).

### Arbre de régression
- Principe : divise l’espace des variables pour créer des règles qui prédisent des valeurs continues.  
- Avantages : non linéaire, facile à interpréter visuellement.  
- Inconvénients : tendance au surapprentissage si l’arbre est profond.

### Random Forest (régression)
- Principe : ensemble d’arbres de régression construits sur des sous-échantillons du jeu de données (bagging).  
- Avantages : réduit le surapprentissage, robuste, calcule l’importance des variables.  
- Inconvénients : moins interprétable qu’un arbre unique, plus lent.

### Support Vector Regressor (SVR)
- Principe : similaire au SVM mais pour la régression, optimise une marge autour de la prédiction.  
- Avantages : performant pour relations complexes non linéaires via kernels.  
- Inconvénients : paramétrage délicat (C, gamma, epsilon), sensible à l’échelle des variables.


## Evaluation des modèles

### Erreur quadratique moyenne (MSE)
- Moyenne des carrés des différences entre valeurs observées et prédites.  
- Formule : `MSE = Σ (y_i - ŷ_i)^2 / n`  
- Sensible aux valeurs extrêmes.

### Root Mean Squared Error (RMSE)
- Racine carrée de la MSE, même unité que la variable cible.  
- Plus interprétable que la MSE.

### Mean Absolute Error (MAE)
- Moyenne des différences absolues entre valeurs observées et prédites.  
- Moins sensible aux outliers que la MSE.

### R² (coefficient de détermination)
- Proportion de variance expliquée par le modèle.  
- `R² = 1 - (SS_res / SS_tot)`  
- R² proche de 1 → modèle explique bien la variance.


### Exemple illustratif

**Jeu de données DPE : prédire `besoin_chauffage`**

| Observé | Prévu (modèle linéaire) | Erreur |
|---------|-------------------------|--------|
| 20 000  | 19 500                  | 500    |
| 15 000  | 14 800                  | 200    |
| 18 000  | 18 200                  | -200   |

- MSE : (500² + 200² + (-200)²)/3 = 166 666 kWh²  
- RMSE : √166 666 ≈ 408 kWh  
- MAE : (500 + 200 + 200)/3 ≈ 300 kWh  
- R² : 0,95 (le modèle explique 95% de la variance)


## Compromis sous-apprentissage / sur-apprentissage

- **Sous-apprentissage** : modèle trop simple pour capturer la variance de la cible.  
  - Exemple : régression linéaire simple avec une seule variable alors que la relation est complexe.  
- **Sur-apprentissage** : modèle trop flexible, prédit parfaitement le train mais pas le test.  
  - Exemple : arbre de régression très profond ou régression polynomiale de degré élevé.  

**Solutions** : régularisation (Ridge, Lasso, Elastic Net), validation croisée, limitation de profondeur pour les arbres.


## Validation croisée

- Même principe que pour la classification : découpage du jeu de données en k folds.  
- Évaluation des erreurs (MSE, RMSE, MAE, R²) sur chaque fold.  
- Permet de choisir les hyperparamètres optimaux et détecter sur-apprentissage.


## Points clés à retenir

- La régression supervise l’apprentissage sur des **données continues**.  
- Le choix de la méthode dépend de :  
  - Type de relation (linéaire, non linéaire)  
  - Présence de multicolinéarité  
  - Besoin d’interprétation  
  - Taille du jeu de données  
- Toujours comparer plusieurs modèles et utiliser plusieurs métriques pour évaluer la généralisation.  
- La validation croisée est essentielle pour ajuster les hyperparamètres et détecter le sur-apprentissage.


## Régression Linéaire

### Introduction

La **régression linéaire** est une méthode d’apprentissage supervisé utilisée pour prédire une **variable continue** à partir de variables explicatives.  
Elle permet de comprendre comment les variables explicatives influencent la variable cible et d’effectuer des prédictions.

**Exemple**

- Variables explicatives : `"annee_construction"`, `"type_batiment"`, `"deperditions_murs"`, `"hauteur_sous_plafond"`, `"surface_habitable"`  
- Variable cible : `"besoin_chauffage"` (kWh)


### Configuration et méthodologie

1. **Sélection des variables explicatives et de la cible**  
   - Exemple : `X = ["annee_construction", "type_batiment", "deperditions_murs", "hauteur_sous_plafond", "surface_habitable"]`  
   - Cible : `Y = "besoin_chauffage"`

2. **Séparation train / test**  
   - L’ensemble **d’entraînement** sert à apprendre le modèle.  
   - L’ensemble **de test** sert à évaluer la généralisation.

3. **Prétraitement éventuel**  
   - Encodage des variables catégorielles (`type_batiment`)  
   - Normalisation ou standardisation si nécessaire (non obligatoire pour une régression linéaire simple)


### Modèle : régression linéaire

#### Création et entraînement du modèle

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X = df[["annee_construction", "type_batiment", "deperditions_murs", "hauteur_sous_plafond", "surface_habitable"]]
Y = df["besoin_chauffage"]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

reg_lin = LinearRegression()
reg_lin.fit(X_train, y_train)
```


#### Coefficients et intercept

```python
import pandas as pd

coeffs = pd.DataFrame(reg_lin.coef_, index=X_train.columns, columns=['Coefficient'])
intercept = reg_lin.intercept_

print("Coefficients :\n", coeffs)
print("Intercept :", intercept)
```

- Chaque coefficient indique l’effet d’une **variable explicative** sur la variable cible, toutes choses égales par ailleurs.  
- L’intercept représente la valeur prédite lorsque toutes les variables explicatives sont égales à zéro.


#### Prédictions sur le jeu de test

```python
y_pred = reg_lin.predict(X_test)
```


#### Évaluation du modèle

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MSE :", mse)
print("RMSE :", rmse)
print("MAE :", mae)
print("R² :", r2)
```

- **MSE (Mean Squared Error)** : moyenne des carrés des erreurs  
- **RMSE (Root Mean Squared Error)** : racine carrée du MSE, même unité que la cible  
- **MAE (Mean Absolute Error)** : moyenne des erreurs absolues  
- **R²** : proportion de variance expliquée par le modèle (1 = parfait, 0 = modèle nul)


#### Visualisation des valeurs observées vs prédites

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Valeurs observées")
plt.ylabel("Valeurs prédites")
plt.title("Régression linéaire : Observé vs Prévu")
plt.show()
```

- Les points proches de la diagonale rouge indiquent une bonne prédiction.  


#### Notes sur la régression multiple

- La **régression multiple** est une extension de la régression simple avec plusieurs variables explicatives.  
- Les coefficients doivent être interprétés **conditionnellement aux autres variables**.  
- Attention aux **corrélations fortes entre variables** (multicolinéarité), qui peuvent rendre les coefficients instables.


## Régression Polynomiale

### Introduction

La **régression polynomiale** est une extension de la régression linéaire qui permet de modéliser des relations **non linéaires** entre les variables explicatives et la variable cible.  
Elle consiste à transformer les variables explicatives en **puissances supérieures** (carré, cube, etc.) pour capturer des courbes et des interactions.

**Exemple  :**

- Variables explicatives : `"annee_construction"`, `"deperditions_murs"`, `"hauteur_sous_plafond"`, `"surface_habitable"`  
- Variable cible : `"besoin_chauffage"` (kWh)  
- Objectif : capturer des effets non linéaires (ex. surface² ou interactions entre variables)


### Transformation polynomiale des variables

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X = df[["annee_construction", "deperditions_murs", "hauteur_sous_plafond", "surface_habitable"]]
Y = df["besoin_chauffage"]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Transformation polynomiale (ex: degré 2)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)
```

- Chaque variable est transformée pour inclure ses **carrés et interactions** avec les autres variables.  
- `include_bias=False` supprime le terme constant, qui sera géré par l'intercept du modèle.


### Création et entraînement du modèle

```python
reg_poly = LinearRegression()
reg_poly.fit(X_train_poly, y_train)
```


### Coefficients et intercept

```python
coeffs = pd.DataFrame(reg_poly.coef_, index=poly.get_feature_names_out(X_train.columns), columns=['Coefficient'])
intercept = reg_poly.intercept_

print("Coefficients :\n", coeffs)
print("Intercept :", intercept)
```

- Les coefficients indiquent l’effet **non linéaire ou interaction** des variables transformées sur la cible.


### Prédictions sur le jeu de test

```python
y_pred = reg_poly.predict(X_test_poly)
```


### Évaluation du modèle

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MSE :", mse)
print("RMSE :", rmse)
print("MAE :", mae)
print("R² :", r2)
```

- **MSE, RMSE, MAE** et **R²** ont le même sens que pour la régression linéaire.  
- Un R² plus élevé qu’avec la régression linéaire simple indique que la polynomiale capture mieux les relations non linéaires.


### Visualisation des valeurs observées vs prédites

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Valeurs observées")
plt.ylabel("Valeurs prédites")
plt.title("Régression polynomiale : Observé vs Prévu")
plt.show()
```

- Les points proches de la diagonale rouge indiquent une bonne prédiction.  


### Notes importantes

- La **régression polynomiale** peut très rapidement conduire à du **sur-apprentissage** si le degré est trop élevé.  
- Utiliser une **validation croisée** pour sélectionner le degré optimal.  
- Les coefficients deviennent nombreux et moins interprétables, surtout pour les interactions de haut degré.


## Régression Ridge

### Introduction

La **régression Ridge** est une variante de la régression linéaire qui inclut une **pénalisation L2** pour réduire le risque de **sur-apprentissage** lorsque les variables explicatives sont nombreuses ou fortement corrélées.  
Elle minimise la somme des carrés des résidus et une pénalité proportionnelle au carré des coefficients.

**Exemple  :**

- Variables explicatives : `"annee_construction"`, `"deperditions_murs"`, `"hauteur_sous_plafond"`, `"surface_habitable"`  
- Variable cible : `"besoin_chauffage"` (kWh)  
- Objectif : limiter l’influence des coefficients extrêmes et améliorer la généralisation.


### Création du modèle Ridge

```python
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, train_test_split

X = df[["annee_construction", "deperditions_murs", "hauteur_sous_plafond", "surface_habitable"]]
Y = df["besoin_chauffage"]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

ridge = Ridge()
```


### Grid Search pour optimiser l’hyperparamètre alpha

- **alpha** : paramètre de régularisation L2.  
  - Petit alpha → approche proche de la régression linéaire classique  
  - Grand alpha → coefficients fortement contraints (réduction du sur-apprentissage)

```python
parameters = {'alpha': [0.01, 0.1, 1, 10, 100]}
grid_ridge = GridSearchCV(ridge, parameters, scoring='neg_mean_squared_error', cv=5)
grid_ridge.fit(X_train, y_train)

best_ridge = grid_ridge.best_estimator_
print("Meilleur alpha :", best_ridge.alpha)
```


### Prédictions sur le jeu de test

```python
y_pred = best_ridge.predict(X_test)
```


### Évaluation du modèle

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MSE :", mse)
print("RMSE :", rmse)
print("MAE :", mae)
print("R² :", r2)
```


### Coefficients et intercept

```python
coeffs = pd.DataFrame(best_ridge.coef_, index=X_train.columns, columns=['Coefficient'])
intercept = best_ridge.intercept_

print("Coefficients :\n", coeffs)
print("Intercept :", intercept)
```

- Les coefficients sont **réduits** par la régularisation L2 par rapport à une régression linéaire simple.  


### Visualisation des valeurs observées vs prédites

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Valeurs observées")
plt.ylabel("Valeurs prédites")
plt.title("Régression Ridge : Observé vs Prévu")
plt.show()
```

- Les points proches de la diagonale indiquent une bonne prédiction.


### Notes importantes

- Ridge est particulièrement utile lorsque les **variables explicatives sont corrélées**.  
- La **validation croisée** permet de choisir le meilleur alpha pour éviter le sur-apprentissage.  
- Contrairement à Lasso, Ridge **ne fait pas de sélection de variables**, il réduit seulement les coefficients.


## Régression Lasso

### Introduction

La **régression Lasso** est une variante de la régression linéaire qui inclut une **pénalisation L1**.  
Elle permet de **réduire certains coefficients à zéro**, ce qui fait également office de **sélection de variables** et limite le sur-apprentissage.

**Exemple  :**

- Variables explicatives : `"annee_construction"`, `"deperditions_murs"`, `"hauteur_sous_plafond"`, `"surface_habitable"`  
- Variable cible : `"besoin_chauffage"` (kWh)  
- Objectif : identifier les variables les plus pertinentes tout en améliorant la généralisation.


### Création du modèle Lasso

```python
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV, train_test_split

X = df[["annee_construction", "deperditions_murs", "hauteur_sous_plafond", "surface_habitable"]]
Y = df["besoin_chauffage"]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

lasso = Lasso()
```


### Grid Search pour optimiser l’hyperparamètre alpha

- **alpha** : paramètre de régularisation L1  
  - Petit alpha → approche proche de la régression linéaire classique  
  - Grand alpha → coefficients contraints, certains réduits à zéro

```python
parameters = {'alpha': [0.01, 0.1, 1, 10, 100]}
grid_lasso = GridSearchCV(lasso, parameters, scoring='neg_mean_squared_error', cv=5)
grid_lasso.fit(X_train, y_train)

best_lasso = grid_lasso.best_estimator_
print("Meilleur alpha :", best_lasso.alpha)
```


### Prédictions sur le jeu de test

```python
y_pred = best_lasso.predict(X_test)
```


### Évaluation du modèle

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MSE :", mse)
print("RMSE :", rmse)
print("MAE :", mae)
print("R² :", r2)
```


### Coefficients et intercept

```python
coeffs = pd.DataFrame(best_lasso.coef_, index=X_train.columns, columns=['Coefficient'])
intercept = best_lasso.intercept_

print("Coefficients :\n", coeffs)
print("Intercept :", intercept)
```

- Les coefficients **réduits à zéro** indiquent les variables moins pertinentes pour prédire le besoin de chauffage.  


### Visualisation des valeurs observées vs prédites

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Valeurs observées")
plt.ylabel("Valeurs prédites")
plt.title("Régression Lasso : Observé vs Prévu")
plt.show()
```

- Les points proches de la diagonale indiquent une bonne prédiction.


### Notes importantes

- Lasso est utile pour **sélectionner les variables les plus pertinentes** dans un jeu de données.  
- La **validation croisée** est importante pour choisir le meilleur alpha et éviter le sur-apprentissage.  
- Lasso peut réduire certains coefficients à zéro, contrairement à Ridge, qui réduit tous les coefficients mais ne les annule pas.


## Régression ElasticNet

### Introduction

La **régression ElasticNet** combine les pénalisations **L1 (Lasso)** et **L2 (Ridge)**.  
Elle permet à la fois :

- de **sélectionner des variables** (L1)  
- de **réduire le sur-apprentissage** (L2)  

**Exemple  :**

- Variables explicatives : `"annee_construction"`, `"deperditions_murs"`, `"hauteur_sous_plafond"`, `"surface_habitable"`  
- Variable cible : `"besoin_chauffage"` (kWh)  
- Objectif : bénéficier à la fois de la régularisation Lasso et Ridge.


### Création du modèle ElasticNet

```python
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV, train_test_split

X = df[["annee_construction", "deperditions_murs", "hauteur_sous_plafond", "surface_habitable"]]
Y = df["besoin_chauffage"]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

elastic = ElasticNet()
```


### Grid Search pour optimiser les hyperparamètres

- **alpha** : paramètre global de régularisation  
- **l1_ratio** : proportion de pénalisation L1 (0 = Ridge, 1 = Lasso)

```python
parameters = {'alpha': [0.01, 0.1, 1, 10],
              'l1_ratio': [0.1, 0.5, 0.7, 0.9, 1]}

grid_elastic = GridSearchCV(elastic, parameters, scoring='neg_mean_squared_error', cv=5)
grid_elastic.fit(X_train, y_train)

best_elastic = grid_elastic.best_estimator_
print("Meilleur alpha :", best_elastic.alpha)
print("Meilleur l1_ratio :", best_elastic.l1_ratio)
```


### Prédictions sur le jeu de test

```python
y_pred = best_elastic.predict(X_test)
```


### Évaluation du modèle

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MSE :", mse)
print("RMSE :", rmse)
print("MAE :", mae)
print("R² :", r2)
```


### Coefficients et intercept

```python
coeffs = pd.DataFrame(best_elastic.coef_, index=X_train.columns, columns=['Coefficient'])
intercept = best_elastic.intercept_

print("Coefficients :\n", coeffs)
print("Intercept :", intercept)
```

- Les coefficients proches de zéro indiquent des variables moins pertinentes.  
- La combinaison L1/L2 permet de conserver certaines variables tout en réduisant l’effet du bruit.


### Visualisation des valeurs observées vs prédites

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Valeurs observées")
plt.ylabel("Valeurs prédites")
plt.title("Régression ElasticNet : Observé vs Prévu")
plt.show()
```

- Les points proches de la diagonale montrent une bonne prédiction.  


### Notes importantes

- ElasticNet est particulièrement utile lorsque les variables explicatives sont **corrélées**.  
- La validation croisée permet d’ajuster **alpha** et **l1_ratio** pour optimiser la performance et éviter le sur-apprentissage.  
- ElasticNet combine les avantages de **Ridge** et **Lasso**, permettant à la fois régularisation et sélection de variables.


## Arbre de Régression — Projet DPE

### Introduction

Les **arbres de régression** sont une méthode d’apprentissage supervisé pour prédire une **variable continue**.  
Ils fonctionnent en divisant récursivement les données selon les variables explicatives afin de créer des **règles de prédiction** adaptées à chaque sous-groupe.

- Chaque nœud de l’arbre correspond à une **décision sur une variable** (ex. `surface_habitable > 80 m²`)  
- Chaque feuille fournit une **valeur prédite** (moyenne des observations dans ce sous-groupe)  
- Avantages : facile à interpréter, visualisable, capture des relations non linéaires et interactions complexes  
- Inconvénients : tendance au sur-apprentissage, sensible aux variations des données, instable (petites modifications du jeu de données peuvent fortement modifier l’arbre)

**Exemple  :**

- Variables explicatives : `"annee_construction"`, `"type_batiment"`, `"deperditions_murs"`, `"hauteur_sous_plafond"`, `"surface_habitable"`  
- Variable cible : `"besoin_chauffage"` (kWh)  
- Objectif : prédire la consommation énergétique des logements


### Création et entraînement du modèle

```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

X = df[["annee_construction", "deperditions_murs", "hauteur_sous_plafond", "surface_habitable"]]
Y = df["besoin_chauffage"]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Création de l'arbre
arbre_reg = DecisionTreeRegressor(max_depth=5, min_samples_split=50, min_samples_leaf=20, random_state=42)
arbre_reg.fit(X_train, y_train)
```

- **max_depth** : profondeur maximale de l’arbre, limite le sur-apprentissage  
- **min_samples_split** : nombre minimal d’observations pour diviser un nœud  
- **min_samples_leaf** : nombre minimal d’observations dans une feuille  


### Prédictions sur le jeu de test

```python
y_pred = arbre_reg.predict(X_test)
```


### Évaluation du modèle

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MSE :", mse)
print("RMSE :", rmse)
print("MAE :", mae)
print("R² :", r2)
```

- **MSE** : moyenne des carrés des erreurs  
- **RMSE** : racine carrée du MSE, même unité que la variable cible  
- **MAE** : moyenne des erreurs absolues, robuste aux outliers  
- **R²** : proportion de variance expliquée par le modèle


### Visualisation de l'arbre

```python
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(20,8))
plot_tree(arbre_reg, feature_names=X.columns, filled=True, fontsize=10)
plt.show()
```

- Chaque nœud indique la variable de division, le seuil et la valeur prédite pour cette feuille  
- Couleur : valeur de la prédiction dans le nœud


### Interprétation

- Les arbres de régression capturent facilement **les interactions non linéaires** entre variables (ex. `surface_habitable` et `deperditions_murs`)  
- Les feuilles contiennent la **valeur moyenne de la consommation** pour les logements correspondant à ces critères  
- La profondeur et le nombre de feuilles contrôlent **la complexité et la capacité à généraliser**  


### Points clés

- Arbre simple et interprétable, idéal pour visualiser la relation entre variables et cible  
- Attention au **sur-apprentissage** : mieux vaut limiter la profondeur ou utiliser un ensemble (Random Forest)  
- Possibilité de combiner plusieurs arbres pour améliorer la performance et la stabilité (Bagging / Random Forest)


## Random Forest Regressor

### Introduction

Le **Random Forest Regressor** est une extension des arbres de régression qui utilise la méthode de **bagging** (Bootstrap Aggregating) pour créer un ensemble d’arbres.  

- Chaque arbre est construit sur un **échantillon bootstrap** des données (tirage aléatoire avec remplacement)  
- Lors de chaque division, une **sous-partie aléatoire des variables explicatives** est testée pour choisir le meilleur split  
- La prédiction finale est la **moyenne des prédictions de tous les arbres**  

**Avantages** :  
- Réduit la variance par rapport à un arbre simple  
- Plus robuste et stable  
- Capable de capturer les relations non linéaires et interactions complexes  

**Inconvénients** :  
- Moins interprétable qu’un arbre simple  
- Plus coûteux en calculs  

**Exemple  :**  

- Variables explicatives : `"annee_construction"`, `"type_batiment"`, `"deperditions_murs"`, `"hauteur_sous_plafond"`, `"surface_habitable"`  
- Variable cible : `"besoin_chauffage"` (kWh)  


### Création et entraînement du modèle

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

X = df[["annee_construction", "deperditions_murs", "hauteur_sous_plafond", "surface_habitable"]]
Y = df["besoin_chauffage"]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Création de la forêt
rf_reg = RandomForestRegressor(n_estimators=100, max_depth=7, min_samples_split=20, min_samples_leaf=10, random_state=42)
rf_reg.fit(X_train, y_train)
```

- **n_estimators** : nombre d’arbres dans la forêt  
- **max_depth** : limite la profondeur maximale des arbres pour éviter le sur-apprentissage  
- **min_samples_split** : nombre minimum d’observations pour diviser un nœud  
- **min_samples_leaf** : nombre minimum d’observations dans une feuille  


### Prédictions sur le jeu de test

```python
y_pred = rf_reg.predict(X_test)
```


### Évaluation du modèle

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MSE :", mse)
print("RMSE :", rmse)
print("MAE :", mae)
print("R² :", r2)
```

- **MSE** : moyenne des carrés des erreurs  
- **RMSE** : racine carrée du MSE  
- **MAE** : moyenne des erreurs absolues  
- **R²** : proportion de variance expliquée  


### Importance des variables

```python
import pandas as pd

feature_importances = pd.DataFrame({
    "Variable": X.columns,
    "Importance": rf_reg.feature_importances_
}).sort_values(by="Importance", ascending=False)

feature_importances
```

- Montre l’influence relative de chaque variable sur les prédictions  
- Plus la valeur est élevée, plus la variable contribue à réduire l’erreur  


### Interprétation

- Random Forest combine **la puissance des arbres multiples** avec de l’aléatoire pour obtenir des prédictions robustes  
- Chaque arbre “voit” seulement une partie des variables à chaque split, ce qui réduit la corrélation entre arbres et améliore la généralisation  
- Les variables les plus importantes peuvent être utilisées pour **feature selection** ou interprétation du modèle  


### Points clés

- Plus stable et performant qu’un arbre unique  
- Permet de gérer des relations non linéaires et interactions complexes  
- Importance des variables aide à identifier les facteurs clés  
- Peut être combiné avec Grid Search pour optimiser les hyperparamètres (n_estimators, max_depth, min_samples_leaf)


## Support Vector Regressor (SVR)

### Introduction

Le **Support Vector Regressor (SVR)** est l’adaptation de la méthode SVM pour les problèmes de régression.  
L’idée principale est de trouver une **fonction qui s’ajuste aux données** tout en **maximisant la marge d’erreur tolérable (epsilon)**.

- L’objectif est de minimiser une fonction de perte ε-insensitive : seules les erreurs supérieures à ε sont pénalisées.  
- Le SVR peut utiliser des **kernels** pour modéliser des relations **non linéaires** (linéaire, polynomial, RBF).  
- Les points de données situés **en dehors de la marge** (support vectors) déterminent la position de l’hyperplan.  

**Avantages** : efficace pour petites et moyennes bases, capture les non-linéarités, robuste aux outliers dans la zone de tolérance  
**Inconvénients** : paramétrage délicat (C, epsilon, kernel, gamma), difficilement interprétable  

**Exemple  :**  
- Variables explicatives : `"annee_construction"`, `"type_batiment"`, `"deperditions_murs"`, `"hauteur_sous_plafond"`, `"surface_habitable"`  
- Variable cible : `"besoin_chauffage"`  


### Configuration et Grid Search

```python
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, train_test_split

X = df[["annee_construction", "deperditions_murs", "hauteur_sous_plafond", "surface_habitable"]]
Y = df["besoin_chauffage"]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Définition du modèle
svr = SVR()

# Grille d'hyperparamètres
param_grid = {
    "kernel": ["linear", "poly", "rbf"],
    "C": [0.1, 1, 10],
    "epsilon": [0.01, 0.1, 0.5],
    "gamma": ["scale", "auto"]
}

grid_search = GridSearchCV(estimator=svr,
                           param_grid=param_grid,
                           scoring="neg_mean_squared_error",
                           cv=5,
                           verbose=2,
                           n_jobs=-1)

grid_search.fit(X_train, y_train)
```

- **kernel** : type de fonction pour transformer les données (linéaire, polynomial, RBF)  
- **C** : paramètre de régularisation, contrôle le compromis biais/variance  
- **epsilon** : tolérance pour la zone de non-pénalisation  
- **gamma** : influence d’un point de donnée sur la fonction de décision (pour RBF)  
- **scoring** : négatif MSE car GridSearch maximise la métrique  


### Meilleur modèle et hyperparamètres

```python
best_svr = grid_search.best_estimator_
print("Meilleur modèle :", best_svr)
print("Meilleurs paramètres :", grid_search.best_params_)
```


### Prédictions et évaluation

```python
y_pred = best_svr.predict(X_test)

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MSE :", mse)
print("RMSE :", rmse)
print("MAE :", mae)
print("R² :", r2)
```


### Interprétation

- Les **support vectors** sont les observations qui définissent la fonction prédictive.  
- Le paramètre **C** équilibre la précision sur l’entraînement et la capacité de généralisation.  
- Le paramètre **epsilon** définit la marge de tolérance, utile pour ignorer le bruit.  
- Les **kernels** permettent de capturer des relations non linéaires entre les variables explicatives et la cible.  


### Points clés

- SVR est particulièrement efficace pour les **relations complexes et non linéaires**.  
- Grid Search avec validation croisée permet de trouver les paramètres optimaux et de détecter le sur-apprentissage.  
- Comparé aux Random Forest et Gradient Boosting, SVR est plus difficilement interprétable mais peut être performant sur de petites bases et des relations subtiles.  
- Pour le projet DPE, SVR peut prédire les besoins de chauffage à partir des caractéristiques du logement, en tenant compte des non-linéarités.


## XGBoost Regressor

### Introduction

Le **XGBoost Regressor** est une implémentation optimisée du **Gradient Boosting** pour les problèmes de régression.  
L’idée principale est de construire une **série d’arbres faibles (weak learners)**, chaque arbre corrigeant les erreurs du modèle précédent.

- Chaque nouvel arbre est entraîné sur le **résidu** du modèle précédent, minimisant ainsi la fonction de perte (souvent MSE pour la régression).  
- Les arbres sont ajoutés **séquentiellement**, avec un poids d’apprentissage (`learning_rate`) qui contrôle leur contribution.  
- XGBoost inclut des **options de régularisation** (`alpha`, `lambda`) pour réduire le surapprentissage.

**Avantages** : très performant, gère les relations non linéaires et interactions complexes, robuste aux outliers  
**Inconvénients** : paramétrage complexe (nombre d’arbres, profondeur, learning rate), moins interprétable qu’un arbre unique  

**Exemple  :**  
- Variables explicatives : `"annee_construction"`, `"type_batiment"`, `"deperditions_murs"`, `"hauteur_sous_plafond"`, `"surface_habitable"`  
- Variable cible : `"besoin_chauffage"`  


### Configuration et Grid Search

```python
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, train_test_split

X = df[["annee_construction", "deperditions_murs", "hauteur_sous_plafond", "surface_habitable"]]
Y = df["besoin_chauffage"]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

xgb = XGBRegressor(objective="reg:squarederror", random_state=42)

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.2],
    "subsample": [0.7, 1],
    "colsample_bytree": [0.7, 1]
}

grid_search = GridSearchCV(estimator=xgb,
                           param_grid=param_grid,
                           scoring="neg_mean_squared_error",
                           cv=5,
                           verbose=2,
                           n_jobs=-1)

grid_search.fit(X_train, y_train)
```

- **n_estimators** : nombre d’arbres à construire  
- **max_depth** : profondeur maximale de chaque arbre  
- **learning_rate** : poids de chaque arbre dans la combinaison finale  
- **subsample** : proportion d’observations utilisées pour chaque arbre (bagging partiel)  
- **colsample_bytree** : proportion de variables utilisées pour chaque arbre  
- **scoring** : négatif MSE car GridSearch maximise la métrique  


### Meilleur modèle et hyperparamètres

```python
best_xgb = grid_search.best_estimator_
print("Meilleur modèle :", best_xgb)
print("Meilleurs paramètres :", grid_search.best_params_)
```


### Prédictions et évaluation

```python
y_pred = best_xgb.predict(X_test)

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MSE :", mse)
print("RMSE :", rmse)
print("MAE :", mae)
print("R² :", r2)
```


### Interprétation

- Chaque arbre est construit pour corriger les erreurs résiduelles du modèle précédent.  
- Les variables les plus importantes pour la prédiction peuvent être extraites avec `best_xgb.feature_importances_`.  
- Le **learning_rate** et le **nombre d’arbres** doivent être choisis avec soin pour éviter le surapprentissage.  
- Les paramètres de sous-échantillonnage (`subsample`, `colsample_bytree`) réduisent la variance et améliorent la généralisation.


### Points clés

- XGBoost Regressor est très performant pour les problèmes de régression complexes, capturant à la fois **non-linéarités** et **interactions** entre variables.  
- Grid Search avec validation croisée permet de sélectionner les hyperparamètres optimaux.  
- Pour le projet DPE, XGBoost peut prédire les besoins de chauffage en combinant efficacement les caractéristiques des logements et en corrigeant les erreurs progressivement.

## En bref


## Récapitulatif

| Méthode | Principe | Avantages | Inconvénients | Type de relation | Interprétabilité | Sensibilité aux données |
|---------|----------|-----------|---------------|----------------|----------------|-----------------------|
| Régression linéaire simple | Ajuste une droite pour prédire une variable continue à partir d’une variable explicative | Simple, rapide, facile à interpréter | Limité aux relations linéaires, sensible aux outliers | Linéaire | Très bonne | Sensible aux outliers |
| Régression linéaire multiple | Ajuste un modèle linéaire avec plusieurs variables explicatives | Permet d’intégrer plusieurs variables, interprétation des coefficients | Multicolinéarité possible, linéarité supposée | Linéaire | Bonne | Sensible aux outliers et multicolinéarité |
| Régression polynomiale | Ajoute des puissances des variables pour modéliser des relations non linéaires | Capte des relations non linéaires simples | Risque de surapprentissage si degré élevé, difficile à interpréter | Non linéaire | Moyenne | Sensible aux valeurs extrêmes |
| Ridge | Régression linéaire avec pénalisation L2 | Réduit la variance, limite le surapprentissage | Ne fait pas de sélection de variables | Linéaire | Bonne | Moins sensible aux multicolinéarités |
| Lasso | Régression linéaire avec pénalisation L1 | Peut sélectionner automatiquement les variables importantes | Risque de sur-regularisation, biais | Linéaire | Moyenne | Sensible aux corrélations fortes |
| ElasticNet | Combinaison L1 + L2 | Compromis entre Ridge et Lasso, sélection + régularisation | Paramétrage complexe (alpha, l1_ratio) | Linéaire | Moyenne | Sensible aux corrélations fortes |
| Arbre de régression | Divise récursivement les données pour prédire une valeur | Non linéaire, capture interactions, facile à visualiser | Surapprentissage, sensible aux variations | Non linéaire | Bonne (visualisation possible) | Peu sensible aux valeurs extrêmes, mais sensible aux petits changements |
| Random Forest Regressor | Ensemble d’arbres pour réduire la variance | Robuste, performant, importance des variables | Moins interprétable, plus lent | Non linéaire | Moyenne | Robuste aux outliers |
| SVM Regressor | Maximiser la marge avec tolérance à une certaine erreur | Efficace pour non linéaire via kernels | Paramétrage complexe (C, epsilon, kernel), peu interprétable | Linéaire/non linéaire (kernel) | Faible | Sensible au choix du kernel et des paramètres |
| XGBoost Regressor | Gradient boosting avec arbres séquentiels | Très performant, capture non linéarités et interactions, régularisation | Paramétrage complexe, moins interprétable | Non linéaire | Moyenne | Robuste aux valeurs aberrantes si bien paramétré |


# Exercice : Prédiction de la consommation énergétique

Dans cet exercice, vous allez appliquer toutes les méthodes de régression vues précédemment pour prédire la **consommation énergétique ou le besoin en énergie** des logements à partir d’un jeu de données étendu. L’objectif est de comparer les performances des modèles et de déterminer les variables les plus importantes, tout en faisant attention au **data leakage**.
