# Chapitre 13 - MLops

- [Chapitre 13 - MLops](#chapitre-13---mlops)
- [Introduction](#introduction)
- [Les Pipelines](#les-pipelines)
  - [Pourquoi utiliser un Pipeline ?](#pourquoi-utiliser-un-pipeline-)
    - [Avantages :](#avantages-)
  - [Exemple simple : normalisation + régression logistique](#exemple-simple--normalisation--régression-logistique)
  - [Pipelines avec variables numériques + catégorielles (ColumnTransformer)](#pipelines-avec-variables-numériques--catégorielles-columntransformer)
  - [Pipeline + GridSearchCV](#pipeline--gridsearchcv)
  - [Pipeline pour la régression](#pipeline-pour-la-régression)
  - [Pipelines avec arbres / forêts](#pipelines-avec-arbres--forêts)
  - [Pipeline + sauvegarde (pickle / joblib)](#pipeline--sauvegarde-pickle--joblib)
  - [Résumé des points forts](#résumé-des-points-forts)
  - [Quand utiliser un Pipeline ?](#quand-utiliser-un-pipeline-)
  - [Exemple complet (cas DPE)](#exemple-complet-cas-dpe)
  - [Conclusion générale](#conclusion-générale)
- [Introduction à **MLflow**](#introduction-à-mlflow)
  - [Qu’est-ce que MLflow ?](#quest-ce-que-mlflow-)
  - [Pourquoi utiliser MLflow ?](#pourquoi-utiliser-mlflow-)
  - [Architecture des modules MLflow](#architecture-des-modules-mlflow)
    - [MLflow Tracking](#mlflow-tracking)
    - [MLflow Projects](#mlflow-projects)
    - [MLflow Models](#mlflow-models)
    - [Model Registry](#model-registry)
  - [Exemple complet avec Random Forest](#exemple-complet-avec-random-forest)
    - [Suivi d'une expérience](#suivi-dune-expérience)
    - [Visualisation avec l’UI](#visualisation-avec-lui)
    - [Enregistrement dans le Model Registry](#enregistrement-dans-le-model-registry)
    - [Chargement d’un modèle depuis le Registry](#chargement-dun-modèle-depuis-le-registry)
  - [Avantages de MLflow](#avantages-de-mlflow)
  - [Conclusion](#conclusion)


# Introduction 

MLOps est une culture et une pratique du ML qui unifient le développement d'applications de ML (Dev) avec le déploiement et les opérations (Ops) du système de ML

<p align="center">
  <img src="https://dlab.berkeley.edu/sites/default/files/styles/openberkeley_image_full/public/mlops-2022-05-10.png?itok=UMY4fgfd&timestamp=1652227637" alt="Source de l'image" width="600"/>
</p>

# Les Pipelines

## Pourquoi utiliser un Pipeline ?

En machine learning, un workflow complet comporte plusieurs étapes :

- nettoyage des données  
- transformation des variables  
- normalisation / standardisation  
- encodage des variables catégorielles  
- sélection ou création de features  
- entraînement d’un modèle  
- prédiction  

**Problème :**  
--> Faire ces étapes séparément augmente les risques d'erreur, de *data leakage*, et le code devient difficile à maintenir.

**Problème :**  : le Pipeline

Un **Pipeline** permet de chaîner toutes les étapes de traitement en une seule structure cohérente.

### Avantages :

1) Évite le *data leakage*  
Les transformateurs (scaler, PCA, imputations…) sont **ajustés uniquement sur le train**.  
Avec un Pipeline, scikit-learn garantit automatiquement ce comportement.

2) Code plus propre  
Une seule ligne pour entraîner, une seule pour prédire.

3) Reproductibilité & traçabilité  
Le Pipeline devient un objet complet qui contient toutes les étapes.

4) Compatible **GridSearchCV**  
Permet d’optimiser les hyperparamètres de n’importe quelle étape du pipeline.

5) Industrialisation facile  
Un seul objet exporté = tout le workflow ML.


##Structure d’un Pipeline

```python
Pipeline([
    ('nom_etape1', objet_sklearn),
    ('nom_etape2', objet_sklearn),
    ...
    ('model', modele_ml)
])
```

- Toutes les étapes sauf la dernière doivent implémenter : `.fit()` et `.transform()`
- La dernière étape implémente : `.fit()` et `.predict()`


## Exemple simple : normalisation + régression logistique

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('log_reg', LogisticRegression())
])

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
```

**Résultat :**

- `StandardScaler` est ajusté sur `X_train` uniquement  
- les données sont transformées  
- le modèle apprend sur les données transformées  
- `.predict()` applique automatiquement tout le workflow  


## Pipelines avec variables numériques + catégorielles (ColumnTransformer)

Pour gérer plusieurs types de variables, on combine :

- **ColumnTransformer** → traitements adaptés par type
- **Pipeline** → assemblage complet

Exemple :

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

preprocessing = ColumnTransformer([
    ('num', StandardScaler(), ['surface_habitable', 'annee_construction']),
    ('cat', OneHotEncoder(handle_unknown='ignore'), ['type_batiment'])
])

pipe = Pipeline([
    ('prep', preprocessing),
    ('model', RandomForestClassifier())
])

pipe.fit(X_train, y_train)
```


## Pipeline + GridSearchCV

Permet d’optimiser simultanément les hyperparamètres **du modèle et des transformations**.

Exemple :

```python
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier())
])

params = {
    'knn__n_neighbors': [3, 5, 7], 
    'knn__weights': ['uniform', 'distance']
}

grid = GridSearchCV(pipe, params, cv=5, scoring='f1_macro')
grid.fit(X_train, y_train)

print(grid.best_params_)
print(grid.best_score_)
```

**Syntaxe à retenir :**  
--> `nom_etape__nom_parametre`


## Pipeline pour la régression

Exemple : Features polynomiaux + régression linéaire.

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ('poly', PolynomialFeatures(degree=3)),
    ('reg', LinearRegression())
])

pipe.fit(X_train, y_train)
```


 ## Pipelines avec arbres / forêts

Les modèles d’arbres ne nécessitent pas de normalisation.

```python
pipe = Pipeline([
    ('prep', preprocessing),
    ('model', RandomForestRegressor())
])
```


## Pipeline + sauvegarde (pickle / joblib)

Indispensable pour production :

```python
import joblib
joblib.dump(pipe, "modele_complet.pkl")

# chargement
pipe_loaded = joblib.load("modele_complet.pkl")
pipe_loaded.predict(X_new)
```


## Résumé des points forts

| Avantage | Pourquoi c’est utile |
|---------|------------------------|
| Anti data-leakage | Le scaler est appris uniquement sur le train |
| Simplicité | Une seule ligne pour entraîner & prédire |
| Industrialisation | On exporte tout le workflow ML |
| Robustesse | Compatible GridSearchCV |
| Clarté | Code structuré & lisible |
| Sécurité | Les étapes sont appliquées dans le bon ordre |


## Quand utiliser un Pipeline ?

**Presque toujours.**

Cas typiques :

- normalisation obligatoire (KNN, SVM, logistic regression)
- encodage des variables catégorielles
- feature engineering complexe
- optimisation d'hyperparamètres
- production / APIs


## Exemple complet (cas DPE)

Pipeline complet : imputation + normalisation + encodage + modèle.

```python
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor

num_features = ['surface_habitable', 'annee_construction', 'hauteur_sous_plafond']
cat_features = ['type_batiment']

numeric_transform = Pipeline([
    ('impute', SimpleImputer(strategy='median')),
    ('scale', StandardScaler())
])

categorical_transform = Pipeline([
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('encode', OneHotEncoder(handle_unknown='ignore'))
])

preprocess = ColumnTransformer([
    ('num', numeric_transform, num_features),
    ('cat', categorical_transform, cat_features)
])

pipe = Pipeline([
    ('prep', preprocess),
    ('model', GradientBoostingRegressor())
])

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
```


## Conclusion générale

Le module **Pipeline** de scikit-learn est indispensable en machine learning :

- structure le workflow ML  
- réduit les erreurs  
- évite le data leakage  
- simplifie le code  
- permet l’optimisation hyperparamétrique  
- facilite l'industrialisation  

--> **Bonne pratique incontournable**, même pour des modèles simples.

# Introduction à **MLflow**

<p align="center">
  <img src="https://mlflow.org/docs/latest/api_reference/_static/MLflow-logo-final-black.png" alt="Source de l'image" width="600"/>
</p>


## Qu’est-ce que MLflow ?

MLflow est une plateforme open-source conçue pour **gérer l’ensemble du cycle de vie d’un projet de Machine Learning**.  
Il facilite le suivi, la reproductibilité, le déploiement et la gestion de modèles, quels que soient :

- la technologie utilisée (Scikit-Learn, XGBoost, TensorFlow…)
- l’environnement (local, cloud)
- les outils de développement (Python, R, Java…)

MLflow n’est **pas** un framework de ML.  
C’est un **outil d’orchestration**, destiné à organiser un projet ML de A à Z.

MLflow comporte 4 modules principaux :

1. **MLflow Tracking** – suivre les expériences et leurs résultats  
2. **MLflow Projects** – standardiser l’exécution d’un projet  
3. **MLflow Models** – gérer et déployer des modèles  
4. **MLflow Registry** – versionner les modèles (comme Git mais pour les modèles)


## Pourquoi utiliser MLflow ?

Dans un vrai projet, on teste souvent :

- plusieurs modèles  
- plusieurs jeux d’hyperparamètres  
- plusieurs pipelines  
- différents prétraitements  

Sans outils, c'est vite le chaos.

MLflow permet :

| Besoin | Solution MLflow |
|--------|----------------|
| Comparer facilement des modèles | Tracking automatique des métriques |
| Sauvegarder les paramètres | Logging des hyperparamètres |
| Suivre les versions de modèles | Registry |
| Reproduire une expérience | Artifacts + Projects |
| Déployer un modèle | MLflow Models |


## Architecture des modules MLflow

### MLflow Tracking

Permet d’enregistrer automatiquement :

- les **paramètres** (`n_estimators`, `max_depth`, etc.)  
- les **métriques** (`rmse`, `accuracy`, etc.)  
- les **artifacts** (graphiques, fichiers, modèles pickle)  
- les **modèles** eux-mêmes

Visualisation dans une interface web avec :

```bash
mlflow ui
```

Accessible sur http://127.0.0.1:5000


### MLflow Projects

Permet de structurer un projet ML de manière reproductible :

- spécifier les dépendances (Conda, pip)  
- définir un script principal  
- relancer le training dans n’importe quel environnement  


### MLflow Models

Format standardisé pour enregistrer et recharger des modèles ML.  
Fonctionne avec :

- scikit-learn
- xgboost
- pytorch
- tensorflow
- lightgbm

Exemple : `mlflow.sklearn.log_model(model, "model")`


### Model Registry

Un espace centralisé pour gérer les modèles :

- stage : `None`, `Staging`, `Production`, `Archived`  
- versioning automatique  
- commentaires et validations  
- promotion/déploiement  

C'est l'équivalent d’un **Git pour modèles ML**.


## Exemple complet avec Random Forest

### Suivi d'une expérience

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt

# Modèle Random Forest
model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)

with mlflow.start_run():

    # --- Entrainement ---
    model.fit(X_train, y_train)

    # --- Prédictions ---
    y_pred = model.predict(X_test)

    # --- Calcul des métriques ---
    rmse = sqrt(mean_squared_error(y_test, y_pred))
    mlflow.log_metric("rmse", rmse)

    # --- Paramètres ---
    mlflow.log_params({
        "n_estimators": 200,
        "max_depth": 10
    })

    # --- Sauvegarde du modèle ---
    mlflow.sklearn.log_model(model, "model")
```


### Visualisation avec l’UI

```bash
mlflow ui
```

Dans l’interface :

- Expériences et runs  
- Paramètres et métriques  
- Artifacts (graphiques, fichiers)  
- Modèles sauvegardés  


### Enregistrement dans le Model Registry

```python
mlflow.register_model(
    model_uri="runs:/<run_id>/model",
    name="random_forest_dpe"
)
```


### Chargement d’un modèle depuis le Registry

```python
model_loaded = mlflow.pyfunc.load_model(
    model_uri="models:/random_forest_dpe/Production"
)

# Prédiction sur de nouvelles données
pred = model_loaded.predict(X_new)
```


## Avantages de MLflow

- **Standardisation** : même structure pour tous les projets  
- **Traçabilité** : version des modèles, paramètres, code et librairies  
- **Reproductibilité** : éviter les écarts entre environnements  
- **Collaboration** : partager et comparer les modèles entre équipes  
- **Déploiement facile** : servir un modèle avec `mlflow models serve -m model_path`


## Conclusion

MLflow est un outil indispensable pour :

- organiser et suivre vos expériences ML  
- versionner et gérer les modèles  
- déployer facilement vos modèles en production  

Avec l’exemple du **Random Forest pour prédire la consommation énergétique ou le DPE**, MLflow permet :

- de suivre vos hyperparamètres  
- d’évaluer vos métriques sur train/test  
- de sauvegarder et recharger vos modèles  
- de collaborer avec votre équipe de manière efficace
