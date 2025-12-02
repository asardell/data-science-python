# Chapitre 12 : Inteprétabilités des modèles complexes

- [Chapitre 12 : Inteprétabilités des modèles complexes](#chapitre-12--inteprétabilités-des-modèles-complexes)
- [Méthode SHAP (SHapley Additive exPlanations)](#méthode-shap-shapley-additive-explanations)
  - [Pourquoi a-t-on besoin de SHAP ?](#pourquoi-a-t-on-besoin-de-shap-)
  - [Qu’est-ce que SHAP ?](#quest-ce-que-shap-)
    - [Idée intuitive](#idée-intuitive)
  - [Pourquoi SHAP est essentiel ?](#pourquoi-shap-est-essentiel-)
    - [Avantages](#avantages)
    - [Limites](#limites)
  - [Types d’explications SHAP](#types-dexplications-shap)
    - [Importance globale des variables](#importance-globale-des-variables)
    - [Graphique de dépendance](#graphique-de-dépendance)
    - [Explication individuelle (force plot)](#explication-individuelle-force-plot)
  - [Exemple intuitif appliqué au DPE](#exemple-intuitif-appliqué-au-dpe)
  - [Code Python pour utiliser SHAP](#code-python-pour-utiliser-shap)
    - [Installation (si besoin)](#installation-si-besoin)
    - [Exemple complet avec XGBoost Regressor (DPE consommation)](#exemple-complet-avec-xgboost-regressor-dpe-consommation)
    - [Importance des variables (global)](#importance-des-variables-global)
    - [Graphique de dépendance](#graphique-de-dépendance-1)
    - [Explication d’une prédiction spécifique](#explication-dune-prédiction-spécifique)
    - [Force plot (explication locale)](#force-plot-explication-locale)
  - [Interprétation des résultats SHAP](#interprétation-des-résultats-shap)
    - [Importance globale](#importance-globale)
    - [Interprétation locale](#interprétation-locale)
  - [À retenir](#à-retenir)
- [Méthode LIME : (Local Interpretable Model-Agnostic Explanations)](#méthode-lime--local-interpretable-model-agnostic-explanations)
  - [Idée principale](#idée-principale)
  - [Pourquoi utiliser LIME ?](#pourquoi-utiliser-lime-)
    - [Avantages](#avantages-1)
    - [Limites](#limites-1)
  - [Exemple intuitif appliqué au DPE](#exemple-intuitif-appliqué-au-dpe-1)
  - [Exemple complet : LIME pour une régression DPE](#exemple-complet--lime-pour-une-régression-dpe)
    - [Entraîner un modèle complexe (XGBoost ici)](#entraîner-un-modèle-complexe-xgboost-ici)
    - [Utiliser LIME](#utiliser-lime)
  - [LIME pour la classification (étiquettes DPE)](#lime-pour-la-classification-étiquettes-dpe)
  - [Comment interpréter LIME ?](#comment-interpréter-lime-)
  - [LIME vs SHAP : quand utiliser quoi ?](#lime-vs-shap--quand-utiliser-quoi-)
  - [Conclusion](#conclusion)

# Méthode SHAP (SHapley Additive exPlanations) 

## Pourquoi a-t-on besoin de SHAP ?
Les modèles modernes très performants comme :

- Random Forest  
- Gradient Boosting (XGBoost, LightGBM, CatBoost)  
- SVM  
- Réseaux de neurones  

sont souvent des **boîtes noires** :  
difficiles à interpréter, impossibles à visualiser simplement.

Pour le DPE, comprendre un modèle est essentiel pour :

- expliquer pourquoi un logement a telle consommation,  
- savoir quelles caractéristiques influencent le plus le résultat,  
- proposer des rénovations pertinentes,  
- gagner la confiance des utilisateurs.

**SHAP permet d’ouvrir la boîte noire.**


## Qu’est-ce que SHAP ?
SHAP utilise des **valeurs de Shapley**, issues de la théorie des jeux.

### Idée intuitive
Chaque variable (surface, isolation, année…) est un **joueur** qui contribue à la prédiction finale.

SHAP calcule combien chaque variable :

- augmente la prédiction  
- diminue la prédiction  
- influence différemment d’un logement à l’autre  

Les contributions s’additionnent pour reconstruire la prédiction.


## Pourquoi SHAP est essentiel ?
### Avantages
- Fonctionne avec presque tous les modèles  
- Interprétation locale (par observation)  
- Interprétation globale (vue d’ensemble du modèle)  
- Théorie mathématique solide  
- Recommandé dans l'industrie pour la transparence  

### Limites
- Peut être lent  
- Complexe à expliquer à des débutants  
- Nécessite des visualisations  


## Types d’explications SHAP

<p align="center">
  <img src="https://www.researchgate.net/publication/349883007/figure/fig3/AS:11431281348131931@1743701923529/Feature-importance-based-on-SHAP-values-On-the-left-side-the-mean-absolute-SHAP-values.tif" alt="Source de l'image" width="600"/>
</p>

### Importance globale des variables
Montre quelles caractéristiques influencent le plus le modèle.

### Graphique de dépendance
Montre comment une variable influence la prédiction.

### Explication individuelle (force plot)
Explique une prédiction précise :

- quelles variables augmentent la consommation  
- quelles variables la diminuent  

Très utile en DPE pour proposer des actions.

<p align="center">
  <img src="https://www.aidancooper.co.uk/content/images/2021/11/force_plots.png" alt="Source de l'image" width="600"/>
</p>


## Exemple intuitif appliqué au DPE
Supposons une prédiction :

- Prédiction : 250 kWh/m²/an  
- Moyenne du dataset : 180 kWh/m²/an  

Les valeurs SHAP pourraient être :

| Variable | SHAP | Interprétation |
|---------|------|----------------|
| Déperditions murs | +40 | Mauvaise isolation |
| Surface habitable | -20 | Moins de déperdition par m² |
| Année construction | +30 | Ancien logement |
| Hauteur plafond | +10 | Plus de volume |
| Type bâtiment | +10 | Bâtiment énergivore |

Total :  
180 + (40 - 20 + 30 + 10 + 10) = **250**


## Code Python pour utiliser SHAP

###  Installation (si besoin)
```python
pip install shap
pip install xgboost
```


### Exemple complet avec XGBoost Regressor (DPE consommation)

```python
import shap
import xgboost as xgb
import matplotlib.pyplot as plt

# Entraînement du modèle (exemple)
model = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)

# --- SHAP explainer ---
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_train)
```


### Importance des variables (global)
```python
shap.summary_plot(shap_values, X_train)
```


### Graphique de dépendance
```python
shap.dependence_plot("surface_habitable", shap_values.values, X_train)
```


### Explication d’une prédiction spécifique  
Exemple : premier logement

```python
i = 0
shap.plots.waterfall(shap_values[i])
```


### Force plot (explication locale)
```python
shap.plots.force(shap_values[i])
```


## Interprétation des résultats SHAP

### Importance globale
Permet de voir quelles variables influencent le plus le modèle.
Exemple typique pour le DPE :

1. déperditions murs  
2. isolation toiture  
3. année construction  
4. type de bâtiment  
5. surface habitable  

### Interprétation locale
Pour un logement :

- si les valeurs SHAP sont positives → augmentent la consommation  
- si elles sont négatives → améliorent la performance énergétique  


## À retenir
SHAP permet de rendre les modèles complexes :

- transparents  
- explicables  
- auditables  
- fiables  

Pour le DPE, il devient enfin possible d’expliquer :

- pourquoi un logement est mal classé  
- quelles caractéristiques sont responsables  
- quelles améliorations auront le plus d’impact  

C’est une compétence essentielle dans tout projet utilisant des modèles avancés.



# Méthode LIME : (Local Interpretable Model-Agnostic Explanations) 

## Idée principale
Même si un modèle est complexe globalement, **autour d’une observation spécifique**, il peut être approximé par un modèle simple :

- un modèle linéaire (pour expliquer une prédiction)  
- ou un arbre de décision court  

LIME :

1. Prend une observation à expliquer  
2. Génère des points similaires autour d’elle  
3. Observe comment le modèle prédit ces variations  
4. Entraîne un modèle simple (interprétable) sur ces points  
5. Utilise ce modèle simple comme explication locale  

LIME n’explique **pas** le modèle entier, seulement **une prédiction à la fois**.


## Pourquoi utiliser LIME ?
### Avantages
- Très simple à comprendre  
- Agnostique : fonctionne avec **n’importe quel modèle**  
- Explication locale très claire  
- Fonctionne pour la régression et la classification  
- Rapide  

### Limites
- Explication uniquement locale  
- Pas de vision globale du modèle  
- Reproductibilité limitée (car explainer aléatoire)  
- Méthode approximative  


## Exemple intuitif appliqué au DPE
Supposons qu’un modèle prédit une consommation :

**250 kWh/m²/an**

LIME va :

1. Faire varier légèrement les caractéristiques (surface, année, isolation…)  
2. Observer comment la prédiction change  
3. Ajuster un petit modèle linéaire  
4. Dire quelles variables influencent cette prédiction spécifique

Exemple de sortie LIME :

| Variable | Influence LIME | Interprétation |
|---------|----------------|----------------|
| Déperditions murs = élevées | +45 | augmente la prédiction |
| Année construction < 1970 | +30 | ancien logement |
| Surface habitable grande | -18 | réduit besoin par m² |
| Isolation toiture bonne | -12 | réduit les pertes |


## Exemple complet : LIME pour une régression DPE

### Entraîner un modèle complexe (XGBoost ici)

```python
import xgboost as xgb

model = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)
```


### Utiliser LIME

```python
from lime.lime_tabular import LimeTabularExplainer
import numpy as np

# Création de l'explainer
explainer = LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X_train.columns,
    mode='regression'
)

# Observation à expliquer
i = 0
observation = X_test.iloc[i].values

# Explication locale
exp = explainer.explain_instance(
    data_row=observation,
    predict_fn=model.predict
)

exp.show_in_notebook()
```

Pour exporter en console :

```python
print(exp.as_list())
```


## LIME pour la classification (étiquettes DPE)

```python
explainer = LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X_train.columns,
    class_names=['A','B','C','D','E','F','G'],
    mode='classification'
)

i = 0
exp = explainer.explain_instance(
    X_test.iloc[i].values,
    model.predict_proba
)

exp.show_in_notebook()
```


## Comment interpréter LIME ?
LIME renvoie typiquement une liste de règles du type :

```
déperditions_murs > 120   |  +42.5
année_construction < 1975 |  +28.1
surface_habitable > 90    |  -14.2
```

Interprétation :

- Valeurs **positives** → augmentent la consommation prédite  
- Valeurs **négatives** → diminuent la consommation prédite  

LIME donne enfin une explication simple pour **une prédiction précise**.


## LIME vs SHAP : quand utiliser quoi ?

| Critère | SHAP | LIME |
|--------|------|------|
| Nature | théorique, solide | approximatif |
| Analyse locale | excellent | excellent |
| Analyse globale | très bon | limitée |
| Interprétation | parfois complexe | très simple |
| Vitesse | plus lent | rapide |
| Variance | déterministe | dépend de l'aléatoire |

En résumé :

- LIME = explications simples, rapides et locales  
- SHAP = explications plus complètes et mathématiquement solides  

## Conclusion
LIME est un outil essentiel pour rendre les modèles complexes :

- transparents  
- interprétables  
- pédagogiques  

Dans le cadre du DPE, LIME permet d’expliquer de manière accessible :

- pourquoi le modèle prédit une étiquette énergétique  
- pourquoi la consommation estimée est élevée ou faible  
- quels paramètres ont influencé la prédiction  

Un outil parfait pour communiquer avec des utilisateurs non-experts.

