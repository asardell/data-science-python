# Chapitre 9 : Apprentissage supervisé - Classification

- [Chapitre 9 : Apprentissage supervisé - Classification](#chapitre-9--apprentissage-supervisé---classification)
  - [Introduction à la classification](#introduction-à-la-classification)
  - [Méthodologie générale](#méthodologie-générale)
    - [Préparation de l’échantillon](#préparation-de-léchantillon)
    - [Méthodes de classification](#méthodes-de-classification)
      - [Arbres de décision](#arbres-de-décision)
      - [Random Forest](#random-forest)
      - [K-Nearest Neighbors (KNN)](#k-nearest-neighbors-knn)
      - [Régression logistique](#régression-logistique)
      - [Naive Bayes](#naive-bayes)
      - [Support Vector Machine (SVM)](#support-vector-machine-svm)
      - [Gradient Boosting / XGBoost](#gradient-boosting--xgboost)
    - [Evaluation des modèles](#evaluation-des-modèles)
      - [Matrice de confusion](#matrice-de-confusion)
      - [Accuracy (taux de bonnes classifications)](#accuracy-taux-de-bonnes-classifications)
      - [Précision et rappel](#précision-et-rappel)
      - [F1-Score](#f1-score)
      - [Courbes ROC et AUC](#courbes-roc-et-auc)
    - [Exemple illustratif](#exemple-illustratif)
    - [Compromis sous-apprentissage / sur-apprentissage](#compromis-sous-apprentissage--sur-apprentissage)
    - [Validation croisée](#validation-croisée)
      - [Principe](#principe)
      - [Validation croisée et utilisation après estimation](#validation-croisée-et-utilisation-après-estimation)
    - [Points clés à retenir](#points-clés-à-retenir)
  - [La librairie `scikit-learn`](#la-librairie-scikit-learn)
    - [Pourquoi scikit-learn est-elle si populaire ?](#pourquoi-scikit-learn-est-elle-si-populaire-)
    - [Installation de la librairie](#installation-de-la-librairie)
    - [Preprocessing](#preprocessing)
      - [Chargement du dataset et sélection des variables utiles](#chargement-du-dataset-et-sélection-des-variables-utiles)
      - [Analyse du taux de valeurs manquantes](#analyse-du-taux-de-valeurs-manquantes)
      - [Suppression des colonnes trop manquantes](#suppression-des-colonnes-trop-manquantes)
      - [Imputation des valeurs manquantes restantes](#imputation-des-valeurs-manquantes-restantes)
      - [Création de la variable cible](#création-de-la-variable-cible)
      - [Définition de X (features) et Y (target)](#définition-de-x-features-et-y-target)
      - [Encodage des variables catégorielles](#encodage-des-variables-catégorielles)
      - [Gestion des valeurs extrêmes par clipping](#gestion-des-valeurs-extrêmes-par-clipping)
      - [Séparation train/test](#séparation-traintest)
      - [Analyse échantillon](#analyse-échantillon)
        - [Taille des jeux de données](#taille-des-jeux-de-données)
        - [Répartition de la variable cible (Y)](#répartition-de-la-variable-cible-y)
      - [Standardisation des variables numériques](#standardisation-des-variables-numériques)
      - [Conclusion](#conclusion)
  - [Arbres de décision pour la classification](#arbres-de-décision-pour-la-classification)
    - [Introduction](#introduction)
    - [Préparation de l’échantillon](#préparation-de-léchantillon-1)
    - [Modèle 1 :](#modèle-1-)
      - [Paramètres choisis](#paramètres-choisis)
      - [Entraînement](#entraînement)
      - [Évaluation sur test](#évaluation-sur-test)
      - [Validation croisée](#validation-croisée-1)
    - [Modèle 2 : arbre sans validation croisée](#modèle-2--arbre-sans-validation-croisée)
      - [Paramètres choisis](#paramètres-choisis-1)
      - [Entraînement et évaluation](#entraînement-et-évaluation)
    - [Comparaison des deux modèles avec courbe ROC](#comparaison-des-deux-modèles-avec-courbe-roc)
    - [Optimisation des hyperparamètres avec GridSearchCV](#optimisation-des-hyperparamètres-avec-gridsearchcv)
      - [Définir la grille de paramètres](#définir-la-grille-de-paramètres)
      - [Configuration du GridSearchCV](#configuration-du-gridsearchcv)
      - [Meilleur modèle et hyperparamètres](#meilleur-modèle-et-hyperparamètres)
      - [Évaluation sur l’échantillon test](#évaluation-sur-léchantillon-test)
        - [Prédictions et métriques classiques](#prédictions-et-métriques-classiques)
        - [Courbe ROC pour comparer les modèles](#courbe-roc-pour-comparer-les-modèles)
  - [K-Nearest Neighbors (KNN)](#k-nearest-neighbors-knn-1)
    - [Introduction à KNN](#introduction-à-knn)
    - [Configuration du modèle KNN](#configuration-du-modèle-knn)
    - [Évaluation du modèle](#évaluation-du-modèle)
      - [Matrice de confusion](#matrice-de-confusion-1)
      - [Calcul des métriques](#calcul-des-métriques)
  - [Régression Logistique](#régression-logistique-1)
    - [Présentation de la méthode](#présentation-de-la-méthode)
    - [Hyperparamètres importants](#hyperparamètres-importants)
    - [Définir la grille de recherche](#définir-la-grille-de-recherche)
    - [Mise en œuvre du Grid Search](#mise-en-œuvre-du-grid-search)
      - [Analyse des coefficients](#analyse-des-coefficients)
    - [Prédiction sur l’échantillon test](#prédiction-sur-léchantillon-test)
    - [Évaluation du modèle](#évaluation-du-modèle-1)
      - [Matrice de confusion et métriques](#matrice-de-confusion-et-métriques)
  - [Méthodes bayésiennes — Gaussian Naive Bayes](#méthodes-bayésiennes--gaussian-naive-bayes)
    - [Introduction](#introduction-1)
    - [Pourquoi utiliser une baseline ?](#pourquoi-utiliser-une-baseline-)
    - [Le modèle Gaussian Naive Bayes (GNB)](#le-modèle-gaussian-naive-bayes-gnb)
      - [Hypothèses](#hypothèses)
      - [Avantages](#avantages)
      - [Inconvénients](#inconvénients)
      - [Entraînement du modèle](#entraînement-du-modèle)
      - [Prédictions](#prédictions)
      - [Matrice de confusion](#matrice-de-confusion-2)
      - [Métriques de performance](#métriques-de-performance)
  - [Support Vector Machines (SVM)](#support-vector-machines-svm)
    - [Introduction](#introduction-2)
    - [Principes](#principes)
    - [Avantages](#avantages-1)
    - [Inconvénients](#inconvénients-1)
    - [Implémentation Python](#implémentation-python)
    - [Notes pédagogiques](#notes-pédagogiques)
  - [Random Forest](#random-forest-1)
    - [Introduction à Random Forest](#introduction-à-random-forest)
    - [Principe du Bagging (Bootstrap Aggregating)](#principe-du-bagging-bootstrap-aggregating)
      - [Fonctionnement :](#fonctionnement-)
      - [Avantages du bagging :](#avantages-du-bagging-)
    - [Hyperparamètres](#hyperparamètres)
    - [Importance des variables](#importance-des-variables)
    - [Implémentation](#implémentation)
    - [Évaluation du modèle](#évaluation-du-modèle-2)
    - [Feature Importance](#feature-importance)
    - [Conclusion](#conclusion-1)
  - [Gradient Boosting](#gradient-boosting)
    - [Comprendre le Boosting](#comprendre-le-boosting)
    - [La fonction de perte (loss function)](#la-fonction-de-perte-loss-function)
    - [Boosting vs Bagging](#boosting-vs-bagging)
    - [Pourquoi XGBoost ?](#pourquoi-xgboost-)
    - [Hyperparamètres](#hyperparamètres-1)
    - [Implémentation Complète de XGBoost en Python](#implémentation-complète-de-xgboost-en-python)
    - [Modèle initial](#modèle-initial)
    - [Grille de paramètres](#grille-de-paramètres)
    - [Grid Search](#grid-search)
    - [Modèle final](#modèle-final)
    - [Évaluation du Modèle](#évaluation-du-modèle-3)
    - [Importance des Variables](#importance-des-variables-1)
    - [Avantages et Inconvénients](#avantages-et-inconvénients)
      - [Avantages](#avantages-2)
      - [Inconvénients](#inconvénients-2)
    - [Conseils pratiques](#conseils-pratiques)
    - [Conclusion](#conclusion-2)
  - [Récapitulatif](#récapitulatif)
- [Exercice : Prédiction des étiquettes DPE (classification multi-classes)](#exercice--prédiction-des-étiquettes-dpe-classification-multi-classes)

## Introduction à la classification

La **classification** est une sous-catégorie de l’apprentissage supervisé. Elle consiste à prédire une **variable cible catégorielle** à partir de variables explicatives. L'objectif est d’attribuer à chaque observation une **classe** parmi un ensemble de classes possibles.

**Exemple : projet DPE**

- Variables explicatives : surface habitable, consommation chauffage, type de logement, année de construction, etc.
- Variable cible : `passoire énergétique` (Oui / Non) ou `étiquette DPE` (A, B, C, D, E, F, G)

## Méthodologie générale

### Préparation de l’échantillon

1. **Sélection des variables explicatives et de la cible**  
   - On identifie les variables pertinentes pour prédire la classe cible.  
   - Exemple : `X = [surface, conso_chauffage, conso_ecs, type_logement]`, `Y = passoire_energetique`.

2. **Séparation train / test**  
   - L’ensemble **d’entraînement** sert à apprendre le modèle.  
   - L’ensemble **de test** sert à évaluer la généralisation.  
   - Exemple de répartition : 70% train, 30% test, avec **stratification** pour conserver la proportion de classes.

| Ensemble | Nombre de logements | % Passoire énergétique |
|----------|-------------------|----------------------|
| Train    | 700               | 25%                  |
| Test     | 300               | 25%                  |

3. **Prétraitement**  
   - Gestion des valeurs manquantes (imputation moyenne, médiane ou plus sophistiquée)  
   - Encodage des variables catégorielles (Label Encoding ou One-Hot)  
   - Normalisation ou standardisation si certaines méthodes le nécessitent (SVM, KNN)


### Méthodes de classification

#### Arbres de décision
- Principe : diviser récursivement les données en fonction des variables explicatives pour créer des **règles de décision**.
- Avantages : facile à interpréter, visualisable, pas besoin de normalisation.
- Inconvénients : tendance au **surapprentissage**, sensible aux variations des données.

#### Random Forest
- Principe : ensemble d’arbres de décision combinés pour réduire la variance.
- Avantages : robuste, performante, calcule l’importance des variables.
- Inconvénients : moins interprétable qu’un arbre simple.

#### K-Nearest Neighbors (KNN)
- Principe : classification basée sur la proximité dans l’espace des variables.
- Avantages : simple, efficace pour des petits jeux de données.
- Inconvénients : sensible à l’échelle des variables, coûteux pour de grandes bases.

#### Régression logistique
- Principe : modéliser la probabilité d’appartenance à une classe.
- Avantages : probabilités interprétables, fonctionne bien pour 2 classes.
- Inconvénients : linéarité supposée entre variables et logit, moins performant pour relations complexes.

#### Naive Bayes

- Principe : modèle probabiliste basé sur le théorème de Bayes, supposant que les variables explicatives sont indépendantes conditionnellement à la classe.
- Avantages : très rapide, fonctionne bien même avec peu de données, robuste au bruit, excellent pour les données textuelles.
- Inconvénients : l’hypothèse d’indépendance est rarement vraie, ce qui peut réduire la précision ; moins performant sur des relations complexes entre variables.

#### Support Vector Machine (SVM)
- Principe : séparer les classes en maximisant la **marge** entre elles.
- Avantages : efficace pour données non linéaires via kernels.
- Inconvénients : paramétrage délicat (kernel, C, gamma), peu interprétable.

#### Gradient Boosting / XGBoost

- Principe : combinaison d’arbres de décision construits séquentiellement, chaque arbre corrigeant les erreurs du précédent.
- Avantages : très performant, peut gérer données non linéaires et interactions complexes, options de régularisation.
- Inconvénients : paramétrage complexe, sensible au surapprentissage si mal réglé, moins interprétable.


### Evaluation des modèles

Pour comparer et interpréter les modèles de classification, plusieurs **métriques** sont utilisées :

#### Matrice de confusion

| Observé / Prédit | Oui | Non |
|-----------------|-----|-----|
| Oui             | TP  | FN  |
| Non             | FP  | TN  |

- **TP (True Positive)** : vrai positif, correctement prédit comme Oui  
- **TN (True Negative)** : vrai négatif, correctement prédit comme Non  
- **FP (False Positive)** : faux positif, prédit Oui mais observé Non  
- **FN (False Negative)** : faux négatif, prédit Non mais observé Oui

#### Accuracy (taux de bonnes classifications)
- `Accuracy = (TP + TN) / total`  
- Limité si les classes sont déséquilibrées.

#### Précision et rappel
- **Précision** : proportion de prédictions positives correctes (`TP / (TP + FP)`)  
- **Rappel** : proportion de vrais positifs détectés (`TP / (TP + FN)`)

#### F1-Score
- Moyenne harmonique entre précision et rappel, utile en cas de déséquilibre des classes.

\[
F1 = 2 \times \frac{\text{Précision} \times \text{Rappel}}{\text{Précision} + \text{Rappel}}
\]

#### Courbes ROC et AUC
- Permettent de visualiser le compromis entre **taux de vrais positifs** et **taux de faux positifs**.  
- L’AUC mesure la capacité globale du modèle à distinguer les classes (1 = parfait, 0.5 = hasard).


<p align="center">
  <img src="https://cdn.prod.website-files.com/660ef16a9e0687d9cc27474a/662c42679571ef35419c9935_647607123e84a06a426ce627_classification_metrics_014-min.png" alt="Source de l'image" width="600"/>
</p>



### Exemple illustratif

**Jeu de données DPE : prédire `passoire énergétique` (Oui/Non)**

- Modèle : arbre de décision  
- Résultat sur test :

| Observé / Prédit | Oui | Non |
|-----------------|-----|-----|
| Oui             | 50  | 10  |
| Non             | 20  | 220 |

- Accuracy : (50 + 220) / 300 = 90%  
- Précision pour Oui : 50 / (50 + 20) ≈ 0,714  
- Rappel pour Oui : 50 / (50 + 10) ≈ 0,833  
- F1-Score : 2 * (0,714 * 0,833) / (0,714 + 0,833) ≈ 0,77  

**Interprétation** : le modèle détecte correctement la majorité des passoires énergétiques, mais certains faux négatifs restent.

### Compromis sous-apprentissage / sur-apprentissage

- **Sous-apprentissage (underfitting)** :  
  - Le modèle est trop simple pour capturer les relations dans les données.  
  - Symptômes : faible performance sur les données d’entraînement et de test, erreur élevée.  
  - Causes possibles : modèle trop simple (ex. arbre de décision peu profond), trop peu de variables ou pas assez de données.  
  - Solutions : utiliser un modèle plus complexe, ajouter des variables pertinentes, augmenter le nombre de données, réduire la régularisation.  

- **Sur-apprentissage (overfitting)** :  
  - Le modèle est trop complexe et capture le bruit spécifique aux données d’entraînement.  
  - Symptômes : très bonne performance sur les données d’entraînement mais faible performance sur les données de test.  
  - Causes possibles : modèle trop flexible (ex. arbre très profond), trop de variables ou interactions inutiles, peu de données par rapport à la complexité.  
  - Solutions : simplifier le modèle, augmenter la taille du jeu de données, validation croisée pour ajuster les hyperparamètres.  

- **Compromis** :  
  - L’objectif est de trouver un modèle qui généralise bien, c’est-à-dire qui a une bonne performance à la fois sur les données d’entraînement et de test.  
  - Outils pratiques : courbes d’apprentissage (learning curves), validation croisée, analyse des erreurs sur train/test.

### Validation croisée

#### Principe

  La validation croisée permet d’évaluer la performance d’un modèle de manière plus robuste qu’une simple séparation train/test.  
  Elle consiste à découper le jeu de données en plusieurs sous-ensembles (folds), puis à entraîner le modèle sur une partie et tester sur l’autre, en répétant le processus plusieurs fois.

- **Méthodes courantes** :  
  - **k-fold cross-validation** : le jeu de données est divisé en k sous-ensembles. Chaque fold sert une fois de test et les k-1 autres de train. La performance finale est la moyenne des k scores.  
  - **Leave-One-Out (LOO)** : chaque observation est utilisée une fois comme test, le reste comme train. Utile pour petits jeux de données.  
  - **Stratified k-fold** : même principe que k-fold mais en conservant la proportion des classes dans chaque fold, très utile pour les problèmes de classification déséquilibrée.

<p align="center">
  <img src="https://scikit-learn.org/stable/_images/grid_search_cross_validation.png" alt="Source de l'image" width="600"/>
</p>


- **Avantages** :  
  - Fournit une estimation plus fiable de la performance réelle du modèle.  
  - Permet de détecter le sur-apprentissage si le score moyen sur les folds est proche du score sur l’entraînement mais loin de celui sur le test.  
  - Permet d’optimiser les hyperparamètres via des grilles de recherche (Grid Search avec validation croisée).

- **Inconvénients** :  
  - Plus coûteux en temps de calcul que la simple séparation train/test.  
  - Peut être complexe à mettre en œuvre sur de très grands jeux de données si k est élevé.

- **Exemple conceptuel** :  
  Si on a un jeu de données de 1000 observations et k=5 :  
  - Découper le jeu en 5 folds de 200 observations chacun.  
  - Tour 1 : Fold 1 = test, Fold 2-5 = train  
  - Tour 2 : Fold 2 = test, Fold 1,3,4,5 = train  
  - … répéter jusqu’au 5ème fold  
  - Calculer la moyenne des performances (ex. accuracy ou f1-score) sur les 5 tests pour obtenir une estimation globale.

#### Validation croisée et utilisation après estimation

1. **Estimation des performances**  
   - La validation croisée fournit une **moyenne des scores** (accuracy, F1, etc.) et leur **écart-type**.  
   - Elle permet de comparer différents modèles ou réglages d’hyperparamètres de façon robuste.

2. **Sélection du meilleur modèle / hyperparamètres**  
   - Si l’on utilise un **Grid Search CV**, on choisit la combinaison de paramètres donnant le meilleur score moyen sur les folds.  
   - Pour un modèle sans hyperparamètres à optimiser, la validation croisée fournit juste une estimation fiable de la performance attendue.

3. **Ré-entrainement sur tout l’ensemble d’entraînement**  
   - Une fois le modèle ou les hyperparamètres choisis, on **ré-entraine le modèle sur l’intégralité du jeu de données d’entraînement**.  
   - Cela permet au modèle d’apprendre à partir de **toutes les observations disponibles**.

4. **Évaluation finale sur le test set**  
   - Après avoir entraîné sur tout le train, on effectue une évaluation sur l’ensemble de test pour obtenir une **performance finale réaliste**.

### Points clés à retenir

- La classification supervise l’apprentissage sur des **données étiquetées**.  
- Les méthodes sont nombreuses et leur choix dépend :
  - de la **taille du jeu de données**  
  - du **type de variables**  
  - de la **complexité des relations**  
  - de la **besoin d’interprétabilité**  
- La séparation **train/test** et les métriques de performance sont indispensables pour évaluer la **généralisation**.  
- En pratique, comparer plusieurs modèles et utiliser des métriques complémentaires permet de choisir le meilleur modèle pour un projet réel comme le DPE.

## La librairie `scikit-learn`

La librairie **scikit-learn** (souvent appelée *sklearn*) est l’un des outils les plus utilisés en machine learning. Elle fournit un ensemble cohérent, simple d’utilisation et performant pour créer, entraîner, évaluer et déployer des modèles de machine learning classiques.

### Pourquoi scikit-learn est-elle si populaire ?

Scikit-learn est largement adoptée pour plusieurs raisons :

- **Interface unifiée** : tous les modèles utilisent les mêmes méthodes (`fit`, `predict`, `transform`, etc.)
- **Documentation claire** : très accessible pour les débutants comme pour les experts
- **Communauté active** : nombreuses ressources, tutoriels, exemples
- **Performances solides** : implémentations optimisées en C/Cython
- **Compatibilité avec NumPy, Pandas et SciPy** : s’intègre parfaitement à l’écosystème Python

Elle est idéale pour les projets de machine learning **supervisé** (régression, classification) et **non supervisé** (clustering, réduction de dimension).

### Installation de la librairie

```shell
pip install scikit-learn
```

### Preprocessing
 
Les opérations suivantes couvrent tout le pipeline de preprocessing :  
sélection des variables, nettoyage, traitement des valeurs manquantes, encodage, gestion des valeurs extrêmes, standardisation et séparation train/test.


#### Chargement du dataset et sélection des variables utiles

Avant toute préparation, il est nécessaire de charger les données et de sélectionner les colonnes pertinentes.  
Ici, on se limite à une extraction de 10000 lignes pour illustrer le processus.

Ensuite, on conserve uniquement une liste de variables importantes pour la modélisation, incluant des caractéristiques du bâtiment, des déperditions, des consommations et l'étiquette énergétique.


```python
import pandas as pd
import numpy as np

dfDpe = pd.read_csv("dpe_2021_2025.csv", nrows=10000)

cols_utiles = [
    "conso_5_usages_ef", "deperditions_planchers_bas", "annee_construction",
    "type_generateur_chauffage_principal", "type_batiment",
    "type_energie_principale_chauffage", "numero_dpe",
    "surface_habitable_immeuble", "deperditions_murs", "conso_ecs_ef",
    "nombre_appartement", "deperditions_portes", "deperditions_renouvellement_air",
    "deperditions_baies_vitrees", "deperditions_planchers_hauts",
    "zone_climatique", "type_energie_n1", "deperditions_ponts_thermiques",
    "type_installation_chauffage", "surface_habitable_logement",
    "hauteur_sous_plafond", "conso_chauffage_ef", "etiquette_dpe"
]

df = dfDpe[cols_utiles].copy()
```

#### Analyse du taux de valeurs manquantes

L’analyse des valeurs manquantes est une étape incontournable.  
Elle permet de visualiser quelles colonnes sont partiellement ou massivement incomplètes.

Le calcul du pourcentage de valeurs manquantes par colonne guide la stratégie de nettoyage.

```python
missing_rate = df.isna().mean().sort_values(ascending=False) * 100
print("Taux de valeurs manquantes (%) par colonne :")
print(missing_rate)
```

#### Suppression des colonnes trop manquantes

Une colonne contenant plus de 75 % de valeurs manquantes apporte très peu d’information et génère souvent plus de bruit que de valeur.  
On définit donc un seuil de suppression.

```python
threshold = 0.75
cols_to_drop = missing_rate[missing_rate > threshold * 100].index.tolist()

print("Colonnes supprimées (> 75% missing) :")
print(cols_to_drop)

df = df.drop(columns=cols_to_drop)
```

#### Imputation des valeurs manquantes restantes

Une fois les colonnes très manquantes supprimées, il reste des valeurs isolées à traiter.

Deux stratégies sont appliquées :  
- médiane pour les variables numériques  
- mode pour les variables catégorielles  

Ces techniques sont robustes et évitent d’introduire des valeurs extrêmes.

```python
num_cols = df.select_dtypes(include=["int64", "float64"]).columns
cat_cols = df.select_dtypes(include=["object"]).columns

for col in num_cols:
    median_val = df[col].median()
    df[col] = df[col].fillna(median_val)

for col in cat_cols:
    mode_val = df[col].mode(dropna=True)
    if len(mode_val) > 0:
        df[col] = df[col].fillna(mode_val.iloc[0])
    else:
        df[col] = df[col].fillna("Inconnu")

print("Imputation terminée.")
print(df.isna().sum())
```

#### Création de la variable cible

On cherche ici à prédire si un logement est classé en passoire énergétique.  
On définit donc une variable binaire basée sur l'étiquette DPE.

```python
df["passoire_energetique"] = df["etiquette_dpe"].isin(["E", "F", "G"])
```

Toutes les lignes contenant encore des valeurs manquantes sont ensuite supprimées.

```python
df = df.dropna()
```

#### Définition de X (features) et Y (target)

Certaines variables, comme les consommations, sont retirées du modèle afin d’éviter des fuites de données.  
Les variables explicatives contiennent donc uniquement des caractéristiques du bâtiment.

```python
vars_conso = ["conso_5_usages_ef", "conso_chauffage_ef", "conso_ecs_ef"]

Y = df["passoire_energetique"]
X = df.drop(columns=["passoire_energetique", "etiquette_dpe"] + vars_conso)
```

#### Encodage des variables catégorielles

Le modèle ne peut exploiter que des valeurs numériques.  
Les variables textuelles sont donc transformées en variables indicatrices (one-hot encoding).

```python
X = pd.get_dummies(X, drop_first=True)
```

#### Gestion des valeurs extrêmes par clipping

Les valeurs aberrantes perturbent fortement les modèles.  
On applique ici un clipping entre les quantiles 1 % et 99 % afin de limiter l’influence de ces extrêmes sans supprimer d'informations.

```python
num_cols = X.select_dtypes(include=["int64", "float64"]).columns

def manage_outlier(col, low=0.01, high=0.99):
    q_low = col.quantile(low)
    q_high = col.quantile(high)
    return col.clip(lower=q_low, upper=q_high)

for c in num_cols:
    X[c] = manage_outlier(X[c])

print("Outliers capés")
```

#### Séparation train/test

Pour évaluer correctement la performance d’un modèle, on sépare le dataset en un ensemble d’entraînement et un ensemble de test.  
La stratification permet d’équilibrer la proportion de passoires énergétiques dans les deux sous-échantillons.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.30, random_state=42, stratify=Y
)

print("Train/test split réalisé.")
```

#### Analyse échantillon

Une fois le découpage entre les jeux **X_train**, **X_test**, **Y_train** et **Y_test** effectué, il est indispensable de vérifier que l’échantillonnage est équilibré et cohérent.  
L’objectif est de s’assurer que la proportion de *passoires énergétiques* (valeur cible) est similaire entre les deux jeux, afin d’éviter un biais dans l’apprentissage du modèle.

##### Taille des jeux de données

Le premier contrôle consiste à afficher le nombre d’observations dans :
- le dataset total,
- le jeu d’entraînement,
- le jeu de test.

Exemple d’analyse :

```python
print("Taille totale du dataset :", len(X))
print("Taille X_train :", len(X_train))
print("Taille X_test  :", len(X_test))
print("Taille y_train :", len(y_train))
print("Taille y_test  :", len(y_test))
```

##### Répartition de la variable cible (Y)

Comme la variable **passoire_energetique** est binaire, il est essentiel d’examiner sa distribution.  
On utilise ici `value_counts(normalize=True)` afin d’obtenir les proportions.

```python
print("\nDistribution de Y_train :")
print(Y_train.value_counts(normalize=True))

print("\nDistribution de Y_test :")
print(Y_test.value_counts(normalize=True))
```

L’objectif est que les proportions soient **relativement similaires** entre *train* et *test*, ce qui est garanti par l’option `stratify=Y` dans `train_test_split`.

#### Standardisation des variables numériques

La standardisation doit impérativement être effectuée **après le train/test split**.  
En effet, appliquer un `fit_transform()` sur l’ensemble du dataset avant le découpage introduirait une **fuite d’information** : les statistiques (moyenne, écart-type) calculées sur tout le dataset intègrent alors une partie du test, ce qui biaise l’évaluation du modèle.

La règle correcte est donc :

- **fit uniquement sur X_train**  
- **transform sur X_train ET X_test**

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# Fit uniquement sur l'entraînement
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])

# Transform sur l'entraînement et le test
X_test[num_cols] = scaler.transform(X_test[num_cols])

print("Standardisation terminée (post-split).")

X_test
```

Cette approche garantit que le modèle n’a accès à aucune information provenant du jeu de test pendant la phase d’entraînement.

#### Conclusion

Le pipeline de preprocessing présenté ici constitue un flux complet et fiable pour préparer un dataset avant toute modélisation ML.  
Il couvre :

- la sélection des variables  
- l'analyse des valeurs manquantes  
- la suppression des colonnes trop incomplètes  
- l’imputation robuste  
- la création d’une cible pertinente  
- l’encodage des variables catégorielles  
- le traitement des outliers  
- la séparation train/test  
- la standardisation  

Ce cadre constitue une base solide des bonnes pratiques du preprocessing.


## Arbres de décision pour la classification

### Introduction

Les **arbres de décision** sont des modèles supervisés utilisés pour la classification.
Ils segmentent l’espace des variables explicatives en fonction de règles simples afin de prédire une variable cible.

**Avantages** :
- Facilement interprétables.
- Peu sensibles à la normalisation des variables.
- Gèrent variables numériques et catégorielles.

**Inconvénients** :
- Sensibles au sur-apprentissage si l’arbre est trop profond.
- Peu robustes aux petites variations des données.

<p align="center">
  <img src="https://scikit-learn.org/stable/_images/sphx_glr_plot_iris_dtc_002.png" alt="Source de l'image" width="600"/>
</p>


### Préparation de l’échantillon

- Séparer les variables explicatives `X` de la variable cible `y`.
- Découper l’échantillon en **train (70%)** et **test (30%)**, en stratifiant selon la variable cible pour conserver les proportions de classes.

Exemple de tableau de proportions des classes :

| Classe | Train | Test |
|--------|-------|------|
| 0      | 0.65  | 0.65 |
| 1      | 0.35  | 0.35 |


### Modèle 1 :

#### Paramètres choisis

- `max_depth=4` : limite la profondeur pour éviter le sur-apprentissage.
- `min_samples_leaf=30` : nombre minimum d’observations dans une feuille.
- `min_samples_split=50` : nombre minimum d’observations pour créer une division.

#### Entraînement

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree

model_arbre = DecisionTreeClassifier(max_depth=4, min_samples_leaf=30, min_samples_split=50, random_state=42)
model_arbre.fit(X_train, y_train)

plt.figure(figsize=(16,4))
plot_tree(model_arbre, feature_names=X.columns, filled=True, fontsize=10)
plt.show()
```

#### Évaluation sur test

```python
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
import pandas as pd

# Prédiction
y_pred = model_arbre.predict(X_test)
y_pred_proba = model_arbre.predict_proba(X_test)

# Matrice de confusion
mc = pd.DataFrame(confusion_matrix(y_test, y_pred),
                     columns=['pred_0','pred_1'],
                     index=['obs_0','obs_1'])
print(mc)

# Métriques
print('Accuracy :', accuracy_score(y_test, y_pred))
print('Recall (binaire) :', recall_score(y_test, y_pred, average='binary', pos_label=0))
print('Precision (binaire) :', precision_score(y_test, y_pred, average='binary', pos_label=0))
print('F1-score (binaire) :', f1_score(y_test, y_pred, average='binary', pos_label=0))

# Micro / Macro / Weighted
print('Recall (macro) :', recall_score(y_test, y_pred, average='macro'))
print('Precision (macro) :', precision_score(y_test, y_pred, average='macro'))
print('F1-score (macro) :', f1_score(y_test, y_pred, average='macro'))

print('Recall (weighted) :', recall_score(y_test, y_pred, average='weighted'))
print('Precision (weighted) :', precision_score(y_test, y_pred, average='weighted'))
print('F1-score (weighted) :', f1_score(y_test, y_pred, average='weighted'))
```

- `pos_label=0` indique que **0 (False)** est considéré comme la **classe positive** pour le calcul de la précision.  
- `average='macro'` calcule la **moyenne simple** du F1-score sur toutes les classes, sans tenir compte de leur proportion.  
- `average='weighted'` calcule la **moyenne pondérée** du rappel (ou autre métrique) en tenant compte du **nombre d’échantillons de chaque classe**.

#### Validation croisée

```python
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer

score = make_scorer(f1_score, pos_label=0)
cv_scores = cross_val_score(model_arbre, X_train, y_train, cv=5, scoring=score)
print('Scores CV:', cv_scores)
print('Moyenne CV:', cv_scores.mean())
```

### Modèle 2 : arbre sans validation croisée

#### Paramètres choisis

- `max_depth=6` : arbre plus profond.
- `min_samples_leaf=10` : feuilles plus petites.
- `min_samples_split=20` : divisions plus fréquentes.

#### Entraînement et évaluation

```python
model_arbre2 = DecisionTreeClassifier(max_depth=6, min_samples_leaf=10, min_samples_split=20, random_state=42)
model_arbre2.fit(X_train, y_train)

# Prédiction
y_pred2 = model_arbre2.predict(X_test)
y_pred_proba2 = model_arbre2.predict_proba(X_test)

# Métriques
print('Accuracy :', accuracy_score(y_test, y_pred2))
print('F1-score (macro) :', f1_score(y_test, y_pred2, average='macro'))
```

### Comparaison des deux modèles avec courbe ROC

```python
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Probabilités pour la classe positive
proba1 = y_pred_proba[:,1]
proba2 = y_pred_proba2[:,1]

# ROC
fpr1, tpr1, _ = roc_curve(y_test, proba1)
fpr2, tpr2, _ = roc_curve(y_test, proba2)

auc1 = roc_auc_score(y_test, proba1)
auc2 = roc_auc_score(y_test, proba2)

plt.figure(figsize=(8,6))
plt.plot(fpr1, tpr1, label=f'Modèle 1 CV (AUC={auc1:.2f})', color='blue')
plt.plot(fpr2, tpr2, label=f'Modèle 2 (AUC={auc2:.2f})', color='green')
plt.plot([0,1],[0,1],'k--', label='Aléatoire')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('Comparaison des courbes ROC')
plt.legend()
plt.show()
```


### Optimisation des hyperparamètres avec GridSearchCV

Dans cette section, nous allons montrer comment utiliser `GridSearchCV` pour optimiser un modèle d'arbre de décision, puis comparer ses performances avec deux autres modèles précédemment entraînés.

#### Définir la grille de paramètres

La grille de paramètres définit les valeurs possibles pour chaque hyperparamètre que l’on souhaite tester.

```python
import numpy as np

parameters = {
    'max_depth': np.arange(1, 10, 1),
    'min_samples_leaf': np.arange(5, 250, 50),
    'min_samples_split': np.arange(10, 500, 50)
}

total_combinaisons = (
    len(parameters['max_depth']) *
    len(parameters['min_samples_leaf']) *
    len(parameters['min_samples_split'])
)

print(f"Nombre total de modèles à tester: {total_combinaisons}")
```

#### Configuration du GridSearchCV

Nous allons utiliser `f1_score` comme métrique pour comparer les modèles, en nous intéressant particulièrement à la classe positive (ici pos_label=0).

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score

score = make_scorer(f1_score, pos_label=0)
model_arbre_grid = DecisionTreeClassifier()
grid_search = GridSearchCV(model_arbre_grid, parameters, scoring=score, verbose=2, cv=5)
grid_search.fit(X_train, y_train)
```

**Explication :**  
- `cv=5` : validation croisée à 5 plis pour évaluer chaque combinaison.  
- `verbose=2` : affichage des étapes pour suivre la progression.  
- `scoring=score` : métrique utilisée pour sélectionner le meilleur modèle.  

#### Meilleur modèle et hyperparamètres

```python
best_model = grid_search.best_estimator_

print("Meilleurs paramètres :", grid_search.best_params_)
print("Score f1 du meilleur modèle :", grid_search.best_score_)
```

**Remarque :**  
Le `best_model` est **réentraîné sur l’intégralité de X_train** avec les hyperparamètres optimaux.  

#### Évaluation sur l’échantillon test

##### Prédictions et métriques classiques

```python
y_pred_grid = best_model.predict(X_test)

from sklearn.metrics import classification_report, accuracy_score, recall_score, precision_score, f1_score

print(classification_report(y_test, y_pred_grid))
print("Accuracy :", accuracy_score(y_test, y_pred_grid))
print("Recall (binary, pos_label=0) :", recall_score(y_test, y_pred_grid, average='binary', pos_label=0))
print("Precision (binary, pos_label=0) :", precision_score(y_test, y_pred_grid, average='binary', pos_label=0))
print("F1-score (binary, pos_label=0) :", f1_score(y_test, y_pred_grid, average='binary', pos_label=0))

print("Recall (macro) :", recall_score(y_test, y_pred_grid, average='macro'))
print("Precision (macro) :", precision_score(y_test, y_pred_grid, average='macro'))
print("F1-score (macro) :", f1_score(y_test, y_pred_grid, average='macro'))

print("Recall (weighted) :", recall_score(y_test, y_pred_grid, average='weighted'))
print("Precision (weighted) :", precision_score(y_test, y_pred_grid, average='weighted'))
print("F1-score (weighted) :", f1_score(y_test, y_pred_grid, average='weighted'))
```

##### Courbe ROC pour comparer les modèles

Nous comparons le `best_model` GridSearch avec les deux autres modèles d’arbre précédemment entraînés.

```python
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Probabilités de prédiction pour la classe positive
y_proba_grid = best_model.predict_proba(X_test)[:,1]
y_proba_model1 = model_arbre_cv.predict_proba(X_test)[:,1]
y_proba_model2 = model_arbre_2.predict_proba(X_test)[:,1]

# Calcul ROC
fpr_grid, tpr_grid, _ = roc_curve(y_test, y_proba_grid)
fpr1, tpr1, _ = roc_curve(y_test, y_proba_model1)
fpr2, tpr2, _ = roc_curve(y_test, y_proba_model2)

# AUC
auc_grid = roc_auc_score(y_test, y_proba_grid)
auc1 = roc_auc_score(y_test, y_proba_model1)
auc2 = roc_auc_score(y_test, y_proba_model2)

# Tracé
plt.figure(figsize=(8,6))
plt.plot(fpr_grid, tpr_grid, label=f'GridSearchCV (AUC={auc_grid:.2f})', color='red')
plt.plot(fpr1, tpr1, label=f'Modèle 1 (AUC={auc1:.2f})', color='blue')
plt.plot(fpr2, tpr2, label=f'Modèle 2 (AUC={auc2:.2f})', color='green')
plt.plot([0,1],[0,1],'k--', label='Aléatoire')
plt.xlabel('Faux positifs (FPR)')
plt.ylabel('Vrais positifs (TPR)')
plt.title('Comparaison ROC entre modèles d\'arbre')
plt.legend(loc='lower right')
plt.show()
```

## K-Nearest Neighbors (KNN)

### Introduction à KNN

**Principe** :
KNN (K-Nearest Neighbors) est une méthode d'apprentissage supervisé utilisée pour la classification (ou régression).
- Pour classer un nouvel individu, KNN regarde les **K plus proches voisins** dans l’espace des variables explicatives et attribue la classe majoritaire parmi ces voisins.
- La proximité est généralement mesurée avec la distance euclidienne, mais d’autres distances peuvent être utilisées.

<p align="center">
  <img src="https://miro.medium.com/v2/resize:fit:650/1*OyYyr9qY-w8RkaRh2TKo0w.png" alt="Source de l'image" width="600"/>
</p>


**Avantages :**
- Simple à comprendre et à implémenter.
- Non paramétrique : aucune hypothèse sur la distribution des données.

**Inconvénients :**
- Sensible aux variables sur des échelles différentes (nécessite normalisation).
- Peu efficace pour de très grandes bases de données.
- Choix du paramètre K crucial : trop petit → bruit, trop grand → perte de sensibilité locale.


### Configuration du modèle KNN

**Paramètres du modèle :**
- `n_neighbors` : nombre de voisins à considérer pour la classification.
- `weights` : "uniform" (tous les voisins ont le même poids) ou "distance" (pondération inversement proportionnelle à la distance).
- `metric` : type de distance pour calculer la proximité ("euclidean", "manhattan", etc.).


```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, make_scorer, classification_report

# Définition du modèle KNN
knn = KNeighborsClassifier()

# Paramètres à tester dans Grid Search
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

# Définir le scorer : f1 pour la classe positive
f1_scorer = make_scorer(f1_score, pos_label=0)

# Grid Search
grid_knn = GridSearchCV(knn, param_grid, scoring=f1_scorer, cv=5, verbose=2)
grid_knn.fit(X_train, y_train)

# Meilleur modèle
best_knn = grid_knn.best_estimator_
print("Meilleur modèle KNN :", best_knn)
print("Meilleur score f1 (classe 0) :", grid_knn.best_score_)

# Prédictions sur l'ensemble de test
y_pred_knn = best_knn.predict(X_test)
```

### Évaluation du modèle

#### Matrice de confusion

```python
import pandas as pd
from sklearn.metrics import confusion_matrix

mc_knn = pd.DataFrame(confusion_matrix(y_test, y_pred_knn),
                      columns=['pred_0','pred_1'],
                      index=['obs_0','obs_1'])
print(mc_knn)
```

#### Calcul des métriques

```python
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

print("Accuracy :", accuracy_score(y_test, y_pred_knn))
print("Recall (classe 0) :", recall_score(y_test, y_pred_knn, pos_label=0))
print("Precision (classe 0) :", precision_score(y_test, y_pred_knn, pos_label=0))
print("F1-score (classe 0) :", f1_score(y_test, y_pred_knn, pos_label=0))
```

## Régression Logistique

### Présentation de la méthode

- La **régression logistique** est utilisée pour la classification binaire ou multi-classes.
- Modélise la **probabilité** qu’une observation appartienne à une classe à partir d’une combinaison linéaire des variables explicatives.
- Avantages :
  - Simple, interprétable, rapide.
  - Permet d’évaluer l’importance des variables via les coefficients.
- Inconvénients :
  - Ne gère pas naturellement les relations non linéaires.
  - Sensible aux variables corrélées.
  - Nécessite standardisation pour certaines régularisations.


Même si la régression logistique est utilisée pour **des problèmes de classification** (par exemple, prédire si un logement est éligible à la prime rénov ou non), elle repose sur **un modèle linéaire sous-jacent** :

<p align="center">
  <img src="https://94fa3c88.delivery.rocketcdn.me/wp-content/uploads/2020/11/illu_regression_blog-16.png" alt="Source de l'image" width="600"/>
</p>

1. **Transformation linéaire des variables explicatives**  
   La régression logistique calcule une **combinaison linéaire** des variables explicatives, exactement comme dans une régression linéaire classique :

   ```
   z = b0 + b1*x1 + b2*x2 + ... + bn*xn
   ```

   Ici, `z` est une valeur continue qui résume l’influence des variables sur la probabilité d’appartenance à la classe cible.

2. **Fonction logistique (sigmoïde)**  
   Pour transformer cette combinaison linéaire `z` en une **probabilité comprise entre 0 et 1**, on applique la **fonction sigmoïde** :

   ```
   p = 1 / (1 + exp(-z))
   ```

   Cette probabilité `p` représente la chance que l’observation appartienne à la classe positive.

3. **Classification finale**  
   On fixe un **seuil** (souvent 0.5) pour décider de la classe :  
   - Si `p >= 0.5`, l’observation est classée comme positive.  
   - Sinon, elle est classée comme négative.

4. **Pourquoi linéaire ?**  
   - Les coefficients `b1, b2, …` indiquent l’influence **linéaire** de chaque variable sur le score `z`.  
   - La non-linéarité de la classification vient **de la transformation sigmoïde**, pas des coefficients eux-mêmes.  
   - C’est pour ça qu’on parle de **modèle linéaire pour un problème de classification** : la structure du modèle reste linéaire dans les paramètres, mais la sortie est convertie en probabilité.

5. **Avantages de cette approche**  
   - Simple à interpréter : chaque coefficient indique l’impact sur le log-odds.  
   - Rapide à entraîner et efficace sur des données de taille moyenne.

6. **Limites**  
   - Suppose que la relation entre les variables et le log-odds est **linéaire**.  
   - Ne gère pas facilement les interactions complexes ou les relations fortement non linéaires.  
     (Dans ce cas, on peut envisager des transformations ou des modèles plus flexibles comme les arbres ou SVM).


### Hyperparamètres importants

- `penalty` : type de régularisation (`l1`, `l2`, `elasticnet`, `none`).
- `C` : inverse de la force de régularisation (plus C est grand, moins la régularisation est forte).
- `solver` : algorithme d’optimisation (`liblinear`, `lbfgs`, `saga`…).
- `max_iter` : nombre maximum d’itérations pour la convergence.


### Définir la grille de recherche

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score

# Hyperparamètres à tester
param_grid = {
    'penalty': ['l1', 'l2'],
    'C': [0.01, 0.1, 1, 10],
    'solver': ['liblinear', 'saga'],
    'max_iter': [500, 1000]
}

# Choix de la métrique F1 pour la classe positive (pos_label=0)
scorer = make_scorer(f1_score, pos_label=0)
```


### Mise en œuvre du Grid Search

```python
# Création du modèle de base
log_reg = LogisticRegression()

# Grid Search avec validation croisée 5 plis
grid_search = GridSearchCV(log_reg, param_grid, scoring=scorer, cv=5, verbose=2)
grid_search.fit(X_train, y_train)

# Meilleurs paramètres
best_model = grid_search.best_estimator_
print("Meilleurs paramètres :", grid_search.best_params_)
print("Meilleur score F1 (CV) :", grid_search.best_score_)
```

#### Analyse des coefficients

```python
coef = pd.DataFrame(best_model.coef_[0], index=X_train.columns, columns=['Coef'])
coef.loc['Intercept'] = best_model.intercept_
print(coef)
```

- Chaque coefficient représente l’effet de la variable sur la **log-odds** de la classe positive.
- Signe positif → augmente la probabilité d’appartenir à la classe 1.
- Signe négatif → augmente la probabilité d’appartenir à la classe 0.

### Prédiction sur l’échantillon test

```python
# Prédiction des classes
y_pred = best_model.predict(X_test)

# Probabilités pour la classe positive (0)
y_pred_proba = best_model.predict_proba(X_test)[:, 0]
```

### Évaluation du modèle

#### Matrice de confusion et métriques

```python
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

mc = pd.DataFrame(confusion_matrix(y_test, y_pred),
                  columns=['pred_0','pred_1'],
                  index=['obs_0','obs_1'])
print(mc)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Recall, Precision, F1 pour pos_label=0
print("Recall (pos_label=0):", recall_score(y_test, y_pred, pos_label=0))
print("Precision (pos_label=0):", precision_score(y_test, y_pred, pos_label=0))
print("F1-score (pos_label=0):", f1_score(y_test, y_pred, pos_label=0))
```



## Méthodes bayésiennes — Gaussian Naive Bayes  

### Introduction

Les méthodes bayésiennes reposent sur le **théorème de Bayes**, permettant d’estimer la probabilité qu’un individu appartienne à une classe donnée en tenant compte des données observées.

Formule générale du théorème de Bayes :

P(classe | données) = P(données | classe) × P(classe) / P(données)

<p align="center">
  <img src="https://databasecamp.de/wp-content/uploads/naive-bayes-overview-1024x709.png" alt="Source de l'image" width="600"/>
</p>


Un classifieur bayésien choisit la classe **maximisant la probabilité a posteriori**.

Naive Bayes n’est généralement pas le modèle principal dans des pipelines industriels avancés.  
Il sert surtout comme :

- **baseline rapide** (modèle de référence initial)
- modèle phare dans le **traitement du texte (NLP)**  
  (spam filtering, classification de documents)
- méthode très efficace dans des **cas simples** où l’hypothèse d’indépendance est acceptable


Une **baseline** (ou *modèle de référence*) est un modèle simple, rapide à entraîner, utilisé comme **point de départ** pour évaluer les modèles plus complexes.


### Pourquoi utiliser une baseline ?

La baseline sert à répondre à une question essentielle : **« Est-ce que les modèles plus avancés améliorent vraiment les performances ? »**

Elle permet de :

- disposer d’un **niveau minimal de performance** ;
- détecter si un modèle complexe **n’apporte rien** ;
- éviter du sur-apprentissage technologique (utiliser un gros modèle alors qu’un simple suffit).


Exemples typiques de baseline

- **Naive Bayes**  
- **KNN avec petite valeur de k**  
- **Arbre de décision peu profond**  

:bulb: À quoi ça sert concrètement ?

- Si un modèle complexe **ne fait pas mieux** que la baseline il n’est pas utile ou mal paramétré.
- Si un modèle complexe **surpasse la baseline**  l’amélioration est justifiée.
- Si la baseline est **déjà très performante** inutile de chercher un modèle plus lourd.

### Le modèle Gaussian Naive Bayes (GNB)

####  Hypothèses

Gaussian Naive Bayes repose sur deux hypothèses :

1. **Indépendance conditionnelle** entre toutes les variables explicatives  
2. **Distribution normale (gaussienne)** des variables dans chaque classe

Même si ces hypothèses sont rarement parfaitement vérifiées, le modèle fonctionne étonnamment bien dans de nombreux cas.

#### Avantages
- Très rapide à entraîner
- Performant même avec peu de données
- Insensible aux valeurs manquantes si elles sont gérées en amont

#### Inconvénients
- Hypothèse d’indépendance souvent irréaliste
- Moins performant si variables fortement corrélées
- Suppose des distributions gaussiennes pour chaque variable


#### Entraînement du modèle

```python
from sklearn.naive_bayes import GaussianNB

model_gnb = GaussianNB()
model_gnb.fit(X_train, y_train)
```

#### Prédictions

```python
y_pred = model_gnb.predict(X_test)
y_proba = model_gnb.predict_proba(X_test)
```

#### Matrice de confusion

```python
from sklearn.metrics import confusion_matrix
import pandas as pd

mc = pd.DataFrame(
    confusion_matrix(y_test, y_pred),
    columns=['pred_0','pred_1'],
    index=['obs_0','obs_1']
)
mc
```

#### Métriques de performance

```python
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

print("Accuracy :", accuracy_score(y_test, y_pred))
print("Recall (classe 0) :", recall_score(y_test, y_pred, average='binary', pos_label=0))
print("Precision (classe 0) :", precision_score(y_test, y_pred, average='binary', pos_label=0))
print("F1-score (classe 0) :", f1_score(y_test, y_pred, average='binary', pos_label=0))
```


## Support Vector Machines (SVM)

###  Introduction

Le **SVM (Support Vector Machine)** est une méthode de classification supervisée qui cherche à **séparer les classes en maximisant la marge** entre elles. Elle peut être utilisée pour des problèmes linéaires ou non linéaires via des kernels.


<p align="center">
  <img src="https://scikit-learn.org/stable/_images/sphx_glr_plot_iris_svc_001.png" alt="Source de l'image" width="600"/>
</p>

### Principes
- Trouver l'hyperplan qui sépare les classes avec la **marge maximale**.
- Les points les plus proches de l'hyperplan sont appelés **support vectors**.
- Peut gérer des classes non linéairement séparables grâce à des **kernels** (linéaire, polynomial, RBF, sigmoïde).

### Avantages
- Efficace pour des jeux de données avec **beaucoup de variables**.
- Gère bien les problèmes linéaires et non linéaires via les kernels.
- Robuste aux points aberrants si bien paramétré.

### Inconvénients
- Paramétrage délicat : choix du kernel, valeur de `C`, paramètre `gamma`.
- Peu interprétable : difficile de visualiser l'impact de chaque variable.
- Coût computationnel élevé pour de très grands jeux de données.

#####Hyperparamètres importants
- `C` : régularisation, contrôle le compromis entre marge large et erreurs sur l'apprentissage.
- `kernel` : type de fonction noyau (`linear`, `poly`, `rbf`, `sigmoid`).
- `gamma` : influence des points proches sur la séparation (pour `rbf` et `poly`).
- `degree` : degré du polynôme si kernel = `poly`.

### Implémentation Python

```python
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, f1_score, roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler

# Normalisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Définition du modèle
svm_model = SVC(probability=True, random_state=42)

# Paramètres pour Grid Search
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 0.01, 0.1]
}

# Grid Search avec f1_score pour la classe positive
grid_svm = GridSearchCV(svm_model, param_grid, scoring='f1', cv=5, verbose=2)
grid_svm.fit(X_train_scaled, y_train)

# Meilleur modèle
best_svm = grid_svm.best_estimator_
print("Meilleurs paramètres :", grid_svm.best_params_)

# Prédictions
y_pred_svm = best_svm.predict(X_test_scaled)
y_pred_proba_svm = best_svm.predict_proba(X_test_scaled)[:,1]

# Évaluation
print(classification_report(y_test, y_pred_svm))

# f1_score macro et weighted
print('f1_score macro :', f1_score(y_test, y_pred_svm, average='macro'))
print('f1_score weighted :', f1_score(y_test, y_pred_svm, average='weighted'))
```

### Notes pédagogiques
- Toujours **normaliser** avant SVM.
- Grid Search permet de trouver la **meilleure combinaison de paramètres**.
- `f1_score` pour la classe positive est critique si les classes sont déséquilibrées.
- La courbe ROC permet de comparer facilement ce modèle avec des arbres de décision, KNN ou régression logistique.
- SVM est **sensible aux outliers**, donc parfois utile de faire du preprocessing pour limiter leur impact.


## Random Forest

### Introduction à Random Forest
Random Forest est un algorithme de **machine learning supervisé**, fondé sur un ensemble d’arbres de décision.  
Il sert à la **classification** comme à la **régression**, mais ici nous nous concentrons sur la classification.

Le principe est simple :
On construit plusieurs arbres différents, puis on combine leurs prédictions.

<p align="center">
  <img src="https://substackcdn.com/image/fetch/$s_!bLkP!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F972e598f-8afe-4e6a-91d6-4799fba0a55f_2224x1053.png" alt="Source de l'image" width="600"/>
</p>


### Principe du Bagging (Bootstrap Aggregating)

Le **bagging** est la technique fondamentale derrière Random Forest.

#### Fonctionnement :
1. On crée plusieurs échantillons par **bootstrap** (tirage aléatoire *avec remise*).  
2. Pour chaque échantillon, on entraîne un arbre indépendant.  
3. On combine les prédictions :  
   - Vote majoritaire pour la classification  
   - Moyenne pour la régression  

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/c8/Ensemble_Bagging.svg/1200px-Ensemble_Bagging.svg.png" alt="Source de l'image" width="600"/>
</p>

#### Avantages du bagging :
- Réduction de la variance  
- Moins sensible au surapprentissage qu'un arbre seul  
- Très robuste aux données bruitées  

### Hyperparamètres

| Paramètre | Description |
|----------|-------------|
| **n_estimators** | Nombre d'arbres dans la forêt |
| **max_depth** | Profondeur maximale de chaque arbre |
| **min_samples_split** | Nb minimum d'observations pour diviser un nœud |
| **min_samples_leaf** | Nb minimum d’observations dans une feuille |
| **max_features** | Nombre de variables utilisées à chaque division |

### Importance des variables

Random Forest fournit une estimation de la contribution de chaque variable.

### Implémentation


```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, classification_report, roc_curve, auc
import pandas as pd
import matplotlib.pyplot as plt

# Modèle initial
rf = RandomForestClassifier(random_state=42)

# Grille des hyperparamètres
param_grid = {
    'n_estimators': [100, 300],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 10],
    'min_samples_leaf': [1, 5],
    'max_features': ['sqrt', 'log2']
}

# Grid Search optimisant le F1 macro
grid_search_rf = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    scoring='f1_macro',
    cv=5,
    n_jobs=-1
)

grid_search_rf.fit(X_train, y_train)

print("Meilleurs paramètres :", grid_search_rf.best_params_)

# Extraction du meilleur modèle
best_rf = grid_search_rf.best_estimator_
```


### Évaluation du modèle

```python
y_pred_rf = best_rf.predict(X_test)

print(classification_report(y_test, y_pred_rf))
```


### Feature Importance

```python
importances = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': best_rf.feature_importances_
}).sort_values(by='Importance', ascending=False)

print(importances)

plt.figure(figsize=(10,6))
plt.bar(importances['Feature'], importances['Importance'])
plt.xticks(rotation=45)
plt.title("Feature Importances — Random Forest")
plt.show()
```

### Conclusion

- Random Forest utilise le **bagging**, ce qui le rend très robuste.  
- Il réduit largement le surapprentissage des arbres de décision.  
- Il fournit naturellement une mesure de l’importance des variables.  


## Gradient Boosting

### Comprendre le Boosting

Le boosting consiste à entraîner plusieurs arbres **les uns après les autres**, chaque nouvel arbre cherchant à **corriger les erreurs** des précédents.

À chaque itération :

- Le modèle précédent commet des erreurs.
- On ajuste un nouvel arbre court (souvent profondeur 1 à 3) pour corriger ces erreurs.
- On ajoute cet arbre au modèle final.

Le modèle apprend donc **progressivement**, arbre après arbre.
Chaque nouvel arbre est entraîné sur le gradient de la **fonction de perte**, d’où le nom **Gradient Boosting**.

### La fonction de perte (loss function)

La **fonction de perte** est une mesure numérique qui quantifie **à quel point le modèle se trompe** sur les données d’entraînement.

Elle calcule l’écart entre :  

- les **valeurs prédites par le modèle**  
- et les **valeurs réelles** de la variable cible.  

Plus la fonction de perte est grande, plus le modèle fait des erreurs. L’objectif de l’apprentissage est donc **de minimiser cette fonction de perte**.

Dans la **Classification binaire** on utilise la log loss (cross-entropy)  qui mesure combien la probabilité prédite pour la bonne classe est proche de 1.


<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/b5/Ensemble_Boosting.svg/1280px-Ensemble_Boosting.svg.png" alt="Source de l'image" width="600"/>
</p>


### Boosting vs Bagging

| Méthode               | Random Forest (bagging) | Gradient Boosting (boosting) |
|-----------------------|------------------------|-----------------------------|
| Construction          | arbres en parallèle    | arbres en séquence          |
| Objectif              | réduire la variance    | réduire le biais            |
| Arbres                | profonds               | petits                      |
| Surapprentissage      | limité                 | plus élevé si mal réglé     |
| Vitesse               | rapide                 | plus lent                   |


### Pourquoi XGBoost ?

XGBoost est une version optimisée et ultra-performante du gradient boosting classique.

Ses forces :
- Très rapide (parallélisation, optimisations C++).  
- Très précis : souvent gagnant en compétition (Kaggle).  
- Régularisation L1 & L2 intégrée pour éviter l’overfitting.  
- Gestion native des valeurs manquantes.  
- Possibilité d’utiliser **early stopping**.  
- Très nombreux hyperparamètres pour ajuster la performance.

### Hyperparamètres

Voici les plus importants, expliqués de manière simple :

- **n_estimators**
  - Nombre d’arbres dans le modèle
  - Plus il y a d’arbres → modèle plus puissant
  - Trop d’arbres → risque d’overfitting

- **learning_rate**
  - Taux d’apprentissage (clé du boosting)
  - Valeur faible (0.01 à 0.1) → apprentissage lent mais précis
  - Valeur élevée → risque de surapprentissage

- **max_depth**
  - Profondeur maximale des arbres
  - Plus profond = modèle plus complexe
  - Trop profond = risque d’overfitting

- **subsample**
  - Fraction des échantillons utilisés pour chaque arbre (0.5 à 1)
  - Aide à éviter le surapprentissage
  - Introduit de la diversité entre les arbres

- **colsample_bytree**
  - Fraction des colonnes utilisées pour chaque arbre
  - Rend les arbres plus variés
  - Diminue le surapprentissage

- **gamma**
  - Gain minimal requis pour diviser un nœud
  - Valeur élevée → arbres plus simples, plus réguliers

- **reg_alpha (L1)**
  - Régularisation L1
  - Réduit les splits inutiles
  - Aide à simplifier le modèle

- **reg_lambda (L2)**
  - Régularisation L2
  - Stabilise le modèle et réduit le surapprentissage


### Implémentation Complète de XGBoost en Python

Voici un exemple extrêmement classique d’entraînement avec une **GridSearch** permettant d’optimiser les paramètres.

```python
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import pandas as pd
import matplotlib.pyplot as plt
```

### Modèle initial

```python
xgb = XGBClassifier(
random_state=42,
eval_metric='logloss' # Pour éviter les warnings inutiles
)
```

### Grille de paramètres

```python
param_grid = {
'n_estimators': [100, 300],
'max_depth': [3, 6],
'learning_rate': [0.01, 0.1],
'subsample': [0.7, 1.0],
'colsample_bytree': [0.7, 1.0],
'gamma': [0, 1]
}
```

### Grid Search

```python
grid_search_xgb = GridSearchCV(
estimator=xgb,
param_grid=param_grid,
scoring='f1_macro',
cv=5,
n_jobs=-1
)

grid_search_xgb.fit(X_train, y_train)

print("Meilleurs hyperparamètres :", grid_search_xgb.best_params_)
```

### Modèle final

```python
best_xgb = grid_search_xgb.best_estimator_
```

### Évaluation du Modèle

```python
y_pred_xgb = best_xgb.predict(X_test)
print(classification_report(y_test, y_pred_xgb))
```

### Importance des Variables

```python
importances = pd.DataFrame({
'Feature': X_train.columns,
'Importance': best_xgb.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10,6))
plt.bar(importances['Feature'], importances['Importance'])
plt.xticks(rotation=45)
plt.title("Feature Importances — XGBoost")
plt.show()
```

L’importance des variables permet de comprendre quelles colonnes influencent le plus la prédiction.


### Avantages et Inconvénients

#### Avantages
- Performances exceptionnelles  
- Grande flexibilité et nombreuses optimisations  
- Régularisation intégrée  
- Très bonne gestion des données manquantes  
- Rarement dépassé sur les données tabulaires  

#### Inconvénients
- Beaucoup d’hyperparamètres (peut perdre un débutant)  
- Plus lent qu’un Random Forest si mauvaise configuration  
- Peut surapprendre si trop de profondeur / trop d’arbres  

### Conseils pratiques

- Toujours commencer avec **learning_rate = 0.1**  
- Si le modèle surapprend :  
  - diminuer **max_depth**  
  - augmenter **gamma**  
  - baisser **n_estimators**  
- Utiliser **early stopping** pour éviter le surapprentissage  
- Tester plusieurs valeurs de **subsample** et **colsample_bytree**  


### Conclusion

XGBoost est l’un des algorithmes les plus puissants pour les données tabulaires.  
Grâce à ses améliorations internes, sa capacité de régularisation et sa rapidité, il offre des performances souvent supérieures au Random Forest.

Ce cours donne toutes les bases nécessaires pour :
- comprendre son fonctionnement  
- ajuster ses hyperparamètres  
- l’entraîner correctement en Python  
- interpréter ses résultats  

## Récapitulatif


| Méthode                  | Principe rapide                                                                 | Avantages principaux                                                                                     | Inconvénients / Limites                                                                                   | Type de données idéal          | Temps / Complexité                | Interprétation                |
|---------------------------|-------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------|--------------------------------|----------------------------------|-------------------------------|
| Decision Tree (Arbre)     | Diviser récursivement les données selon les variables pour créer des règles    | Facile à interpréter, visualisable, pas besoin de normalisation                                         | Tendance au surapprentissage, sensible aux variations des données                                        | Numérique ou catégoriel       | Faible / O(n log n)              | Très bonne                    |
| Naive Bayes Gaussian      | Probabilités conditionnelles, suppose indépendance des variables              | Rapide, simple, fonctionne bien sur petits jeux de données, baseline                                    | Hypothèse d’indépendance souvent fausse, moins performant sur données complexes                           | Numérique (pour Gaussian)    | Très rapide / faible             | Moyenne                       |
| KNN                       | Classe attribuée selon les k plus proches voisins                              | Simple, non paramétrique, pas besoin d’apprentissage réel                                                | Lent sur grands datasets, sensible au choix de k et à l’échelle des variables                              | Numérique                     | Élevé à prédiction / faible entraînement | Faible                       |
| Régression Logistique     | Modèle linéaire sur logit (log-odds)                                           | Interprétable, probabilités directes, rapide                                                            | Ne capture pas les relations non linéaires sans transformation                                           | Numérique / catégoriel codé  | Faible                           | Bonne                         |
| SVM                       | Sépare les classes en maximisant la marge, possibilité de kernel non linéaire  | Efficace pour petits/moyens datasets, flexible avec kernels                                             | Paramétrage délicat (C, gamma, kernel), peu interprétable, lent sur grands datasets                        | Numérique                     | Moyen à élevé                     | Moyenne à faible              |
| Random Forest             | Ensemble d’arbres (bagging), vote majoritaire                                   | Réduction du surapprentissage, robuste, importance variables, pas besoin de normalisation               | Moins interprétable qu’un arbre simple, plus lent                                                         | Numérique ou catégoriel       | Moyen à élevé                     | Moyenne                       |
| Gradient Boosting / XGBoost | Addition séquentielle d’arbres corrigeant erreurs précédentes                 | Très performant, gère données déséquilibrées, importance variables                                      | Sensible aux hyperparamètres, risque surapprentissage, plus lent                                          | Numérique ou catégoriel       | Élevé                             | Moyenne                       |


# Exercice : Prédiction des étiquettes DPE (classification multi-classes)

Dans cet exercice, vous allez appliquer toutes les méthodes de **classification** vues précédemment pour prédire les **étiquettes DPE** (A, B, C, D, E, F, G) des logements. L’objectif est de comparer les performances des modèles et d’identifier les variables les plus importantes tout en évitant le **data leakage**.
