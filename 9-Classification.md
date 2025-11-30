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

## Introduction à la classification

La **classification** est une sous-catégorie de l’apprentissage supervisé. Elle consiste à prédire une **variable cible catégorielle** à partir de variables explicatives. L'objectif est d’attribuer à chaque observation une **classe** parmi un ensemble de classes possibles.

**Exemple : projet DPE**

- Variables explicatives : surface habitable, consommation chauffage, type de logement, année de construction, etc.
- Variable cible : `passoire énergétique` (Oui / Non) ou `étiquette DPE` (A, B, C, D, E, F, G)

---

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

---

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

#### Support Vector Machine (SVM)
- Principe : séparer les classes en maximisant la **marge** entre elles.
- Avantages : efficace pour données non linéaires via kernels.
- Inconvénients : paramétrage délicat (kernel, C, gamma), peu interprétable.

#### Gradient Boosting / XGBoost

- Principe : combinaison d’arbres de décision construits séquentiellement, chaque arbre corrigeant les erreurs du précédent.
- Avantages : très performant, peut gérer données non linéaires et interactions complexes, options de régularisation.
- Inconvénients : paramétrage complexe, sensible au surapprentissage si mal réglé, moins interprétable.

---

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

#### Courbes ROC et AUC
- Permettent de visualiser le compromis entre **taux de vrais positifs** et **taux de faux positifs**.  
- L’AUC mesure la capacité globale du modèle à distinguer les classes (1 = parfait, 0.5 = hasard).

---

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

### Préparation de l’échantillon

- Séparer les variables explicatives `X` de la variable cible `y`.
- Découper l’échantillon en **train (70%)** et **test (30%)**, en stratifiant selon la variable cible pour conserver les proportions de classes.

Exemple de tableau de proportions des classes :

| Classe | Train | Test |
|--------|-------|------|
| 0      | 0.65  | 0.65 |
| 1      | 0.35  | 0.35 |

```python
# Exemple Python pour préparation des données
from sklearn.model_selection import train_test_split

X = df[df.columns.difference(['passoire_energetique'])]
y = df['passoire_energetique']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
```

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