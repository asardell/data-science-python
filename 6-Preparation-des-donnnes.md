# Chapitre 6 : Préparation et nettoyage des données

La qualité des données est cruciale pour obtenir des modèles fiables et performants en Machine Learning.  
Une mauvaise préparation peut entraîner des résultats biaisés, des erreurs d’apprentissage ou des interprétations fausses.  

Ce chapitre présente les concepts et étapes clés pour préparer les données avant de les utiliser pour :

- Apprentissage supervisé (régression, classification)  
- Apprentissage non supervisé (clustering)  
- Analyse exploratoire et réduction de dimension (PCA, etc.)

:bulb: Pourquoi préparer les données ?

- Les données brutes contiennent souvent des erreurs, valeurs manquantes ou aberrantes.  
- Les algorithmes de ML ont besoin de formats et échelles cohérents.  
- Une bonne préparation permet de **réduire le bruit**, **améliorer la performance** et **éviter le data leakage**.

- [Chapitre 6 : Préparation et nettoyage des données pour le Machine Learning (avec exemples)](#chapitre-6--préparation-et-nettoyage-des-données-pour-le-machine-learning-avec-exemples)
  - [Gestion des valeurs manquantes](#gestion-des-valeurs-manquantes)
  - [Détection et suppression des doublons](#détection-et-suppression-des-doublons)
  - [Détection et traitement des outliers](#détection-et-traitement-des-outliers)
    - [Exercice : Gestion des outliers avec une fonction](#exercice--gestion-des-outliers-avec-une-fonction)
  - [Normalisation et standardisation](#normalisation-et-standardisation)
  - [Encodage des variables catégorielles](#encodage-des-variables-catégorielles)
    - [Label encoding : attribuer un entier à chaque catégorie.](#label-encoding--attribuer-un-entier-à-chaque-catégorie)
    - [One Hot Encoding : (codage disjonctif) : créer une colonne par catégorie.](#one-hot-encoding--codage-disjonctif--créer-une-colonne-par-catégorie)
  - [Feature Engineering](#feature-engineering)
  - [Détection des corrélations et réduction de dimension](#détection-des-corrélations-et-réduction-de-dimension)


## Gestion des valeurs manquantes

- Les valeurs manquantes (NaN, null, vide) peuvent biaiser l’apprentissage.  
- Stratégies classiques :
  - **Suppression** : retirer lignes ou colonnes avec trop de valeurs manquantes.
  - **Imputation statistique** : remplacer par la moyenne, médiane ou mode.
  - **Imputation avancée** : utiliser un modèle pour prédire les valeurs manquantes.  
- La stratégie dépend du pourcentage de valeurs manquantes et de leur importance.

Supposons notre dataset ADEME contient une colonne 'surface_habitable_logement' avec des valeurs manquantes

```python
# Théorique : si 5% de données manquantes → on peut imputer à la médiane
median_surface = df['surface_habitable_logement'].median()
df['surface_habitable_logement'].fillna(median_surface, inplace=True)

# Vérification
missing_after = df['surface_habitable_logement'].isna().sum()
print("Valeurs manquantes restantes :", missing_after)
```


## Détection et suppression des doublons

- Les doublons peuvent fausser les statistiques et modèles.  
- Identifier par un identifiant unique ou par une combinaison de colonnes pertinentes.  
- Décider de supprimer ou fusionner selon le contexte.

```python
# On vérifie les doublons sur la clé 'numero_dpe'
duplicates = df.duplicated(subset=['numero_dpe']).sum()
print("Nombre de doublons :", duplicates)

# Suppression des doublons
df.drop_duplicates(subset=['numero_dpe'], inplace=True)
```


## Détection et traitement des outliers

- Outliers : valeurs très éloignées de la distribution normale.  
- Méthodes :
  - **Z-score** : valeurs > 3 écarts-types de la moyenne.
  - **IQR (Interquartile Range)** : valeurs en dehors de [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
  - **Centiles** : valeurs au-delà du 1er et 99e centile (ou 5e / 95e centile).  
- Stratégies : 
  -  supprimer les lignes contenant ces valeurs → perte d’information.
  -  transformer la variable (log, sqrt, etc.) → réduit l’impact des valeurs extrêmes.
  -  caper (ou "winsorizer") → remplacer les valeurs extrêmes par un seuil défini (souvent les centiles, comme le 1er et 99e centile).

```python
# Détection des outliers par centiles pour 'besoin_chauffage'
q_low = df['besoin_chauffage'].quantile(0.01)
q_high = df['besoin_chauffage'].quantile(0.99)

outliers = df[(df['besoin_chauffage'] < q_low) | (df['besoin_chauffage'] > q_high)]
print("Nombre d’outliers détectés :", len(outliers))

# On peut les supprimer ou les caper
df['besoin_chauffage'] = df['besoin_chauffage'].clip(lower=q_low, upper=q_high)
```

### Exercice : Gestion des outliers avec une fonction

1) Objectif
- Créer une fonction `manage_outlier` qui détecte et cappe les outliers sur une colonne.  
- Appliquer cette fonction à toutes les colonnes numériques du dataset.

2) Théorie rapide
- On utilise les **centiles** pour limiter les valeurs extrêmes.  
- Exemple : caper toutes les valeurs en dessous du 1er centile et au-dessus du 99e centile.

3) Étapes
- Créer une fonction prenant en entrée :
   - un `Series` pandas
   - un centile bas (`low=0.01`) et haut (`high=0.99`)
- Calculer les valeurs de centiles.
- Caper les valeurs en dehors de ces centiles.
- Retourner la série modifiée.

```python
def manage_outlier(col, low=0.01, high=0.99):
    """
    Caper les valeurs extrêmes d'une colonne pandas selon les centiles.
    """
    q_low = col.quantile(low)
    q_high = col.quantile(high)
    return col.clip(lower=q_low, upper=q_high)

# Application sur toutes les colonnes numériques
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
for c in num_cols:
    df[c] = manage_outlier(df[c])

print("Outliers capés pour toutes les colonnes numériques.")
```

## Normalisation et standardisation

- Certains modèles (SVM, kNN, réseaux de neurones) sont sensibles à l’échelle des variables.  
- **Normalisation** : ramène les valeurs entre 0 et 1 (ou [-1,1])  
- **Standardisation** : transforme les valeurs pour avoir moyenne 0 et écart-type 1.  
- Permet de rendre toutes les variables comparables et d’éviter que certaines dominent le modèle.

```python
# Standardisation du besoin en chauffage
mean_besoin = df['besoin_chauffage'].mean()
std_besoin = df['besoin_chauffage'].std()
df['besoin_chauffage_std'] = (df['besoin_chauffage'] - mean_besoin) / std_besoin

print("Moyenne standardisée :", df['besoin_chauffage_std'].mean())
print("Écart type standardisé :", df['besoin_chauffage_std'].std())
```


## Encodage des variables catégorielles

- Les algorithmes ML utilisent des nombres, pas des chaînes.  
Méthodes courantes :

### Label encoding : attribuer un entier à chaque catégorie.

Le **Label Encoding** transforme une variable catégorielle en une variable numérique en **attribuant un entier unique à chaque catégorie**.

Exemple simple :  
- "A" → 0  
- "B" → 1  
- "C" → 2  

C’est utile pour les modèles de type arbres (Random Forest, Gradient Boosting) car ils ne sont pas sensibles à l’ordre numérique artificiel.

:warning:Limite : Cette technique introduit un **ordre implicite** entre les catégories.  
Pour les modèles sensibles aux distances (kNN, SVM, Régressions linéaires), privilégier le **One Hot Encoding**.

### One Hot Encoding : (codage disjonctif) : créer une colonne par catégorie.  

Le **One-hot encoding** (codage disjonctif) transforme chaque catégorie en une colonne.  

:warning: éviter la multicolinéarité ("dummy variable trap") pour certains modèles linéaires.

```python
# Encodage one-hot pour 'etiquette_dpe'
df_encoded = pd.get_dummies(df, columns=['etiquette_dpe'], prefix='dpe')

print("Colonnes après encodage :", df_encoded.columns)
```


## Feature Engineering

- Créer de nouvelles variables à partir des existantes pour enrichir le modèle.  
- Exemples :
  - Ratios (conso_chauffage / surface)
  - Interaction entre variables
  - Transformations (log, sqrt, polynomial)
- Risques :
  - **Data leakage** : utiliser une information qui serait connue uniquement après la prédiction

```python
# Création d’un ratio consommation / surface
df['conso_par_m2'] = df['besoin_chauffage'] / df['surface_habitable_logement']

print(df[['besoin_chauffage', 'surface_habitable_logement', 'conso_par_m2']].head())
```


## Détection des corrélations et réduction de dimension

- Identifier les variables fortement corrélées pour :
  - Réduire la redondance
  - Prévenir le multicolinéarité
- PCA ou autres méthodes de réduction permettent de transformer les variables corrélées en un ensemble non corrélé.

```python
# Corrélation entre variables numériques
corr_matrix = df[['besoin_chauffage', 'conso_chauffage_ef', 'surface_habitable_logement']].corr()
print(corr_matrix)
```

:bulb: Si certaines variables sont trop corrélées, on pourrait envisager de combiner ou supprimer.


