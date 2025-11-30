# Chapitre 8 : Apprentissage Non Supervisé – Clustering

# Introduction à l’apprentissage non supervisé

L’**apprentissage non supervisé** consiste à analyser des données **sans variable cible** pour en découvrir la structure. Contrairement à l’apprentissage supervisé, on n’a pas d’étiquette à prédire.

Les objectifs principaux sont :

- **Détection de groupes (clusters)** dans les données  
- **Réduction de dimension et exploration**  
- **Détection d’anomalies** ou outliers  

Nous nous concentrons ici sur le **clustering**, pour segmenter les logements ADEME selon leurs caractéristiques énergétiques ou de consommation.

- [Chapitre 8 : Apprentissage Non Supervisé – Clustering](#chapitre-8--apprentissage-non-supervisé--clustering)
- [Introduction à l’apprentissage non supervisé](#introduction-à-lapprentissage-non-supervisé)
- [K-Means](#k-means)
    - [Principe](#principe)
    - [Configuration et paramètres](#configuration-et-paramètres)
    - [Interprétation](#interprétation)
    - [Exemple pédagogique](#exemple-pédagogique)
    - [Exemple sur le jeu de données ADEME](#exemple-sur-le-jeu-de-données-ademe)
- [CAH (Clustering Ascendant Hiérarchique)](#cah-clustering-ascendant-hiérarchique)
    - [Principe](#principe-1)
    - [Configuration](#configuration)
    - [Interprétation](#interprétation-1)
    - [Exemple pédagogique](#exemple-pédagogique-1)
    - [Exemple sur le jeu de données ADEME](#exemple-sur-le-jeu-de-données-ademe-1)
- [DBSCAN](#dbscan)
    - [Principe](#principe-2)
    - [Paramètres](#paramètres)
    - [Interprétation](#interprétation-2)
    - [Exemple pédagogique](#exemple-pédagogique-2)
    - [Exemple sur le jeu de données ADEME](#exemple-sur-le-jeu-de-données-ademe-2)
  - [4. Évaluation des clusters](#4-évaluation-des-clusters)
    - [Calcul du coefficient de silhouette](#calcul-du-coefficient-de-silhouette)
  - [5. Synthèse des méthodes](#5-synthèse-des-méthodes)
- [Exercice : Clustering combiné sur les logements DPE](#exercice--clustering-combiné-sur-les-logements-dpe)
  - [Objectif](#objectif)
  - [Données à utiliser](#données-à-utiliser)
  - [Étapes de l’exercice](#étapes-de-lexercice)
    - [Préparation des données](#préparation-des-données)
    - [Transformation des variables qualitatives](#transformation-des-variables-qualitatives)
    - [Clustering](#clustering)
    - [Analyse des clusters](#analyse-des-clusters)


# K-Means

### Principe

- Partitionne les données en **K clusters** prédéfinis.  
- Chaque point appartient au cluster dont le **centre (centroïde)** est le plus proche.  
- Objectif : **minimiser la variance intra-cluster** (somme des distances au carré par rapport au centroïde).

### Configuration et paramètres

- `n_clusters` : nombre de clusters K  
- `init` : méthode d’initialisation des centroïdes (`k-means++` recommandé)  
- `n_init` : nombre de réinitialisations pour choisir la meilleure solution  
- `max_iter` : nombre maximum d’itérations  
- `random_state` : pour reproductibilité  

### Interprétation

- **Centroïdes** : valeur moyenne des variables dans chaque cluster  
- **Labels** : cluster assigné à chaque observation  
- **Inertia** : somme des distances au carré des points à leur centroïde  
- **Silhouette score** : mesure de cohérence des clusters (proche de 1 = bon cluster, proche de 0 = chevauchement, <0 = erreur)

### Exemple pédagogique

```python
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Création d'un dataset factice
X, _ = make_blobs(n_samples=200, centers=3, n_features=2, random_state=42)

# KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X)
centroids = kmeans.cluster_centers_

# Visualisation
plt.scatter(X[:,0], X[:,1], c=labels, cmap='viridis')
plt.scatter(centroids[:,0], centroids[:,1], color='red', marker='X', s=200)
plt.title("K-Means - exemple pédagogique")
plt.show()
```

### Exemple sur le jeu de données ADEME

```python
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import pandas as pd

# Exemple : colonnes numériques
cols = ['surface_habitable_logement', 'besoin_chauffage', 'conso_chauffage_ef']
df_ade = df[cols].dropna()  # nettoyage minimal
X = StandardScaler().fit_transform(df_ade)

# KMeans
kmeans = KMeans(n_clusters=4, random_state=42)
labels = kmeans.fit_predict(X)
df_ade['cluster'] = labels

# Évaluation
score = silhouette_score(X, labels)
print("Silhouette score :", score)

# Analyse des centroïdes
centroids = pd.DataFrame(kmeans.cluster_centers_, columns=cols)
print(centroids)
```

> Interprétation : chaque cluster correspond à un groupe de logements avec caractéristiques similaires. Le centroïde montre la **valeur moyenne des variables dans le cluster**, utile pour décrire le profil type.


# CAH (Clustering Ascendant Hiérarchique)

### Principe

- Construire une **hiérarchie de clusters** sans pré-définir le nombre de clusters.  
- On commence avec **chaque observation comme cluster**, puis on **fusionne progressivement** les clusters les plus proches.  
- Résultat : **dendrogramme** montrant la fusion des clusters.

### Configuration

- `affinity` : métrique de distance (`euclidean` par défaut)  
- `linkage` : méthode de fusion (`ward`, `complete`, `average`)  
  - `ward` → minimise variance intra-cluster  
  - `complete` → distance max entre points de clusters  
  - `average` → distance moyenne  
- Pas besoin de spécifier K initialement, on coupe le dendrogramme à la hauteur souhaitée.

### Interprétation

- **Dendrogramme** : visualiser comment les observations se regroupent  
- **Coupe du dendrogramme** : choix du nombre optimal de clusters  
- **Silhouette score** possible pour évaluer la qualité des clusters

### Exemple pédagogique

```python
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt

# Dataset factice
X, _ = make_blobs(n_samples=50, centers=3, n_features=2, random_state=42)

# CAH
Z = linkage(X, method='ward')
plt.figure(figsize=(8,5))
dendrogram(Z)
plt.title("Dendrogramme - CAH")
plt.show()

# Couper à 3 clusters
labels = fcluster(Z, 3, criterion='maxclust')
```

### Exemple sur le jeu de données ADEME

```python
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

X = StandardScaler().fit_transform(df_ade[cols])
Z = linkage(X, method='ward')

# Dendrogramme
plt.figure(figsize=(10,6))
dendrogram(Z, truncate_mode='level', p=5)
plt.title("Dendrogramme CAH - ADEME")
plt.show()

# Attribution des clusters
df_ade['cluster_cah'] = fcluster(Z, 4, criterion='maxclust')
```


# DBSCAN

### Principe

- **Density-Based Spatial Clustering of Applications with Noise**  
- Forme des clusters basés sur la **densité locale des points**.  
- Avantages : détection d’**outliers** automatiquement, pas besoin de fixer le nombre de clusters.

### Paramètres

- `eps` : rayon de voisinage pour former un cluster  
- `min_samples` : nombre minimum de points dans ce rayon pour créer un cluster  
- Points isolés → **label = -1** (outliers)

### Interprétation

- Points avec **label ≥0** : appartiennent à un cluster  
- Points avec **label = -1** : considérés comme anomalies ou bruit  
- Visualisation possible pour vérifier densité et structure

### Exemple pédagogique

```python
from sklearn.cluster import DBSCAN

X, _ = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)

dbscan = DBSCAN(eps=1.0, min_samples=5)
labels = dbscan.fit_predict(X)

plt.scatter(X[:,0], X[:,1], c=labels, cmap='plasma')
plt.title("DBSCAN - exemple pédagogique")
plt.show()
```

### Exemple sur le jeu de données ADEME

```python
dbscan = DBSCAN(eps=1.5, min_samples=10)
labels = dbscan.fit_predict(X)
df_ade['cluster_dbscan'] = labels

# Analyse des outliers
outliers = df_ade[df_ade['cluster_dbscan']==-1]
print("Nombre d'outliers :", len(outliers))
```


## 4. Évaluation des clusters

Même sans variable cible, on peut évaluer la qualité des clusters :

- **Silhouette score** : mesure la cohérence intra-cluster et séparation inter-cluster  
  - Valeur proche de 1 → clusters bien séparés  
  - Valeur proche de 0 → chevauchement  
  - Valeur négative → mauvais clustering

- **Inertia (pour K-Means)** : somme des distances au carré aux centroïdes  

- **Observation des centroïdes / moyenne des variables** pour interpréter chaque cluster

### Calcul du coefficient de silhouette

```python
from sklearn.metrics import silhouette_score

# KMeans
silhouette_kmeans = silhouette_score(X, df_ade['cluster'])
print("Silhouette KMeans :", silhouette_kmeans)

# CAH
silhouette_cah = silhouette_score(X, df_ade['cluster_cah'])
print("Silhouette CAH :", silhouette_cah)

# DBSCAN (ignorer outliers pour le score)
mask = df_ade['cluster_dbscan'] != -1
silhouette_dbscan = silhouette_score(X[mask], df_ade['cluster_dbscan'][mask])
print("Silhouette DBSCAN :", silhouette_dbscan)
```


## 5. Synthèse des méthodes

| Méthode     | Nombre clusters | Détection outliers | Type de clusters | Interprétation |
|------------|----------------|------------------|-----------------|---------------|
| K-Means    | fixe           | Non              | sphérique        | centroïdes, inertia |
| CAH        | flexible       | Non              | hiérarchique     | dendrogramme, silhouette |
| DBSCAN     | automatique    | Oui              | densité          | labels, points isolés |


💡 **Conseils pratiques** :

- Toujours **standardiser les variables** avant clustering  
- Explorer **plusieurs méthodes et paramètres** pour comparer  
- Visualiser les clusters en 2D ou 3D pour valider  
- K-Means et CAH → mieux pour clusters globaux  
- DBSCAN → mieux pour détecter des anomalies ou clusters irréguliers


# Exercice : Clustering combiné sur les logements DPE

## Objectif
- Appliquer la **MCA** pour transformer les variables qualitatives en variables numériques continues.  
- Combiner les composantes MCA avec des variables numériques standardisées.  
- Appliquer un **algorithme de clustering** sur les données mixtes.  
- Interpréter les clusters obtenus pour identifier des groupes de logements similaires.


## Données à utiliser
- Variables qualitatives :  
  - `etiquette_dpe`  
  - `type_batiment`  
- Variables numériques :  
  - `surface_habitable_logement`  
  - `besoin_chauffage`  
  - `conso_chauffage_ef`  
  - `conso_ecs_ef`  


## Étapes de l’exercice

### Préparation des données
1. Sélectionner les colonnes qualitatives et numériques.  
2. Gérer les valeurs manquantes si nécessaire (imputation ou suppression).  
3. Standardiser les variables numériques pour qu’elles soient comparables.

### Transformation des variables qualitatives
1. Appliquer la **MCA** sur les variables qualitatives.  
2. Extraire un nombre choisi de composantes principales (ex. 2 à 5).  
3. Ajouter ces composantes au jeu de données standardisé.

### Clustering
1. Choisir un algorithme de clustering : **K-Means**, **CAH**, ou **DBSCAN**.  
2. Appliquer le clustering sur le jeu de données mixte (composantes MCA + variables numériques).  
3. Tester différents paramètres pour observer leur impact (ex. nombre de clusters k pour K-Means).

### Analyse des clusters
1. Visualiser les clusters sur un scatter plot en utilisant les 2 premières composantes MCA ou les 2 variables numériques les plus significatives.  
2. Comparer la distribution des clusters par rapport aux variables qualitatives (`etiquette_dpe`, `type_batiment`).  
3. Discuter des insights obtenus : quels types de logements se regroupent ensemble ? Y a-t-il des comportements énergétiques similaires ?
