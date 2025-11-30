# Chapitre 5 : Visualisation de données

La visualisation est essentielle en Data Science pour comprendre la distribution des variables, détecter des anomalies et explorer les relations entre variables.  
Dans ce chapitre, nous allons couvrir :

- Les graphes univariés : comprendre une seule variable
- Les graphes bivariés : explorer les relations entre deux variables
- Les graphes multivariés : explorer des interactions de plusieurs variables

Nous utiliserons trois bibliothèques : Matplotlib, Seaborn, et Plotly.

- [Chapitre 5 : Visualisation de données](#chapitre-5--visualisation-de-données)
  - [Introduction aux bibliothèques](#introduction-aux-bibliothèques)
    - [Matplotlib](#matplotlib)
    - [Seaborn](#seaborn)
    - [Plotly](#plotly)
  - [Graphiques univariés](#graphiques-univariés)
    - [Histogramme](#histogramme)
    - [Boxplot](#boxplot)
  - [Graphiques bivariés](#graphiques-bivariés)
    - [Nuage de points (scatter plot)](#nuage-de-points-scatter-plot)
    - [Diagramme en barres](#diagramme-en-barres)
    - [Boxplot bivarié](#boxplot-bivarié)
  - [Graphiques multivariés](#graphiques-multivariés)
    - [Heatmap de corrélation](#heatmap-de-corrélation)
    - [Pairplot (Seaborn)](#pairplot-seaborn)
  - [Aller plus vite avec `pandas_profiling`](#aller-plus-vite-avec-pandas_profiling)
    - [Exemple](#exemple)
  - [Cartographie avec Folium](#cartographie-avec-folium)



## Introduction aux bibliothèques

### Matplotlib
- Bibliothèque de base pour les graphiques en Python
- Permet un contrôle précis de tous les éléments du graphique : axes, titres, couleurs
- Exemples d’utilisation : plot, scatter, hist, bar

### Seaborn
- Basé sur Matplotlib, mais simplifie les graphiques statistiques
- Propose des thèmes par défaut et des fonctions pour histogrammes, boxplots, nuages de points, heatmaps
- Très utile pour des graphiques rapides et élégants

### Plotly
- Bibliothèque interactive pour créer des graphiques web interactifs
- Permet zoom, survol, filtrage dynamique
- Très pratique pour les dashboards et présentations


## Graphiques univariés

Permettent d’analyser la distribution d’une variable unique.

### Histogramme

- Montre la répartition des valeurs sur des intervalles
- Utile pour détecter des concentrations ou valeurs extrêmes

Matplotlib :

```python
import matplotlib.pyplot as plt

plt.hist(df['besoin_chauffage'], bins=20, color='skyblue', edgecolor='black')
plt.title("Histogramme du besoin en chauffage")
plt.xlabel("Besoin en chauffage (kWh)")
plt.ylabel("Nombre de logements")
plt.show()
```

Seaborn :

```python
import seaborn as sns

sns.histplot(df['besoin_chauffage'], bins=20, kde=True, color='skyblue')
plt.title("Histogramme du besoin en chauffage")
plt.show()
```

Plotly :

```python
import plotly.express as px

fig = px.histogram(df, x='besoin_chauffage', nbins=20, title="Histogramme du besoin en chauffage")
fig.show()
```

Explications :

- bins/nbins : nombre d’intervalles
- kde=True (Seaborn) : ajoute une courbe de densité

### Boxplot

- Permet de visualiser la médiane, quartiles et valeurs extrêmes (outliers)
- Utile pour détecter des distributions asymétriques ou valeurs aberrantes

Matplotlib :

```python
plt.boxplot(df['conso_chauffage_ef'], vert=True)
plt.title("Boxplot consommation chauffage EF")
plt.ylabel("kWh")
plt.show()
```

Seaborn :

```python
sns.boxplot(y=df['conso_chauffage_ef'], color='lightgreen')
plt.title("Boxplot consommation chauffage EF")
plt.show()
```

Plotly :

```python
fig = px.box(df, y='conso_chauffage_ef', title="Boxplot consommation chauffage EF")
fig.show()
```

Explications :

- La boîte montre le 1er et 3ème quartile
- La ligne horizontale est la médiane
- Les points au-delà des « moustaches » sont des outliers


## Graphiques bivariés

Permettent d’explorer la relation entre deux variables.

### Nuage de points (scatter plot)

Visualise la relation entre deux variables numériques.  
Peut aider à détecter corrélations ou tendances.

Matplotlib :

```python
plt.scatter(df['surface_habitable_logement'], df['besoin_chauffage'], color='orange')
plt.title("Surface vs besoin chauffage")
plt.xlabel("Surface habitable (m²)")
plt.ylabel("Besoin chauffage (kWh)")
plt.show()
```

Seaborn :

```python
sns.scatterplot(data=df, x='surface_habitable_logement', y='besoin_chauffage', hue='etiquette_dpe')
plt.title("Surface vs besoin chauffage par étiquette DPE")
plt.show()
```

Plotly :

```python
fig = px.scatter(df, x='surface_habitable_logement', y='besoin_chauffage', color='etiquette_dpe', title="Surface vs besoin chauffage")
fig.show()
```

Explications :

- hue / color : permet de distinguer une variable catégorielle
- Permet de détecter des tendances ou regroupements

### Diagramme en barres

Visualiser la répartition d’une variable catégorielle ou comparer des moyennes.

Matplotlib :

```python
df.groupby('etiquette_dpe')['besoin_chauffage'].mean().plot(kind='bar', color='lightblue')
plt.title("Besoin chauffage moyen par étiquette DPE")
plt.ylabel("kWh")
plt.show()
```

Seaborn :

```python
sns.barplot(x='etiquette_dpe', y='besoin_chauffage', data=df, palette='Blues_d')
plt.title("Besoin chauffage moyen par étiquette DPE")
plt.show()
```

Plotly :

```python
fig = px.bar(df.groupby('etiquette_dpe')['besoin_chauffage'].mean().reset_index(), x='etiquette_dpe', y='besoin_chauffage', title="Besoin chauffage moyen par DPE")
fig.show()
```

### Boxplot bivarié

Permet de comparer la distribution d’une variable numérique selon une variable catégorielle.

Matplotlib :

```python
df.boxplot(column='besoin_chauffage', by='etiquette_dpe', grid=False, color='purple')
plt.title("Besoin chauffage par étiquette DPE")
plt.suptitle("")
plt.ylabel("kWh")
plt.show()
```

Seaborn :

```python
sns.boxplot(x='etiquette_dpe', y='besoin_chauffage', data=df, palette='pastel')
plt.title("Besoin chauffage par étiquette DPE")
plt.show()
```

Plotly :

```python
fig = px.box(df, x='etiquette_dpe', y='besoin_chauffage', title="Besoin chauffage par étiquette DPE")
fig.show()
```


## Graphiques multivariés

Permettent d’explorer les interactions entre plusieurs variables.

### Heatmap de corrélation

Permet de visualiser les corrélations entre variables numériques.  
Valeurs proches de 1 ou -1 = forte corrélation positive ou négative.

Seaborn :

```python
corr = df[['besoin_chauffage', 'conso_chauffage_ef', 'conso_ecs_ef', 'surface_habitable_logement']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Heatmap des corrélations")
plt.show()
```

Plotly :

```python
import plotly.figure_factory as ff

corr = df[['besoin_chauffage', 'conso_chauffage_ef', 'conso_ecs_ef', 'surface_habitable_logement']].corr().values
labels = ['besoin_chauffage', 'conso_chauffage_ef', 'conso_ecs_ef', 'surface_habitable_logement']
fig = ff.create_annotated_heatmap(corr, x=labels, y=labels, colorscale='RdBu', showscale=True)
fig.show()
```

Explications :

- corr() : calcule la corrélation entre colonnes numériques
- annot=True : affiche les valeurs dans la heatmap
- Palette de couleurs : rouge = corrélation négative, bleu = positive

### Pairplot (Seaborn)

Affiche tous les nuages de points pour les paires de variables numériques.  
Utile pour visualiser rapidement les relations multiples.

```python
sns.pairplot(df[['besoin_chauffage', 'conso_chauffage_ef', 'surface_habitable_logement', 'etiquette_dpe']], hue='etiquette_dpe')
plt.show()
```


## Aller plus vite avec `pandas_profiling`

`pandas_profiling` permet de générer un rapport complet automatiquement sur un DataFrame, incluant :  
- résumé statistique des colonnes  
- valeurs manquantes  
- distribution des variables  
- corrélations  
- histogrammes et boxplots  

### Exemple

```python
from pandas_profiling import ProfileReport

profile = ProfileReport(df, title="Profiling ADEME", explorative=True)
profile.to_widgets()   # Affiche le rapport interactif dans Jupyter
# ou
profile.to_file("profiling_ademe.html")  # Exporte le rapport en HTML
```

Explications :  
- ProfileReport(df) : crée le rapport  
- title : titre du rapport  
- explorative=True : active les options interactives  
- to_widgets() : affichage dans Jupyter Notebook  
- to_file("…") : export HTML pour consultation extérieure  

**Astuce :** très pratique pour avoir une vue globale du jeu de données en quelques secondes, surtout pour les grands DataFrames.


## Cartographie avec Folium

Folium permet de créer des cartes interactives à partir de données GPS.  
Très utile pour visualiser la localisation des logements DPE ou tout autre point géographique.

1) Installation

```shell
pip install folium
```

2) Création d’une carte simple

```python
import folium

# Coordonnées centrales (ex : Paris)
centre = [48.8566, 2.3522]

# Créer la carte
m = folium.Map(location=centre, zoom_start=12, tiles='OpenStreetMap')

# Afficher la carte (dans un notebook, mettre juste `m`)
m
```

3)  Ajouter des marqueurs

```python
# Exemple avec une liste de logements
logements = [
    {"latitude": 48.8566, "longitude": 2.3522, "etiquette": "D", "surface": 32},
    {"latitude": 48.864, "longitude": 2.341, "etiquette": "B", "surface": 45},
]

for log in logements:
    folium.Marker(
        location=[log['latitude'], log['longitude']],
        popup=f"Étiquette: {log['etiquette']}, Surface: {log['surface']} m²",
        icon=folium.Icon(color='blue' if log['etiquette'] in ['A','B','C'] else 'red')
    ).add_to(m)

m
```

4) Ajouter des clusters de marqueurs

```python
from folium.plugins import MarkerCluster

# Créer un cluster
cluster = MarkerCluster().add_to(m)

for log in logements:
    folium.Marker(
        location=[log['latitude'], log['longitude']],
        popup=f"Étiquette: {log['etiquette']}, Surface: {log['surface']} m²"
    ).add_to(cluster)

m
```

5) Ajouter des cercles (CircleMarker)

```python
# Exemple pour visualiser la surface proportionnelle
for log in logements:
    folium.CircleMarker(
        location=[log['latitude'], log['longitude']],
        radius=log['surface'] / 5,  # taille proportionnelle
        color='green',
        fill=True,
        fill_color='green',
        fill_opacity=0.6,
        popup=f"Surface: {log['surface']} m²"
    ).add_to(m)

m
```

:bulb: Récap

- Folium crée des cartes interactives directement dans les notebooks ou en HTML exportable.
- `Marker` : pour un point simple.
- `MarkerCluster` : regroupe automatiquement les points proches pour plus de lisibilité.
- `CircleMarker` : utile pour représenter une grandeur continue (surface, consommation, etc.).
- Les cartes peuvent être enrichies avec popups, couleurs, icônes et couches multiples.

