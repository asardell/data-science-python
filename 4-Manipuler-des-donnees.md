# Chapitre 4 - Manipuler les données

- [Chapitre 4 - Manipuler les données](#chapitre-4---manipuler-les-données)
  - [Manipuler les données avec NumPy](#manipuler-les-données-avec-numpy)
    - [Importer NumPy et charger une colonne en tableau](#importer-numpy-et-charger-une-colonne-en-tableau)
    - [Calculer des statistiques de base](#calculer-des-statistiques-de-base)
    - [Opérations sur les tableaux](#opérations-sur-les-tableaux)
    - [Tableaux multidimensionnels](#tableaux-multidimensionnels)
    - [Statistiques sur plusieurs colonnes](#statistiques-sur-plusieurs-colonnes)
  - [Manipuler les données avec SciPy](#manipuler-les-données-avec-scipy)
    - [Importer SciPy et préparer les données](#importer-scipy-et-préparer-les-données)
    - [Statistiques descriptives avancées](#statistiques-descriptives-avancées)
    - [Test de normalité](#test-de-normalité)
    - [Corrélation entre deux variables](#corrélation-entre-deux-variables)
    - [Tests de comparaison](#tests-de-comparaison)
      - [Test t pour vérifier si la moyenne d’une variable diffère d’une valeur](#test-t-pour-vérifier-si-la-moyenne-dune-variable-diffère-dune-valeur)
      - [Test t pour comparer deux groupes](#test-t-pour-comparer-deux-groupes)
  - [Manipulation les données avec Pandas](#manipulation-les-données-avec-pandas)
    - [Importer Pandas et charger le CSV](#importer-pandas-et-charger-le-csv)
    - [Affichage et configuration](#affichage-et-configuration)
    - [Création de nouvelles colonnes](#création-de-nouvelles-colonnes)
    - [Gestion des types de colonnes avec Pandas (`dtypes`)](#gestion-des-types-de-colonnes-avec-pandas-dtypes)
      - [Vérifier les types de colonnes](#vérifier-les-types-de-colonnes)
    - [Changer le type d’une colonne](#changer-le-type-dune-colonne)
    - [Statistiques descriptives avec `describe()`](#statistiques-descriptives-avec-describe)
    - [Nettoyage et gestion des données manquantes](#nettoyage-et-gestion-des-données-manquantes)
    - [Filtrage simple et complexe](#filtrage-simple-et-complexe)
    - [Supprimer les lignes avec valeurs manquantes](#supprimer-les-lignes-avec-valeurs-manquantes)
    - [Agrégations et statistiques](#agrégations-et-statistiques)
    - [Renommer des colonnes](#renommer-des-colonnes)
    - [Concaténation de DataFrames](#concaténation-de-dataframes)
    - [Jointures de DataFrames avec Pandas](#jointures-de-dataframes-avec-pandas)
    - [Exporter les données préparées](#exporter-les-données-préparées)
  - [Exercice : Préparation des données ADEME pour les isualisation et ML](#exercice--préparation-des-données-ademe-pour-les-isualisation-et-ml)
    - [Importer les bibliothèques et charger le CSV](#importer-les-bibliothèques-et-charger-le-csv)
    - [Sélection des colonnes pertinentes](#sélection-des-colonnes-pertinentes)
    - [Filtrer les lignes invalides](#filtrer-les-lignes-invalides)
    - [Vérifier les valeurs manquantes](#vérifier-les-valeurs-manquantes)
    - [Imputer les valeurs manquantes](#imputer-les-valeurs-manquantes)
    - [Supprimer les colonnes avec trop de valeurs manquantes](#supprimer-les-colonnes-avec-trop-de-valeurs-manquantes)
    - [Créer des colonnes calculées utiles](#créer-des-colonnes-calculées-utiles)
    - [Vérifier le DataFrame final](#vérifier-le-dataframe-final)
    - [Exporter le DataFrame préparé](#exporter-le-dataframe-préparé)
    - [Résultat attendu](#résultat-attendu)


## Manipuler les données avec NumPy

NumPy est la bibliothèque principale pour **le calcul scientifique et la manipulation de tableaux numériques** en Python.  
Même si Pandas est idéal pour les données tabulaires, NumPy est très utile pour :

- effectuer des calculs rapides sur des colonnes numériques  
- créer des tableaux multidimensionnels  
- appliquer des fonctions vectorisées  

Dans cette section, nous allons utiliser **les données ADEME** stockées dans `dpe_paris.csv`. Pour explorer les données il est recommandé de travailler sur un noteook plutôt que des scripts.


### Importer NumPy et charger une colonne en tableau

```python
import numpy as np  
import pandas as pd

## Charger le CSV
df = pd.read_csv("dpe_paris.csv")

## Créer un tableau NumPy à partir de la colonne surface habitable
surface = np.array(df['surface_habitable_logement'])

## Afficher le tableau
print(surface)
```

**Explication :**  
- `np.array()` transforme une série Pandas en **tableau NumPy**.  
- Les tableaux NumPy permettent de faire des calculs vectorisés sans boucles.


### Calculer des statistiques de base

```python
## Moyenne de la surface
moyenne_surface = np.mean(surface)
print("Surface moyenne :", moyenne_surface)

## Médiane
mediane_surface = np.median(surface)
print("Surface médiane :", mediane_surface)

## Écart-type
ecart_type_surface = np.std(surface)
print("Écart type :", ecart_type_surface)

## Minimum et maximum
min_surface = np.min(surface)
max_surface = np.max(surface)
print("Surface min :", min_surface, "max :", max_surface)
```

**Explication :**  
- `np.mean()` calcule la moyenne.  
- `np.median()` calcule la médiane.  
- `np.std()` calcule l’écart-type, utile pour mesurer la dispersion.  
- `np.min()` et `np.max()` renvoient les valeurs minimale et maximale.


### Opérations sur les tableaux

```python
## Ajouter 10 m² à chaque logement
surface_plus10 = surface + 10
print(surface_plus10[:5])

## Filtrer les logements supérieurs à 50 m²
surface_sup50 = surface[surface > 50]
print(surface_sup50)

## Multiplier par un facteur pour convertir m² en ft² (1 m² = 10.764 ft²)
surface_ft2 = surface * 10.764
print(surface_ft2[:5])
```

**Explication :**  
- Les opérations sur un tableau NumPy sont **vectorisées**, donc très rapides.  
- On peut filtrer directement avec des conditions, comme `surface > 50`.  


### Tableaux multidimensionnels

Parfois, on veut travailler sur plusieurs colonnes à la fois.

```python
## Extraire besoin_chauffage et conso_chauffage_ef
besoin_chauffage = np.array(df['besoin_chauffage'])
conso_chauffage_ef = np.array(df['conso_chauffage_ef'])

## Créer un tableau 2D avec ces colonnes
data = np.column_stack((surface, besoin_chauffage, conso_chauffage_ef))
print(data[:5])
```

**Explication :**  
- `np.column_stack()` permet de créer un **tableau 2D** avec plusieurs colonnes.  
- Chaque ligne représente un logement, chaque colonne une variable.  


### Statistiques sur plusieurs colonnes

```python
## Moyenne par colonne
moyennes = np.mean(data, axis=0)
print("Moyennes (surface, besoin chauffage, conso chauffage ef) :", moyennes)

## Somme par colonne
sommes = np.sum(data, axis=0)
print("Sommes (surface, besoin chauffage, conso chauffage ef) :", sommes)

## Écart-type par colonne
ecarts = np.std(data, axis=0)
print("Écarts-types (surface, besoin chauffage, conso chauffage ef) :", ecarts)
```

**Explication :**  
- `axis=0` signifie que l’on calcule **par colonne**.  
- `axis=1` permettrait de calculer **par ligne** (par logement).  

## Manipuler les données avec SciPy

SciPy est une bibliothèque Python utilisée pour les **calculs scientifiques avancés**, en particulier pour :

- les statistiques  
- l’optimisation  
- l’interpolation  
- les tests scientifiques  

Dans cette section, nous allons nous concentrer sur **les statistiques avec SciPy** en utilisant le jeu de données ADEME.


### Importer SciPy et préparer les données

```python
import pandas as pd
from scipy import stats
import numpy as np

## Charger le CSV
df = pd.read_csv("dpe_paris.csv")

## Créer des tableaux NumPy pour les variables numériques
surface = np.array(df['surface_habitable_logement'])
besoin_chauffage = np.array(df['besoin_chauffage'])
conso_chauffage_ef = np.array(df['conso_chauffage_ef'])
```

### Statistiques descriptives avancées

```python
## Moyenne, médiane, écart-type avec SciPy
mean_surface = stats.tmean(surface)
median_surface = stats.tmedian(surface)
std_surface = stats.tstd(surface)

print("Moyenne surface :", mean_surface)
print("Médiane surface :", median_surface)
print("Écart-type surface :", std_surface)
```

**Explication :**  
- `stats.tmean()` calcule la moyenne  
- `stats.tmedian()` calcule la médiane  
- `stats.tstd()` calcule l’écart-type



### Test de normalité

Le test de normalité de Shapiro-Wilk permet de vérifier si les données suivent une **distribution normale**.

```python
stat, p = stats.shapiro(surface)
print("Statistique Shapiro-Wilk :", stat, "p-value :", p)

if p > 0.05:
    print("La surface habitable suit une distribution normale")
else:
    print("La surface habitable ne suit pas une distribution normale")
```

**Explication :**  
- `stats.shapiro()` renvoie une statistique et une p-value.  
- Si `p > 0.05`, on considère que la distribution est proche d’une normale.


### Corrélation entre deux variables

On peut mesurer la corrélation linéaire entre deux colonnes avec **Pearson** :

```python
correlation, pval = stats.pearsonr(besoin_chauffage, conso_chauffage_ef)
print("Corrélation :", correlation, "p-value :", pval)
```

**Explication :**  
- `stats.pearsonr(x, y)` renvoie un coefficient de corrélation et une p-value.  
- Valeur proche de 1 → forte corrélation positive  
- Valeur proche de -1 → forte corrélation négative  
- Valeur proche de 0 → pas de corrélation


### Tests de comparaison

#### Test t pour vérifier si la moyenne d’une variable diffère d’une valeur

```python
t_stat, p_val = stats.ttest_1samp(besoin_chauffage, 2000)
print("t-statistic :", t_stat, "p-value :", p_val)

if p_val < 0.05:
    print("La moyenne du besoin chauffage est significativement différente de 2000 kWh")
else:
    print("Pas de différence significative avec 2000 kWh")
```

**Explication :**  
- `ttest_1samp()` teste si la moyenne d’un échantillon est différente d’une valeur hypothétique.  



#### Test t pour comparer deux groupes

```python
# Par exemple, comparer le besoin chauffage entre appartements et maisons
group_appartement = df[df['type_batiment'] == 'appartement']['besoin_chauffage']
group_maison = df[df['type_batiment'] == 'maison']['besoin_chauffage']

t_stat, p_val = stats.ttest_ind(group_appartement, group_maison, nan_policy='omit')
print("t-statistic :", t_stat, "p-value :", p_val)
```

**Explication :**  
- `ttest_ind()` compare les moyennes de deux échantillons indépendants.  
- `nan_policy='omit'` ignore les valeurs manquantes.


## Manipulation les données avec Pandas

Pandas est la bibliothèque centrale pour manipuler des données tabulaires en Python.  
Elle permet de charger, explorer, nettoyer, transformer et exporter des données facilement.

### Importer Pandas et charger le CSV

Pour utiliser Pandas, il faut importer la bibliothèque et charger un fichier CSV dans un DataFrame.

```python
import pandas as pd

# Charger le CSV ADEME
df = pd.read_csv("dpe_paris.csv", dtype={"identifiant_ban": str})
```

Explications des arguments de pd.read_csv :  
- "dpe_paris.csv" : chemin du fichier CSV  
- dtype={"identifiant_ban": str} : force la colonne identifiant_ban en chaîne de caractères  
- sep=',' : séparateur (par défaut `,`)  
- index_col=None : ne pas utiliser de colonne comme index  
- parse_dates=['date_reception_dpe'] : convertir automatiquement une colonne en date si besoin  

### Affichage et configuration

```python
# Afficher les premières lignes
print(df.head(5))  # head(n) retourne les n premières lignes

# Afficher les dernières lignes
print(df.tail(5))  # tail(n) retourne les n dernières lignes

# Informations générales sur le DataFrame
print(df.info())  # types de colonnes, nombre de valeurs non-nulles

# Statistiques descriptives des colonnes numériques
print(df.describe())  # count, mean, std, min, max, quartiles

# Voir toutes les colonnes et éviter les coupures
pd.set_option('display.max_columns', None)
print(df.head())
```

Autres options utiles :  
- display.max_rows : nombre maximal de lignes affichées  
- display.float_format : format des nombres flottants  
- display.max_colwidth : largeur maximale pour afficher les chaînes  


### Création de nouvelles colonnes

```python
# Copier une colonne existante
df['surface_m2'] = df['surface_habitable_logement']

# Créer une colonne calculée
df['besoin_par_m2'] = df['besoin_chauffage'] / df['surface_habitable_logement']

# Créer une colonne conditionnelle avec np.where
import numpy as np
df['grande_surface'] = np.where(df['surface_habitable_logement'] > 50, True, False)

Explications :  
- np.where(condition, valeur_si_vrai, valeur_si_faux)  
- df['nouvelle_colonne'] = ... crée une nouvelle colonne  
```

### Gestion des types de colonnes avec Pandas (`dtypes`)

En Data Science, il est important de connaître et contrôler le type des colonnes dans un DataFrame :  
- Numérique : int, float  
- Catégoriel / texte : object ou string  
- Date / heure : datetime  

#### Vérifier les types de colonnes

```python
# Affiche le type de chaque colonne
print(df.dtypes)
```

Explications :
- df.dtypes : retourne un objet Series avec le type de chaque colonne
- Exemple : surface_habitable_logement → float64, identifiant_ban → object, date_reception_dpe → datetime64[ns]

### Changer le type d’une colonne

1) Conversion en entier

```python
df['surface_habitable_logement'] = df['surface_habitable_logement'].astype(int)
```

Explications :
- astype(int) : convertit les valeurs en entier
- Les valeurs NaN doivent être gérées avant la conversion (sinon erreur)

2) Conversion en float

```python
df['besoin_chauffage'] = df['besoin_chauffage'].astype(float)
```

3) Conversion en chaîne (string)

```python
df['identifiant_ban'] = df['identifiant_ban'].astype(str)
```

4) Conversion en datetime

```python
df['date_reception_dpe'] = pd.to_datetime(df['date_reception_dpe'], format='%Y-%m-%d', errors='coerce')
```

Explications :
- pd.to_datetime() : convertit une colonne en type datetime
- format='%Y-%m-%d' : indique le format de date attendu
- errors='coerce' : transforme les valeurs invalides en NaT (Not a Time)

:bulb: Astuces pratiques

- Toujours vérifier les types après un `read_csv()` car Pandas peut mal détecter certaines colonnes  
- Les colonnes numériques avec NaN seront souvent en float, même si ce sont des entiers  
- Les colonnes date doivent être converties pour pouvoir trier, filtrer ou extraire le mois/année  
- Pour convertir plusieurs colonnes en même temps :  

```python
cols = ['col1', 'col2']  
df[cols] = df[cols].astype(float)
```

### Statistiques descriptives avec `describe()`

La méthode `describe()` de Pandas permet d’obtenir rapidement un résumé statistique des colonnes numériques d’un DataFrame.  
Elle fournit : nombre de valeurs non nulles (`count`), moyenne (`mean`), écart-type (`std`), minimum (`min`), maximum (`max`) et les quartiles (`25%`, `50%`, `75%`).  

On peut également ajouter des percentiles personnalisés.

1) Décrire une seule colonne

```python
df['besoin_chauffage'].describe()
```

Explications :  
- count : nombre de valeurs non nulles  
- mean : moyenne  
- std : écart-type  
- min : valeur minimale  
- 25%, 50%, 75% : quartiles  
- max : valeur maximale  

2) Décrire plusieurs colonnes

```python
df[['surface_habitable_logement', 'besoin_chauffage', 'conso_chauffage_ef']].describe()
```

3) Ajouter des percentiles personnalisés

```python
df['besoin_chauffage'].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
```

Explications :  
- percentiles=[…] permet de définir les percentiles que l’on veut afficher  
- 0.1 = 10ème percentile, 0.9 = 90ème percentile, etc.  
- Utile pour détecter les valeurs extrêmes ou la distribution des données
  
### Nettoyage et gestion des données manquantes

```python
# Vérifier les valeurs manquantes par colonne
print(df.isnull().sum())

# Supprimer les doublons
df = df.drop_duplicates()

# Imputer une valeur par défaut pour la surface habitable si manquante
df['surface_habitable_logement'] = df['surface_habitable_logement'].fillna(0)

# Imputer la moyenne pour le besoin chauffage
moy_besoin = df['besoin_chauffage'].mean()
df['besoin_chauffage'] = df['besoin_chauffage'].fillna(moy_besoin)
```

Explications :  
- fillna(valeur) : remplace les valeurs NaN par "valeur"  
- drop_duplicates() : supprime les lignes identiques  
- isnull() : retourne True si la valeur est NaN  


### Filtrage simple et complexe

```python
# Filtrer les logements avec surface > 50 m²
df_grands = df[df['surface_habitable_logement'] > 50]

# Filtrer par surface et étiquette DPE
df_filtre = df[(df['surface_habitable_logement'] > 50) & (df['etiquette_dpe'] == 'D')]

# Filtrer par liste de départements
df_dep = df[df['code_departement_ban'].isin(['75', '77', '78'])]

Explications :  
- df[condition] : retourne un DataFrame filtré  
- & pour "ET", | pour "OU"  
- isin([liste]) : filtre sur plusieurs valeurs possibles  
```


### Supprimer les lignes avec valeurs manquantes

```python
# Supprimer les lignes où certaines colonnes sont NaN
df = df.dropna(subset=['etiquette_dpe', 'besoin_chauffage'])

Explications :  
- dropna(subset=[col1, col2]) supprime uniquement si une des colonnes spécifiées est NaN  
- inplace=True peut être utilisé pour modifier le DataFrame sur place  
```


### Agrégations et statistiques

```python
# Moyenne de surface par type de bâtiment
moyenne_par_type = df.groupby('type_batiment')['surface_habitable_logement'].mean()
print(moyenne_par_type)

# Compter le nombre de DPE par étiquette
compte_etiquette = df['etiquette_dpe'].value_counts()
print(compte_etiquette)

# Agrégation multiple
agg = df.groupby('type_batiment').agg({
    'surface_habitable_logement': ['mean', 'max', 'min'],
    'besoin_chauffage': ['mean', 'sum']
})
print(agg)
```

Explications :  
- groupby('colonne') : regroupe les données par valeur unique  
- agg({col: [fonctions]}) : appliquer plusieurs fonctions d’agrégation sur différentes colonnes  
- value_counts() : compte le nombre d’occurrences pour chaque valeur unique  


### Renommer des colonnes

```python
df = df.rename(columns={
    'surface_habitable_logement': 'surface_logement',
    'besoin_chauffage': 'besoin_chauffage_kwh'
})
```

Explications :  
- rename(columns={ancien: nouveau}) renomme les colonnes  
- inplace=True pour modifier sur place  


### Concaténation de DataFrames

```python
# Exemple : concaténer deux DataFrames (par exemple différents départements)
df_75 = df[df['code_departement_ban'] == '75']
df_77 = df[df['code_departement_ban'] == '77']

df_concat = pd.concat([df_75, df_77], ignore_index=True)
print(df_concat.info())
```

Explications :  
- pd.concat([df1, df2], axis=0) : concaténation verticale  
- ignore_index=True : réindexe les lignes du DataFrame résultant  


### Jointures de DataFrames avec Pandas

Lorsque vous avez plusieurs jeux de données (par exemple ADEME + adresses avec coordonnées GPS), utilisez merge().

```python
# Charger le CSV des adresses
df_addr = pd.read_csv("adresses-noDept.csv", dtype={"Identifiant__BAN": str})

# Renommer la colonne pour correspondre
df_addr = df_addr.rename(columns={"id": "identifiant_ban"})

# Réaliser la jointure
df_merged = pd.merge(
    left=df_concat,
    right=df_addr[['identifiant_ban', 'latitude', 'longitude']],
    on='identifiant_ban',
    how='left'
)

# Vérification
print(df_merged.head())
print(df_merged.info())
```

Explications :  
- on='identifiant_ban' : colonne clé pour faire la correspondance  
- how='left' : conserve toutes les lignes de df_concat, ajoute latitude/longitude quand disponible  
- how='inner' : ne garde que les lignes présentes dans les deux DataFrames  
- left_on / right_on : si les colonnes ont des noms différents  
- after merge, vérifier les NaN et nettoyer si nécessaire  


### Exporter les données préparées

```python
# Export CSV
df_concat.to_csv("dpe_prepared.csv", index=False)

# Export Excel
df_concat.to_excel("dpe_prepared.xlsx", index=False)
```

Explications :  
- index=False : ne pas écrire l’index dans le fichier  
- to_csv() et to_excel() exportent le DataFrame sur disque  


## Exercice : Préparation des données ADEME pour les isualisation et ML

**Objectif :** partir du CSV `dpe_paris.csv` et obtenir un DataFrame prêt pour l’analyse et le Machine Learning, en :
- sélectionnant les colonnes pertinentes
- filtrant certaines lignes
- gérant les valeurs manquantes
- créant quelques colonnes calculées


### Importer les bibliothèques et charger le CSV

```python
import pandas as pd  
import numpy as np  

df = pd.read_csv("dpe_paris.csv", dtype={"identifiant_ban": str}, parse_dates=['date_reception_dpe'])
```

Explications :
- dtype={"identifiant_ban": str} : force la colonne identifiant_ban en chaîne
- parse_dates=['date_reception_dpe'] : convertit cette colonne en type date


### Sélection des colonnes pertinentes

On ne garde que les colonnes utiles pour des graphiques ou ML.  
Exemples :

```python
colonnes_quanti = [
    'surface_habitable_logement',
    'besoin_chauffage',
    'conso_chauffage_ef',
    'conso_ecs_ef',
    'conso_5_usages_ef',
    'emission_ges_5_usages_par_m2'
]

colonnes_quali = [
    'etiquette_dpe',
    'etiquette_ges',
    'type_batiment',
    'periode_construction',
    'zone_climatique'
]

colonnes_a_garder = colonnes_quanti + colonnes_quali
df = df[colonnes_a_garder]
```

Explications :
- df[colonnes_a_garder] : ne conserve que ces colonnes
- colonnes_quanti : variables numériques
- colonnes_quali : variables catégorielles


### Filtrer les lignes invalides

- Supprimer les logements avec surface <= 0 ou besoin_chauffage <= 0
- Supprimer les lignes avec valeurs manquantes critiques

```python
df = df[df['surface_habitable_logement'] > 0]  
df = df[df['besoin_chauffage'] > 0]
```

Explications :
- df[condition] retourne uniquement les lignes qui respectent la condition


### Vérifier les valeurs manquantes

```python
print(df.isnull().sum())
```

- Identifier les colonnes avec trop de NaN
- Décider de drop ou d’imputer


### Imputer les valeurs manquantes

- Pour les colonnes quantitatives, on peut utiliser la moyenne ou la médiane
- Pour les colonnes qualitatives, on peut utiliser la valeur la plus fréquente

```python
# Imputation pour les colonnes numériques
for col in colonnes_quanti:
    median_val = df[col].median()
    df[col] = df[col].fillna(median_val)

# Imputation pour les colonnes catégorielles
for col in colonnes_quali:
    mode_val = df[col].mode()[0]
    df[col] = df[col].fillna(mode_val)
```

Explications :
- fillna(valeur) remplace les NaN par la valeur donnée
- median() ou mode()[0] pour remplir intelligemment


### Supprimer les colonnes avec trop de valeurs manquantes

- Si une colonne a plus de 50% de NaN, on peut décider de la supprimer

```python
taux_nan = df.isnull().mean()  
colonnes_a_drop = taux_nan[taux_nan > 0.5].index  
df = df.drop(columns=colonnes_a_drop)
```

Explications :
- isnull().mean() retourne le pourcentage de NaN par colonne
- drop(columns=...) supprime les colonnes identifiées


### Créer des colonnes calculées utiles

```python
# Exemple : besoin par m²
df['besoin_par_m2'] = df['besoin_chauffage'] / df['surface_habitable_logement']
# Exemple : consommation totale par logement (chauffage + ECS + autres usages)
df['conso_totale'] = df['conso_chauffage_ef'] + df['conso_ecs_ef'] + df['conso_5_usages_ef']
```


### Vérifier le DataFrame final

```python
print(df.info())  
print(df.head())
```

- Vérifier les types de colonnes
- Vérifier les valeurs manquantes restantes
- Vérifier les colonnes calculées


### Exporter le DataFrame préparé

```python
df.to_csv("dpe_prepared.csv", index=False)
```

Explications :
- index=False : ne pas exporter l’index
- Le fichier est prêt pour des graphiques ou des modèles ML dans les prochains chapitres


### Résultat attendu

- Un DataFrame avec environ 10 à 15 colonnes pertinentes (quantitatives et qualitatives)
- Pas de valeurs manquantes pour les colonnes importantes
- Colonnes calculées utiles pour l’analyse
- Fichier CSV prêt à être exploité pour des graphiques ou des modèles
