# Chapitre 2 : Bases de Python

Python est un langage très utilisé en Data Science grâce à sa **syntaxe simple** et à son **écosystème riche**.  
Dans cette section, nous abordons les notions fondamentales pour pouvoir manipuler des données.

- [Chapitre 2 : Bases de Python](#chapitre-2--bases-de-python)
  - [Variables et types primitifs](#variables-et-types-primitifs)
  - [Opérateurs arithmétiques, de comparaison et logiques](#opérateurs-arithmétiques-de-comparaison-et-logiques)
    - [Opérateurs arithmétiques](#opérateurs-arithmétiques)
    - [Opérateurs de comparaison](#opérateurs-de-comparaison)
    - [Opérateurs logiques (Python)](#opérateurs-logiques-python)
  - [Structures de contrôle](#structures-de-contrôle)
    - [Conditionnelles (`if`, `elif`, `else`)](#conditionnelles-if-elif-else)
    - [Boucles](#boucles)
  - [Fonctions](#fonctions)
  - [Programmation orientée objet (POO)](#programmation-orientée-objet-poo)
    - [Exemple : définir une classe Logement](#exemple--définir-une-classe-logement)
    - [Utilisation :](#utilisation-)
  - [Bibliothèque standard](#bibliothèque-standard)
    - [Manipulation de fichiers et dossiers](#manipulation-de-fichiers-et-dossiers)
    - [Statistiques simples](#statistiques-simples)
    - [Travail avec les dates](#travail-avec-les-dates)
  - [Scripts vs Notebooks](#scripts-vs-notebooks)
  - [Exercice : Collecte des données DPE via l'API ADEME](#exercice--collecte-des-données-dpe-via-lapi-ademe)
    - [Objectif pédagogique](#objectif-pédagogique)
    - [1. Introduction : Qu’est-ce qu’une API ?](#1-introduction--quest-ce-quune-api-)
    - [2. Exemple d’une requête simple](#2-exemple-dune-requête-simple)
    - [3. Instructions de l’exercice](#3-instructions-de-lexercice)
    - [4. Correction pas à pas](#4-correction-pas-à-pas)
      - [a) Import et configuration](#a-import-et-configuration)
      - [b) Boucle sur les départements](#b-boucle-sur-les-départements)
      - [c) Export JSON](#c-export-json)
      - [d) Export CSV](#d-export-csv)
    - [5. Points pédagogiques](#5-points-pédagogiques)
  - [Extension POO : Classe pour gérer les DPE](#extension-poo--classe-pour-gérer-les-dpe)
    - [Objectif pédagogique](#objectif-pédagogique-1)
    - [1. Définition d’une classe `DPECollector`](#1-définition-dune-classe-dpecollector)
    - [2. Utilisation de la classe](#2-utilisation-de-la-classe)
    - [3. Points pédagogiques](#3-points-pédagogiques)


## Variables et types primitifs

Une variable est un **nom qui référence une valeur**.  
Les types primitifs les plus courants :

- `int` : entier
- `float` : nombre à virgule
- `str` : chaîne de caractères
- `bool` : valeur booléenne (`True` / `False`)
- `list` : liste d’éléments (ordre préservé, modifiable)
- `dict` : dictionnaire (clé → valeur, modifiable)

```python
# Types simples
numero_dpe = "2275E0252200Y"  # str
annee_construction = 1948       # int
surface_habitable = 32.0        # float
est_logement_individuel = True # bool

# Liste : consommation par usage
conso_usages = [3506.6, 3670.8, 139.2, 0]  # chauffage, ECS, éclairage, refroidissement
print("Consommation chauffage :", conso_usages[0])

# Dictionnaire : informations d’un logement
logement = {
    "numero_dpe": "2275E0252200Y",
    "code_postal": "75013",
    "surface": 32,
    "type_batiment": "appartement",
    "etiquette_dpe": "D"
}
print("Étiquette DPE :", logement["etiquette_dpe"])

# Ajout et modification
logement["proprietaire"] = "Alice"
conso_usages.append(405.2)  # consommation auxiliaires EF

# Affichage combiné
print("Logement :", logement)
print("Liste des consommations par usage :", conso_usages)
```
## Opérateurs arithmétiques, de comparaison et logiques

### Opérateurs arithmétiques

| Opérateur | Exemple | Résultat |
|-----------|---------|----------|
| Addition `+` | `total_surface = surface_logement1 + surface_logement2` | `32 + 45 = 77` |
| Soustraction `-` | `diff_surface = surface_logement2 - surface_logement1` | `45 - 32 = 13` |
| Multiplication `*` | `besoin_total = besoin_chauffage1 * 2` | `2313.7 * 2 = 4627.4` |
| Division `/` | `moyenne_surface = total_surface / 2` | `77 / 2 = 38.5` |
| Modulo `%` | `reste = surface_logement2 % surface_logement1` | `45 % 32 = 13` |
| Puissance `**` | `surface_au_carre = surface_logement1 ** 2` | `32**2 = 1024` |

### Opérateurs de comparaison

| Opérateur | Exemple | Résultat |
|-----------|---------|----------|
| Égal `==` | `etiquette_logement1 == "D"` | `True` |
| Différent `!=` | `etiquette_logement1 != "A"` | `True` |
| Plus grand `>` | `surface_logement2 > surface_logement1` | `45 > 32 → True` |
| Plus petit `<` | `surface_logement1 < surface_logement2` | `32 < 45 → True` |
| Plus grand ou égal `>=` | `surface_logement1 >= 32` | `True` |
| Plus petit ou égal `<=` | `surface_logement2 <= 50` | `True` |

### Opérateurs logiques (Python)

| Type de données | Opérateur | Exemple | Résultat | Remarques |
|-----------------|-----------|---------|----------|------------|
| Booléens simples | `and` | `True and False` | `False` | Utilisé pour combiner des conditions classiques |
| Booléens simples | `or` | `True or False` | `True` | Préféré à `|` pour les conditions simples |
| Booléens simples | `not` | `not True` | `False` | Négation |
| Tableaux / Series (NumPy / pandas) | `&` | `(df["surface"] > 30) & (df["etiquette"] == "D")` | Series booléenne | Nécessite des parenthèses autour de chaque condition |
| Tableaux / Series (NumPy / pandas) | `|` | `(df["surface"] > 30) | (df["etiquette"] == "D")` | Series booléenne | Nécessite des parenthèses autour de chaque condition |
| Tableaux / Series (NumPy / pandas) | `~` | `~(df["surface"] > 30)` | Series booléenne | Négation élément-par-élément |


> 💡 Remarques :
> - En Python pur, **il faut utiliser `and`, `or`, `not`**. Les opérateurs `&&`, `||`, `!` ne fonctionnent pas en natif et provoqueront une erreur.  
> - Certains packages ou bindings avec d’autres langages peuvent accepter `&&` ou `||` (ex : pandas `.query()`), mais pour le Python classique, restez sur `and/or/not`.  
> - Ces opérateurs sont utilisés pour **filtrer, conditionner ou combiner des tests** dans des boucles et fonctions.


## Structures de contrôle

Les structures de contrôle permettent de **modifier le flux d’exécution** selon des conditions ou de répéter des actions.

### Conditionnelles (`if`, `elif`, `else`)

```python
logement = {
    "etiquette_dpe": "D",
    "surface": 32,
    "type_batiment": "appartement"
}

# Vérifier l'étiquette DPE
if logement["etiquette_dpe"] in ["A", "B", "C"]:
    print("Logement économe en énergie")
elif logement["etiquette_dpe"] == "D":
    print("Logement moyen")
else:
    print("Logement énergivore")
```

### Boucles

- **Boucle `for`** : parcourir une liste ou un dictionnaire
```python
conso_usages = [3506.6, 3670.8, 139.2, 0]  # chauffage, ECS, éclairage, refroidissement

for i, conso in enumerate(conso_usages):
    print(f"Usage {i+1} : {conso} kWh")
```

- **Boucle `while`** : répéter tant qu’une condition est vraie
```python
page = 1
nb_donnees = 5  # simulation récupération API

while nb_donnees > 0:
    print(f"Récupération page {page}...")
    page += 1
    nb_donnees -= 1  # fin de boucle après 5 itérations
```

## Fonctions

Une fonction est un **bloc de code réutilisable** qui peut recevoir des paramètres et retourner un résultat.

```python
# Fonction simple : calcul consommation totale
def consommation_totale(usages):
    """
    Calcule la consommation totale à partir d'une liste d'usages.
    """
    return sum(usages)

conso_usages = [3506.6, 3670.8, 139.2, 0]
total = consommation_totale(conso_usages)
print("Consommation totale :", total)

# Fonction avec dictionnaire : évaluation étiquette
def est_logement_econome(logement):
    if logement["etiquette_dpe"] in ["A", "B", "C"]:
        return True
    return False

logement1 = {"etiquette_dpe": "B"}
logement2 = {"etiquette_dpe": "E"}
print(est_logement_econome(logement1))  # True
print(est_logement_econome(logement2))  # False
```

- Les fonctions permettent de **simplifier le code**, **réutiliser des calculs** et **faciliter la lecture**.
- Elles sont essentielles pour **structurer des scripts ou notebooks**, surtout avec des datasets comme ceux de l’ADEME.


## Programmation orientée objet (POO)

La POO permet de **regrouper des données et des comportements** dans des objets.  
C’est utile pour manipuler des entités complexes comme des logements DPE.

### Exemple : définir une classe Logement

```python
class Logement:
    def __init__(self, numero_dpe, surface, etiquette, type_batiment):
        self.numero_dpe = numero_dpe
        self.surface = surface
        self.etiquette = etiquette
        self.type_batiment = type_batiment

    def consommation_estimee(self):
        """
        Retourne une estimation simplifiée de la consommation totale
        en kWh en fonction de l'étiquette.
        """
        coef = {"A": 0.5, "B": 0.7, "C": 0.9, "D": 1.1, "E": 1.3, "F": 1.5, "G": 1.8}
        return self.surface * coef.get(self.etiquette, 1)

    def afficher_info(self):
        print(f"{self.type_batiment} {self.numero_dpe} - Étiquette {self.etiquette}, Surface {self.surface} m²")
```

### Utilisation :

```python
logement1 = Logement("2275E0252200Y", 32, "D", "appartement")
logement2 = Logement("2275E0252201Y", 45, "B", "maison")

logement1.afficher_info()
print("Conso estimée :", logement1.consommation_estimee())

logement2.afficher_info()
print("Conso estimée :", logement2.consommation_estimee())
```

- Avantages de la POO :  
  - Regroupe **attributs et méthodes** dans un objet  
  - Facilite la **réutilisation** et l’**extension**  
  - Rend le code plus **lisible et structuré**  


## Bibliothèque standard

Python dispose d’une **bibliothèque standard riche** pour de nombreuses tâches courantes.  
Quelques exemples utiles pour la Data Science et les données ADEME :

### Manipulation de fichiers et dossiers

```python
import os
import json
import csv

# Créer un dossier
if not os.path.exists("donnees"):
    os.mkdir("donnees")

# Deux logements
logements = [
    {"numero_dpe": "2275E0252200Y", "surface": 32, "etiquette": "D", "type_batiment": "appartement"},
    {"numero_dpe": "2275E0252201Y", "surface": 45, "etiquette": "B", "type_batiment": "maison"}
]

# Sauvegarder en JSON
with open("donnees/logements.json", "w") as f:
    json.dump(logements, f, indent=4)

# Sauvegarder en CSV
csv_file = "donnees/logements.csv"
with open(csv_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=logements[0].keys())
    writer.writeheader()
    writer.writerows(logements)
```

- Le JSON est pratique pour **les échanges entre applications**.  
- Le CSV est idéal pour **l’analyse tabulaire**, lecture avec Pandas ou Excel.  
- Ces fichiers peuvent servir de **jeu de données simplifié** pour vos premiers tests sur l’API ADEME.


### Statistiques simples
```python
import statistics

conso = [3506.6, 3670.8, 139.2, 0]
print("Moyenne :", statistics.mean(conso))
print("Écart type :", statistics.stdev(conso))
```

### Travail avec les dates
```python
from datetime import datetime

date_str = "2022-02-08"
date_obj = datetime.strptime(date_str, "%Y-%m-%d")
print("Année :", date_obj.year)
```

- La **bibliothèque standard** est souvent suffisante pour les premières explorations de données.  
- Ensuite, on pourra introduire **NumPy, Pandas, Matplotlib** pour des traitements plus puissants et analytiques.

## Scripts vs Notebooks

- **Scripts (.py)** :  
  - Exécutables directement  
  - Bon pour **production et automatisation**  
  - Exemple : `analyse_dpe.py` qui parcourt toutes les données ADEME et calcule des statistiques  

- **Notebooks (.ipynb)** :  
  - Mix code + visualisation + texte  
  - Idéal pour **exploration et pédagogie**  
  - Exemple : `exploration_dpe.ipynb` pour visualiser la répartition des étiquettes DPE  


## Exercice : Collecte des données DPE via l'API ADEME

### Objectif pédagogique
- Comprendre ce qu’est une API et comment l’interroger.
- Manipuler les résultats JSON pour créer des dictionnaires et listes Python.
- Utiliser des boucles `for` et `while` pour parcourir les pages et départements.
- Exporter les données collectées en JSON et CSV.

### 1. Introduction : Qu’est-ce qu’une API ?

Une **API (Application Programming Interface)** permet à un programme de **demander des données** à un service et de recevoir une réponse structurée (souvent en **JSON**).  

- **Route** : l’URL ou endpoint que l’on interroge. Exemple :  
  ```
  https://data.ademe.fr/data-fair/api/v1/datasets/dpe03existant/lines
  ```
- **Paramètres** : permettent de filtrer ou paginer les données.  
  Exemple :  
  - `page` : numéro de page
  - `size` : nombre de résultats par page
  - `select` : quelles colonnes récupérer
  - `qs` : filtre sur certaines valeurs (ex. `code_departement_ban:75`)

### 2. Exemple d’une requête simple

```python
import requests

API_URL = "https://data.ademe.fr/data-fair/api/v1/datasets/dpe03existant/lines"

params = {
    "page": 1,
    "size": 5,
    "select": "numero_dpe,date_reception_dpe,code_postal_ban,etiquette_dpe",
    "qs": "code_departement_ban:75"
}

response = requests.get(API_URL, params=params)
data = response.json()

for line in data.get("results", []):
    print(line)
```

> Résultat : un petit tableau de dictionnaires avec les informations DPE pour Paris.

### 3. Instructions de l’exercice

1. **Variables de configuration**

```python
departements = ["75", "69", "13"]
page_size = 5
api_url = "https://data.ademe.fr/data-fair/api/v1/datasets/dpe03existant/lines"
```

2. **Collecte des données**

Pour chaque département :

- Initialiser `page = 1`.
- Tant que l’API renvoie des résultats (`while True`) ou `page < 20`:
  - Interroger l’API avec `requests.get()`.
  - Si la liste de résultats est vide, sortir de la boucle.
  - Pour chaque ligne, créer un dictionnaire contenant :
    - `numero_dpe`
    - `date_reception_dpe`
    - `code_postal_ban`
    - `etiquette_dpe`
    - `surface_habitable_logement`
    - `type_batiment`
  - Ajouter ce dictionnaire à une liste `donnees_collectees`.
  - Passer à la page suivante.

3. **Export**

- Sauvegarder la liste de dictionnaires dans un fichier JSON : `dpe_<departement>.json`
- Bonus : sauvegarder également en CSV.

### 4. Correction pas à pas

#### a) Import et configuration

```python
import requests
import json
import csv

departements = ["75", "69"]
page_size = 5
api_url = "https://data.ademe.fr/data-fair/api/v1/datasets/dpe03existant/lines"
```

#### b) Boucle sur les départements

```python
for dep in departements:
    donnees_collectees = []
    page = 1

    while (True | page<20):
        params = {
            "page": page,
            "size": page_size,
            "select": "numero_dpe,date_reception_dpe,code_postal_ban,etiquette_dpe,surface_habitable_logement,type_batiment",
            "qs": f"code_departement_ban:{dep}"
        }

        response = requests.get(api_url, params=params)
        data = response.json()
        results = data.get("results", [])

        if not results:
            break

        for line in results:
            dpe_dict = {
                "numero_dpe": line.get("numero_dpe"),
                "date_reception_dpe": line.get("date_reception_dpe"),
                "code_postal_ban": line.get("code_postal_ban"),
                "etiquette_dpe": line.get("etiquette_dpe"),
                "surface_habitable_logement": line.get("surface_habitable_logement"),
                "type_batiment": line.get("type_batiment")
            }
            donnees_collectees.append(dpe_dict)

        page += 1
```

#### c) Export JSON

```python
    json_file = f"dpe_{dep}.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(donnees_collectees, f, ensure_ascii=False, indent=2)
    print(f"{len(donnees_collectees)} DPE sauvegardés dans {json_file}")
```

#### d) Export CSV

```python
    csv_file = f"dpe_{dep}.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=donnees_collectees[0].keys())
        writer.writeheader()
        writer.writerows(donnees_collectees)
    print(f"CSV généré : {csv_file}")
```

### 5. Points pédagogiques

- Comprendre comment fonctionne une API REST et la pagination.
- Manipuler les **dictionnaires et listes** Python.
- Utiliser des **boucles `for` et `while`** pour parcourir les données.
- Exporter les données en **JSON et CSV** pour analyses ultérieures.

## Extension POO : Classe pour gérer les DPE

### Objectif pédagogique
- Introduire la **POO** en Python avec des concepts simples : classe, attributs, méthodes.  
- Pratiquer les **conditions** et **boucles** dans une méthode.  
- Calculer des **statistiques simples** sur les données collectées.

### 1. Définition d’une classe `DPECollector`

```python
class DPECollector:
    def __init__(self, departement):
        self.departement = departement
        self.donnees = []

    def ajouter_dpe(self, dpe_dict):
        """Ajoute un dictionnaire DPE à la liste"""
        self.donnees.append(dpe_dict)

    def statistiques_etiquettes(self):
        """Retourne la distribution des étiquettes DPE"""
        distribution = {}
        for dpe in self.donnees:
            etiquette = dpe.get("etiquette_dpe", "Inconnu")
            if etiquette in distribution:
                distribution[etiquette] += 1
            else:
                distribution[etiquette] = 1
        return distribution

    def moyenne_surface(self):
        """Retourne la surface moyenne des logements"""
        total_surface = 0
        count = 0
        for dpe in self.donnees:
            surface = dpe.get("surface_habitable_logement")
            if surface:
                total_surface += surface
                count += 1
        return total_surface / count if count > 0 else 0
```

### 2. Utilisation de la classe

```python
# Création d'un collector pour Paris
collector_paris = DPECollector("75")

# Supposons qu'on ait déjà récupéré quelques données depuis l'API
dpes = [
    {"numero_dpe": "2275E1", "etiquette_dpe": "D", "surface_habitable_logement": 32},
    {"numero_dpe": "2275E2", "etiquette_dpe": "C", "surface_habitable_logement": 45},
    {"numero_dpe": "2275E3", "etiquette_dpe": "D", "surface_habitable_logement": 50},
]

for d in dpes:
    collector_paris.ajouter_dpe(d)

# Statistiques
print("Distribution des étiquettes :", collector_paris.statistiques_etiquettes())
print("Surface moyenne :", collector_paris.moyenne_surface())
```

> Résultat attendu :  
> Distribution des étiquettes : `{'D': 2, 'C': 1}`  
> Surface moyenne : `42.33`

### 3. Points pédagogiques

- **Classe et méthode** : comment organiser le code pour manipuler des données.
- **Boucle et condition** : compter les occurrences d’une étiquette.
- **Méthode calcul** : calculer la moyenne sur une liste d’attributs.
- Facilite l’**extension future** : ajout d’autres statistiques, export CSV/JSON, filtrage.
