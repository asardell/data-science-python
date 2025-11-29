# Chapitre 3 : Structuration du code et bonnes pratiques

Dans ce chapitre, nous allons aborder des notions qui permettent de **mieux organiser votre code Python**, le rendre **plus robuste**, et préparer l'environnement pour travailler sur des bibliothèques comme Pandas et Matplotlib.

- [Chapitre 3 : Structuration du code et bonnes pratiques](#chapitre-3--structuration-du-code-et-bonnes-pratiques)
  - [Gestion des erreurs et exceptions](#gestion-des-erreurs-et-exceptions)
    - [Exemple](#exemple)
    - [Exercice : Gestion des erreurs et exceptions lors de l'interrogation de l'API ADEME](#exercice--gestion-des-erreurs-et-exceptions-lors-de-linterrogation-de-lapi-ademe)
  - [Fonctions et paramètres avancés (args, kwargs)](#fonctions-et-paramètres-avancés-args-kwargs)
    - [Exemple simple](#exemple-simple)
  - [Organisation du code : modules et packages](#organisation-du-code--modules-et-packages)
    - [Rôle de `__init__.py`](#rôle-de-__init__py)
  - [Style PEP8 et bonnes pratiques](#style-pep8-et-bonnes-pratiques)
    - [Exemple correct](#exemple-correct)
    - [Exemple incorrect](#exemple-incorrect)
  - [Environnements virtuels](#environnements-virtuels)
    - [Vérifier Python et pip](#vérifier-python-et-pip)
    - [Installer `virtualenv` (optionnel)](#installer-virtualenv-optionnel)
    - [Créer un environnement virtuel](#créer-un-environnement-virtuel)
    - [Activer et désactiver l’environnement](#activer-et-désactiver-lenvironnement)
    - [Installer des librairies dans l’environnement](#installer-des-librairies-dans-lenvironnement)
    - [Exporter les dépendances](#exporter-les-dépendances)
    - [Créer un second environnement pour test](#créer-un-second-environnement-pour-test)
    - [Utiliser l’environnement virtuel dans VS Code](#utiliser-lenvironnement-virtuel-dans-vs-code)

---

## Gestion des erreurs et exceptions

- Les erreurs peuvent survenir à l'exécution : division par zéro, fichier manquant, connexion API échouée…
- Python permet de les **capturer et gérer** avec `try / except`.

### Exemple

```python
dividende = 10
diviseur = 0

try:
    resultat = dividende / diviseur
except ZeroDivisionError:
    print("Erreur : division par zéro !")
```

- On peut aussi utiliser `finally` pour exécuter du code quoi qu’il arrive (fermer un fichier, libérer une ressource).

### Exercice : Gestion des erreurs et exceptions lors de l'interrogation de l'API ADEME

Dans le chapitre précédent, vous avez interrogé l’API ADEME page par page et stocké les résultats dans un dictionnaire.  
Cependant, plusieurs problèmes peuvent survenir :

- L’API ne répond pas (`ConnectionError`)
- La page n’existe pas ou le serveur renvoie une erreur (`HTTPError`)
- Les données renvoyées ne sont pas au format attendu (`JSONDecodeError`)

Utiliser `try` / `except` pour **capturer toute erreur et sortir proprement de la boucle :

```python
...
    try:
        ...
    except Exception as e:
        print(f"Une erreur est survenue : {e}. Arrêt du script.")
        break
```

## Fonctions et paramètres avancés (args, kwargs)

- Permettent de **rendre vos fonctions plus flexibles**.
- `*args` : liste de paramètres positionnels variables
- `**kwargs` : dictionnaire de paramètres nommés variables

### Exemple simple

```python
def summary_stats(*args, **kwargs):
    print("Valeurs :", args)
    for key, value in kwargs.items():
        print(f"{key} = {value}")

summary_stats(10, 20, 30, moyenne=20, max=30)
```

- Utile pour créer des fonctions qui traitent différents types de mesures, par exemple plusieurs variables DPE.

---

## Organisation du code : modules et packages

- **Module** : fichier Python `.py` que l’on peut importer  
  Exemple : `utils.py` contient des fonctions réutilisables comme `compute_mean()` ou `load_data()`.  

- **Package** : dossier contenant plusieurs modules + fichier `__init__.py`  
  Exemple :

```
dpe_project/
│
├── utils.py
├── data_loader.py
├── analysis/
│   ├── __init__.py
│   ├── stats.py
│   └── plots.py
```

### Rôle de `__init__.py`
- Indique à Python que le dossier doit être traité comme un **package**.  
- Permet d’exposer des modules ou fonctions directement au moment de l’import.  
- Peut rester vide si vous n’avez pas besoin de code d’initialisation.  
- On peut aussi importer des fonctions directement pour simplifier l’accès :

```python
# contenu de analysis/__init__.py
from .stats import compute_mean, compute_median
from .plots import plot_distribution
```

- Ensuite, dans un autre script, on peut importer ainsi :

```python
from analysis import compute_mean, plot_distribution

mean_val = compute_mean(data)
plot_distribution(data)
```


## Style PEP8 et bonnes pratiques

- Respecter un **style uniforme** améliore la lisibilité et la maintenance
- Règles principales :
  - 4 espaces pour l’indentation
  - Noms de variables en `snake_case`
  - Noms de classes en `CamelCase`
  - Limite de 79 caractères par ligne
  - Espaces autour des opérateurs `=`, `+`, etc.

### Exemple correct

```python
def calcul_besoin_chauffage(surface, consommation_par_m2):
    return surface * consommation_par_m2
```

### Exemple incorrect

```python
def calculBesoinChauffage(surface,consommation_par_m2):
 return surface*consommation_par_m2
```

---

## Environnements virtuels

Les environnements virtuels permettent d'isoler les dépendances Python pour chaque projet.  
Cela évite les conflits de version et garantit que le projet fonctionne toujours avec les bonnes librairies.

---

### Vérifier Python et pip

Avant de créer un environnement virtuel, vérifiez que Python 3 est installé :

```bash
python --version
# ou
python -V
```

Vérifiez le chemin d'installation de Python :

```bash
where python   # Windows
which python   # Linux / MacOS
```

Listez les librairies Python déjà installées :

```bash
pip list
```

---

### Installer `virtualenv` (optionnel)

Python 3 inclut `venv`, mais `virtualenv` peut être utilisé pour plus de flexibilité :

```bash
pip install virtualenv
pip list  # vérifier que virtualenv est bien installé
```

---

### Créer un environnement virtuel

Positionnez-vous dans le dossier où vous souhaitez créer votre environnement :

```bash
cd C:\Users\asardell\Documents\my_virtual_envs
```

Créez l'environnement appelé `env-ademe` :

```bash
py -m venv env-ademe
```

- `-m venv` indique à Python d’exécuter le module `venv` comme script.
- ⚠️ **Ne pas créer l’environnement dans le dossier du projet** pour éviter d’inclure des fichiers temporaires.

---

### Activer et désactiver l’environnement

Windows (cmd) :

```bash
cd ./env-ademe/Scripts
activate.bat
```

Linux / MacOS (bash) :

```bash
source env-ademe/bin/activate
```

Pour désactiver l’environnement :

```bash
deactivate
```

Vérifiez que l’environnement est activé et les packages installés :

```bash
pip list
```

---

### Installer des librairies dans l’environnement

Exemple pour Pandas et NumPy :

```bash
pip install numpy pandas matplotlib
pip list  # vérifier l'installation
```

---

### Exporter les dépendances

Pour partager ou recréer un environnement, exportez les dépendances dans un fichier `requirements.txt` :

```bash
cd C:\Users\asardell\Documents\ademe
pip freeze > requirements.txt
```

- ⚠️ **Ne pas créer `requirements.txt` dans le dossier de l’environnement**, mais dans le dossier de votre projet.

Pour installer les dépendances sur un autre environnement :

```bash
pip install -r C:\Users\asardell\Documents\ademe\requirements.txt
```

---

### Créer un second environnement pour test

```bash
py -m venv test-env
source test-env/Scripts/activate  # ou activate.bat sous Windows
pip install -r requirements.txt
pip list  # vérifier que toutes les librairies sont bien installées
```

---

### Utiliser l’environnement virtuel dans VS Code

1. Ouvrez VS Code dans votre projet.
2. Ouvrez le terminal intégré (cmd ou PowerShell).
3. Activez votre environnement virtuel comme montré précédemment.
4. Vérifiez que VS Code utilise l’interpréteur Python de l’environnement activé.


:warning: Pour que VS Code détecte automatiquement vos environnements virtuels, il est conseillé de créer votre environnement virtuel dans votre dossier `ademe`.

:bulb:  Pour une utilisation de Jupyter dans VS Code, installer également `ipykernel` après avoir activité l'environnement `env-ademe` :

```cmd
pip install ipykernel
```