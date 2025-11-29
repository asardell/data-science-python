# Chapitre 1 : Introduction à la Data Science  

- [Chapitre 1 : Introduction à la Data Science](#chapitre-1--introduction-à-la-data-science)
  - [1. Qu’est-ce que la Data Science ?](#1-quest-ce-que-la-data-science-)
    - [Exemple concret (sur données ADEME — DPE)](#exemple-concret-sur-données-ademe--dpe)
  - [2. Différence entre statistiques et Data Science](#2-différence-entre-statistiques-et-data-science)
    - [Statistiques](#statistiques)
    - [Data Science](#data-science)
    - [Comparaison synthétique](#comparaison-synthétique)
  - [3. Champs d’application de la Data Science](#3-champs-dapplication-de-la-data-science)
  - [4. Outils et algorithmes pour faire de la Data Science](#4-outils-et-algorithmes-pour-faire-de-la-data-science)
    - [Langages](#langages)
    - [Libraries Python](#libraries-python)
    - [Écosystème Python pour la Data Science](#écosystème-python-pour-la-data-science)
      - [Environnements de développement](#environnements-de-développement)
      - [Alternatives et outils populaires](#alternatives-et-outils-populaires)
      - [Résumé pédagogique](#résumé-pédagogique)
  - [5. Les méthodes de Machine Learning](#5-les-méthodes-de-machine-learning)
    - [Apprentissage supervisé](#apprentissage-supervisé)
      - [Classification](#classification)
      - [Régression](#régression)
    - [Apprentissage non supervisé](#apprentissage-non-supervisé)
    - [Apprentissage par renforcement](#apprentissage-par-renforcement)


## 1. Qu’est-ce que la Data Science ?

La **Data Science** combine :  
- statistiques  
- programmation  
- connaissance métier  
- visualisation  

👉 Objectif : **transformer des données brutes en décisions**.

Trois grandes activités :  
1. **Comprendre** (exploration, visualisation)  
2. **Prédire** (modèles de machine learning)  
3. **Agir** (décision, automatisation)

### Exemple concret (sur données ADEME — DPE)
On peut :  
- détecter les bâtiments énergivores  
- prédire l’étiquette DPE  
- estimer les émissions CO₂  
- recommander des rénovations  

## 2. Différence entre statistiques et Data Science

### Statistiques  
- Approche **théorique** : on part souvent d’un modèle simple pour comprendre le phénomène.  
- Vise à **expliquer** les relations entre variables.  
- Hypothèses fortes sur les données (normalité, indépendance, linéarité...).  
- Modèles souvent **simples** et interprétables.

**Exemple concret** :  
*"En analysant 100 logements, on constate que l’isolation des murs explique 45 % de la variation de la consommation énergétique."*  
Ici, l’objectif est de comprendre, pas forcément de prédire la consommation future.

### Data Science  
- Approche **pratique et outillée** : on utilise des outils pour traiter de grandes quantités de données et obtenir des résultats rapidement.  
- Vise à **prédire** ou automatiser des décisions.  
- Peu d’hypothèses strictes sur la distribution des données.  
- Modèles souvent **complexes**, optimisés pour la performance, parfois moins interprétables.

**Exemple concret** :  
*"À partir de 200 variables issues des données DPE, prédire automatiquement l’étiquette énergétique d’un logement."*  
Ici, l’objectif est de produire une prédiction fiable, même si le modèle est complexe.

### Comparaison synthétique

| Statistiques                 | Data Science                               |
|-------------------------------|-------------------------------------------|
| Expliquer                     | Prédire                                   |
| Théorie                       | Industrie, pratique                        |
| Modèles simples               | Modèles complexes (ML, réseaux de neurones)|
| Petits échantillons           | Gros volumes de données                     |
| Hypothèses fortes             | Hypothèses légères ou adaptatives          |
| Compréhension des relations   | Performance et automatisation             |

💡 **Remarque pédagogique** :  
Les deux disciplines sont **complémentaires**. On utilise souvent la statistique pour comprendre et nettoyer les données avant d’appliquer des modèles de Data Science plus complexes.


## 3. Champs d’application de la Data Science

- **Industrie & énergie** → prédiction consommation, anomalies  
- **E-commerce** → recommandations  
- **Banque / assurance** → scoring, fraude  
- **Santé** → diagnostic assisté  
- **Automobile** → prédiction pannes, conduite autonome  
- **Grand public** → recherche, traduction, filtres photos

## 4. Outils et algorithmes pour faire de la Data Science

### Langages
- Python 
- R  
- SQL  

### Libraries Python
- **NumPy** (calcul)  
- **Pandas** (tables)  
- **Matplotlib / Seaborn** (visualisation)  
- **Scikit-learn** (machine learning)  
- **TensorFlow / PyTorch** (deep learning)  
- **Spark MLlib** (big data)

### Écosystème Python pour la Data Science

L’écosystème Python pour la Data Science est riche et diversifié. Il comprend à la fois des environnements pour écrire du code et des outils pour analyser, visualiser et manipuler les données.

#### Environnements de développement

- **Jupyter Notebook**  
  - Interface **interactive** dans le navigateur.  
  - Idéal pour : exploration de données, visualisation, prototypes rapides, documentation combinée avec le code.  
  - **Exemple** : charger le dataset DPE et visualiser la distribution des consommations énergétiques avec Matplotlib ou Seaborn dans une cellule.  
  - Avantage : possibilité de **combiner code, graphiques et explications textuelles** dans le même document.  
  - Limite : moins pratique pour organiser un projet de code complexe ou pour production.

- **VSCode / PyCharm / autres IDE**  
  - Environnements de développement **classiques**.  
  - Idéal pour : scripts Python modulaires, projets structurés, intégration avec Git, tests unitaires, déploiement.  
  - **Exemple** : écrire un script `dpe_analysis.py` qui lit le dataset DPE, nettoie les données et produit un fichier CSV ou Parquet.  
  - Avantage : meilleure organisation du code, débogage, gestion des dépendances.

#### Alternatives et outils populaires

- **Google Colab**  
  - Similaire à Jupyter Notebook mais **100 % cloud**, pas besoin d’installation locale.  
  - Partage facile des notebooks et accès à GPU gratuit pour modèles plus lourds.  

- **Spyder**  
  - IDE orienté Data Science, très proche de MATLAB.  
  - Bon pour débuter avec Python scientifique.

- **RStudio (pour R)**  
  - Si vous utilisez R pour certaines analyses statistiques ou visualisations.  
  - R et Python peuvent être complémentaires.

#### Résumé pédagogique

| Outil | Cas d’usage | Points forts | Points faibles |
|-------|------------|-------------|----------------|
| Jupyter Notebook | Exploration, prototypage, visualisation | Interactif, combine code et texte | Difficultés pour projets complexes |
| VSCode / PyCharm | Scripts, projets structurés, production | Organisation, débogage, gestion Git | Moins interactif pour visualisation rapide |
| Google Colab | Cloud, GPU, partage | Facile à démarrer, gratuit | Dépendance Internet, ressources limitées |
| Spyder | Analyse scientifique locale | Interface simple, intégration SciPy | Moins populaire pour projets complexes |

💡 **Conseil pédagogique** :  
Pour débuter, utilisez **Jupyter Notebook** pour comprendre les concepts et manipuler les données.  
Pour des projets structurés ou destinés à la production, préférez **VSCode** ou un autre IDE.

## 5. Les méthodes de Machine Learning

Le **Machine Learning (ML)** regroupe des techniques permettant à un programme d’apprendre à partir de données.  
Pour les algorithmes classiques supervisés ou non supervisés, les données doivent être **tabulaires** :  

- Chaque **ligne** représente une observation (un bâtiment, un client…).  
- Chaque **colonne** représente une variable/feature, quantitative ou qualitative.  

💡 Remarque : Pour des données non tabulaires comme des images, textes ou sons, il faut d’abord les **transformer en vecteurs de features** que l’algorithme pourra exploiter.  

Selon que l’on dispose ou non d’une cible à prédire, les algorithmes se classent en trois grandes familles : **supervisé, non supervisé, et par renforcement**.

### Apprentissage supervisé

**Objectif : prédire une valeur ou une catégorie à partir d’exemples annotés.**  

On dispose d’un **dataset avec les entrées (features)** et les **résultats connus (labels)**. L’algorithme apprend une relation entre les deux.

#### Classification
→ prédire une **catégorie**  

- Exemples génériques :  
  - Spam / Pas spam  
  - Type de logement (maison / appartement)  

- Exemple ADEME :  
  - Prédire `etiquette_dpe` (A, B, C…) à partir de :  
    - `annee_construction`  
    - `surface_habitable`  
    - `qualite_isolation_murs`  
    - `type_installation_chauffage_n1`  
    - `type_energie_principale_chauffage`  

#### Régression
→ prédire une **valeur numérique**  

- Exemples génériques :  
  - Prix d’une maison  
  - Consommation d'énergie 

- Exemple ADEME :  
  - Prédire `conso_chauffage_ef` ou `emission_ges_chauffage` à partir des mêmes variables que ci-dessus  

💡 **Méthodes courantes supervisées** : Régression linéaire, régression logistique, KNN, arbres de décision, Random Forest, SVM.

### Apprentissage non supervisé

**Objectif : découvrir une structure cachée dans les données**  
Aucune variable cible n’est fournie. L’algorithme cherche à **regrouper, résumer ou détecter des anomalies**.

Clustering
- Exemple : K-means  
- Objectif ADEME : regrouper des bâtiments ayant des profils énergétiques similaires, par exemple pour identifier les bâtiments les plus énergivores.

Réduction de dimension
- Exemple : PCA (Principal Component Analysis)  
- Objectif ADEME : résumer 200+ variables DPE en quelques axes principaux pour visualisation ou analyse exploratoire.

### Apprentissage par renforcement

**Objectif : apprendre à agir dans un environnement en recevant des récompenses ou pénalités.**  

- Le modèle teste des actions, observe les conséquences et ajuste sa stratégie pour maximiser la récompense cumulée.  
- **Exemples génériques :** jeux vidéo (AlphaGo), robotique, trading automatique.  
- **Exemple ADEME / énergétique :** piloter un chauffage intelligent dans un bâtiment pour minimiser la consommation tout en maintenant le confort.

Résumé final

| Type | But | Exemple ADEME |
|------|-----|---------------|
| Supervisé | Prédire | étiquette DPE, consommation, émissions |
| Non supervisé | Explorer / regrouper | Clustering de bâtiments, détection d’anomalies |
| Renforcement | Optimiser | Gestion intelligente du chauffage, stratégie énergétique |

💡 **Conseil pédagogique** :  
- Commencer par supervisé pour prédiction simple et intuitive.  
- Explorer non supervisé pour comprendre les patterns cachés.  
- Introduire le renforcement pour montrer comment un système peut apprendre par essais/erreurs.
