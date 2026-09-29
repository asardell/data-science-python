# Chapitre 1 : Introduction à la Data Science  

- [Chapitre 1 : Introduction à la Data Science](#chapitre-1--introduction-à-la-data-science)
  - [Qu’est-ce que la Data Science ?](#quest-ce-que-la-data-science-)
    - [Exemple concret (sur données ADEME — DPE)](#exemple-concret-sur-données-ademe--dpe)
  - [Différence entre statistiques et Data Science](#différence-entre-statistiques-et-data-science)
    - [Statistiques](#statistiques)
    - [Data Science](#data-science)
    - [Comparaison synthétique](#comparaison-synthétique)
  - [Champs d’application de la Data Science](#champs-dapplication-de-la-data-science)
  - [Outils et algorithmes pour faire de la Data Science](#outils-et-algorithmes-pour-faire-de-la-data-science)
    - [Langages](#langages)
    - [Libraries Python](#libraries-python)
    - [Écosystème Python pour la Data Science](#écosystème-python-pour-la-data-science)
      - [Environnements de développement](#environnements-de-développement)
      - [Alternatives et outils populaires](#alternatives-et-outils-populaires)
      - [Résumé pédagogique](#résumé-pédagogique)
  - [Les méthodes de Machine Learning](#les-méthodes-de-machine-learning)
    - [Apprentissage supervisé](#apprentissage-supervisé)
      - [Classification](#classification)
      - [Régression](#régression)
    - [Apprentissage non supervisé](#apprentissage-non-supervisé)
    - [Apprentissage par renforcement](#apprentissage-par-renforcement)
  - [IA, Machine Learning et Deep Learning](#ia-machine-learning-et-deep-learning)
    - [Intelligence Artificielle (IA)](#intelligence-artificielle-ia)
    - [Machine Learning (ML)](#machine-learning-ml)
    - [Deep Learning (DL)](#deep-learning-dl)
  - [Liens utiles :](#liens-utiles-)


## Qu’est-ce que la Data Science ?

La **Data Science** combine :  
- statistiques  
- programmation  
- connaissance métier  
- visualisation  

👉 Objectif : **transformer des données brutes en décisions**.

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/3/38/Data_Science.png" alt="Source de l'image" width="600"/>
</p>

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

## Différence entre statistiques et Data Science

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


## Champs d’application de la Data Science

<p align="center">
  <img src="https://media.geeksforgeeks.org/wp-content/cdn-uploads/20200103174349/11-Industries-That-Benefits-the-Most-From-Data-Science.png" alt="Source de l'image" width="600"/>
</p>


- **Industrie & énergie** → prédiction consommation, anomalies  
- **E-commerce** → recommandations  
- **Banque / assurance** → scoring, fraude  
- **Santé** → diagnostic assisté  
- **Automobile** → prédiction pannes, conduite autonome  
- **Grand public** → recherche, traduction, filtres photos

## Outils et algorithmes pour faire de la Data Science

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

<p align="center">
  <img src="https://docs.jupyter.org/en/latest/_images/jupyterlab.png" alt="Source de l'image" width="600"/>
</p>

- **VSCode / PyCharm / autres IDE**  
  - Environnements de développement **classiques**.  
  - Idéal pour : scripts Python modulaires, projets structurés, intégration avec Git, tests unitaires, déploiement.  
  - **Exemple** : écrire un script `dpe_analysis.py` qui lit le dataset DPE, nettoie les données et produit un fichier CSV ou Parquet.  
  - Avantage : meilleure organisation du code, débogage, gestion des dépendances.


<p align="center">
  <img src="https://code.visualstudio.com/assets/docs/python/editing/sortImports.gif" alt="Source de l'image" width="600"/>
</p>


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

## Les méthodes de Machine Learning

Le **Machine Learning (ML)** regroupe des techniques permettant à un programme d’apprendre à partir de données.  
Pour les algorithmes classiques supervisés ou non supervisés, les données doivent être **tabulaires** :  

- Chaque **ligne** représente une observation (un bâtiment, un client…).  
- Chaque **colonne** représente une variable/feature, quantitative ou qualitative.  

💡 Remarque : Pour des données non tabulaires comme des images, textes ou sons, il faut d’abord les **transformer en vecteurs de features** que l’algorithme pourra exploiter.  

Selon que l’on dispose ou non d’une cible à prédire, les algorithmes se classent en trois grandes familles : **supervisé, non supervisé, et par renforcement**.

<p align="center">
  <img src="https://i0.wp.com/deeplylearning.fr/wp-content/uploads/2018/09/type-of-learning.png?resize=781%2C558&ssl=1" alt="Source de l'image" width="600"/>
</p>


### Apprentissage supervisé

**Objectif : prédire une valeur ou une catégorie à partir d’exemples annotés.**  

On dispose d’un **dataset avec les entrées (features)** et les **résultats connus (labels)**. L’algorithme apprend une relation entre les deux.

<p align="center">
  <img src="https://storage.googleapis.com/algodailyrandomassets/curriculum/machine-learning/ml-interview-questions/classification%20and%20regression.jpeg" alt="Source de l'image" width="600"/>
</p>


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

<p align="center">
  <img src="https://scikit-learn.org/stable/_images/sphx_glr_plot_cluster_comparison_001.png" alt="Source de l'image" width="600"/>
</p>


Clustering
- Exemple : K-means  
- Objectif ADEME : regrouper des bâtiments ayant des profils énergétiques similaires, par exemple pour identifier les bâtiments les plus énergivores.

Réduction de dimension
- Exemple : PCA (Principal Component Analysis)  
- Objectif ADEME : résumer 200+ variables DPE en quelques axes principaux pour visualisation ou analyse exploratoire.

### Apprentissage par renforcement

**Objectif : apprendre à agir dans un environnement en recevant des récompenses ou pénalités.**  

<p align="center">
  <img src="https://miro.medium.com/v2/resize:fit:1400/1*7MxNJJ9IRn9R4tf31DaCLg.png" width="600"/>
</p>



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


## IA, Machine Learning et Deep Learning

Ces trois termes sont liés, mais ils ne désignent pas la même chose.

<p align="center">
  <img src="https://cdn.prod.website-files.com/65686931f8b0bb6aed626331/673da085af8edf86b7ae565e_673da0585cb9c6370eb2ceb1_aiko%2520article%25204.png" alt="Source de l'image" width="600"/>
</p>


### Intelligence Artificielle (IA)

L’**IA** est le domaine général qui cherche à créer des systèmes capables de réaliser des tâches nécessitant normalement une forme d’intelligence humaine.

Exemples :
- comprendre un texte
- reconnaître une image
- prendre une décision
- jouer à un jeu
- générer du contenu

### Machine Learning (ML)

Le **Machine Learning** est une sous-partie de l’IA.

Au lieu de programmer explicitement toutes les règles, on donne au modèle des **données** à partir desquelles il apprend des relations et peut ensuite faire des prédictions ou prendre des décisions.

Exemples :
- régression linéaire
- arbres de décision
- Random Forest
- KNN


### Deep Learning (DL)

Le **Deep Learning** est une sous-partie du Machine Learning basée sur des **réseaux de neurones artificiels comportant plusieurs couches**.

À l’origine, les réseaux de neurones reposaient notamment sur le **perceptron**, un neurone artificiel simple. En empilant plusieurs neurones et plusieurs couches, on obtient des réseaux capables d'apprendre des représentations beaucoup plus complexes.

Le Deep Learning s'est ensuite décliné en différentes architectures adaptées aux problèmes rencontrés :

- **CNN** → images et vision
- **RNN / LSTM** → séquences et séries temporelles
- **Transformers** → texte, langage, images, etc.

<p align="center">
  <img src="https://miro.medium.com/v2/resize:fit:1100/format:webp/1*cuTSPlTq0a_327iTPJyD-Q.png" alt="Source de l'image" width="600"/>
</p>



## Liens utiles :

- [Les roles data](https://learn.microsoft.com/fr-fr/training/modules/data-analytics-microsoft/3-roles?ns-enrollment-type=learningpath&ns-enrollment-id=learn-bizapps.get-started-data-analytics)