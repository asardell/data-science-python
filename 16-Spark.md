# Chapitre 16 :  Introduction à Apache Spark

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/f/f3/Apache_Spark_logo.svg/langfr-330px-Apache_Spark_logo.svg.png" alt="Source de l'image" width="600"/>
</p>


- [Chapitre 16 :  Introduction à Apache Spark](#chapitre-16---introduction-à-apache-spark)
- [Pourquoi Spark ?](#pourquoi-spark-)
  - [Limites de pandas / scikit-learn](#limites-de-pandas--scikit-learn)
  - [Avantages de Spark pour les données DPE](#avantages-de-spark-pour-les-données-dpe)
- [Architecture de Spark (simple mais essentielle)](#architecture-de-spark-simple-mais-essentielle)
  - [Driver](#driver)
  - [Executors](#executors)
  - [Cluster manager](#cluster-manager)
- [RDD et DataFrames distribués](#rdd-et-dataframes-distribués)
- [Transformations vs Actions](#transformations-vs-actions)
  - [Transformations (lazy = n’exécutent rien)](#transformations-lazy--nexécutent-rien)
  - [Actions (déclenchent l’exécution)](#actions-déclenchent-lexécution)
  - [Gestion des valeurs manquantes](#gestion-des-valeurs-manquantes)
    - [Remplir les valeurs manquantes](#remplir-les-valeurs-manquantes)
    - [Suppression des lignes très incomplètes](#suppression-des-lignes-très-incomplètes)
    - [Imputation automatique](#imputation-automatique)
  - [Détection et suppression des doublons](#détection-et-suppression-des-doublons)
  - [Encodage des variables catégorielles](#encodage-des-variables-catégorielles)
  - [Assemblage des features](#assemblage-des-features)
- [Gestion distribuée de la charge : partitions](#gestion-distribuée-de-la-charge--partitions)
  - [Vérifier le nombre de partitions](#vérifier-le-nombre-de-partitions)
  - [Adapter le parallélisme](#adapter-le-parallélisme)
- [Exécution distribuée : ce qui se passe vraiment](#exécution-distribuée--ce-qui-se-passe-vraiment)
- [Résumé visuel du fonctionnement](#résumé-visuel-du-fonctionnement)
- [Pourquoi Spark est indispensable pour les DPE massifs ?](#pourquoi-spark-est-indispensable-pour-les-dpe-massifs-)
  - [Conclusion](#conclusion)
- [Spark MLlib](#spark-mllib)
  - [Introduction à MLlib](#introduction-à-mllib)
  - [Les composants de MLlib](#les-composants-de-mllib)
    - [Transformers](#transformers)
    - [Estimators](#estimators)
    - [Pipelines](#pipelines)
    - [Evaluators](#evaluators)
  - [Préparation des données DPE](#préparation-des-données-dpe)
  - [Pipeline MLlib pour classification](#pipeline-mllib-pour-classification)
    - [Encodage des catégories](#encodage-des-catégories)
    - [Assemblage des features](#assemblage-des-features-1)
    - [Modèle Random Forest (classification)](#modèle-random-forest-classification)
    - [Pipeline complet](#pipeline-complet)
    - [Entraînement](#entraînement)
    - [Évaluation classification](#évaluation-classification)
  - [Pipeline MLlib pour régression](#pipeline-mllib-pour-régression)
  - [Objectif : prédire `consommation_energie`](#objectif--prédire-consommation_energie)
    - [Préparation du pipeline](#préparation-du-pipeline)
    - [Entraînement](#entraînement-1)
    - [Évaluation régression](#évaluation-régression)
  - [Importance des features dans Random Forest](#importance-des-features-dans-random-forest)
  - [Points forts de MLlib](#points-forts-de-mllib)
  - [Quand utiliser MLlib ?](#quand-utiliser-mllib-)
  - [Résumé](#résumé)


# Pourquoi Spark ?

Spark est un moteur de calcul distribué utilisé lorsqu’on dépasse les limites de Python/pandas.

## Limites de pandas / scikit-learn

- Tout tient en **mémoire RAM du PC**  
- Lent dès qu’on dépasse **quelques millions de lignes**  
- Impossible de paralléliser facilement  
- Les modèles ML ne sont **pas distribués**

## Avantages de Spark pour les données DPE

Imagine un fichier DPE national contenant **20 millions de logements** :  
✔ trop gros pour pandas  
✔ temps de calcul énorme en local  
✔ preprocessing lent (OneHotEncoder, imputation, etc.)

Spark permet :

- lire et traiter des fichiers massifs  
- distribuer automatiquement les calculs  
- gérer le ML sur cluster avec MLlib  
- éviter les problèmes de mémoire  
- paralléliser automatiquement toutes les étapes  


# Architecture de Spark (simple mais essentielle)

Spark s’exécute selon une structure à trois niveaux.

## Driver

- C’est ton programme Python (PySpark).  
- Il construit le **DAG** des opérations.  
- Il envoie les instructions au cluster.

## Executors

- Ce sont les "workers" : les machines qui travaillent réellement.  
- Ils exécutent les transformations sur les données DPE.  
- Ils stockent en mémoire les partitions.

## Cluster manager

Exemples : Spark Standalone, YARN, Kubernetes.  
Il distribue les ressources (CPU, RAM).


# RDD et DataFrames distribués

Spark ne traite jamais un fichier entier.  
Il découpe automatiquement les données en **partitions**.

Exemple :  
Un fichier DPE de 10 millions de lignes peut devenir :

- Partition 1 → 500 000 lignes  
- Partition 2 → 500 000 lignes  
- ...  
- Partition 20 → 500 000 lignes  

Chaque partition est envoyée à un executor pour traitement en parallèle.  
Cela explique pourquoi Spark est **massivement plus rapide**.


# Transformations vs Actions

## Transformations (lazy = n’exécutent rien)

Exemples :

- `filter`  
- `select`  
- `withColumn`  
- `groupBy`  
- `dropDuplicates`  
- `fillna`

Exemple :

```python
df = df.filter(df.surface_habitable > 0)
```

Rien n’est exécuté.  
Spark prépare juste les opérations dans un DAG.

## Actions (déclenchent l’exécution)

- `show()`  
- `count()`  
- `collect()`  
- `write`

Exemple :

```python
df.count()
```

Spark exécute finalement tout le DAG.

réparation des données DPE avec Spark

Voici les opérations les plus importantes pour tout préprocessing distribué.

## Gestion des valeurs manquantes

### Remplir les valeurs manquantes

```python
df = df.fillna({
    "surface_habitable": 0,
    "annee_construction": 1900
})
```

### Suppression des lignes très incomplètes

```python
df = df.dropna(thresh=3)
```

### Imputation automatique

```python
from pyspark.ml.feature import Imputer

imputer = Imputer(
    inputCols=["surface_habitable", "annee_construction"],
    outputCols=["surface_habitable_imp", "annee_construction_imp"]
)

df = imputer.fit(df).transform(df)
```


## Détection et suppression des doublons
s utile dans les datasets DPE où des adresses peuvent être dupliquées.

```python
df = df.dropDuplicates(["adresse", "surface_habitable", "annee_construction"])
```


## Encodage des variables catégorielles


- `StringIndexer` → transforme textes en indices  
- `OneHotEncoder` → vectorise  

Exemple pour le DPE :

```python
from pyspark.ml.feature import StringIndexer, OneHotEncoder

indexer = StringIndexer(
    inputCol="energie_principale",
    outputCol="energie_indexed"
)

encoder = OneHotEncoder(
    inputCol="energie_indexed",
    outputCol="energie_encoded"
)

df = indexer.fit(df).transform(df)
df = encoder.fit(df).transform(df)
```


## Assemblage des features

Spark exige une **seule colonne features** via `VectorAssembler`.
python
from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(
    inputCols=[
        "surface_habitable_imp",
        "annee_construction_imp",
        "energie_encoded"
    ],
    outputCol="features"
)

df = assembler.transform(df)
```


# Pipelines Spark

Les pipelines Spark permettent :

- d’enchaîner les étapes (imputation → indexation → encodage → assemblage)  
- de réutiliser le flux complet  
- d’appliquer la même transformation au train/test  

Exemple pipeline DPE :

```python
from pyspark.ml import Pipeline

pipeline = Pipeline(stages=[
    imputer,
    indexer,
    encoder,
    assembler
])

model = pipeline.fit(df)
df_final = model.transform(df)
```


# Gestion distribuée de la charge : partitions

## Vérifier le nombre de partitions

```python
df.rdd.getNumPartitions()
```

## Adapter le parallélisme

```python
df = df.repartition(32)
```

Plus de partitions :

- plus de parallélisme  
- meilleur équilibre de charge  

Trop de partitions = overhead de communication.


# Exécution distribuée : ce qui se passe vraiment

Lorsqu’on écrit :

```python
df = df.filter(df.surface_habitable > 0)
df = df.withColumn("ratio", df.consommation_energie / df.surface_habitable)
df.show()
```

Spark :

1. crée un DAG de transformations  
2. découpe les données en partitions  
3. envoie les partitions aux executors  
4. chaque executor exécute sa partie  
5. Spark renvoie l’échantillon du `show()` au driver  


# Résumé visuel du fonctionnement

```
             DRIVER
   (construit le DAG, orchestre)

        |  (instructions)
        v

   +--------- Spark Cluster ---------+
   |                                 |
   |   Executor 1    Executor 2      |
   |   Partition 1     Partition 2   |
   |   Partition 3     Partition 4   |
   |                                 |
   +---------------------------------+
```


# Pourquoi Spark est indispensable pour les DPE massifs ?

Le DPE national représente **des dizaines de millions de logements**.

Les traitements :

- imputations  
- jointures départementales  
- encodages massifs  
- regroupements par région  

prennent des heures sur pandas…

… mais quelques **minutes** sur Spark.


## Conclusion

Spark est indispensable pour :

- traiter de très gros datasets DPE  
- paralléliser le préprocessing  
- gérer automatiquement la mémoire  
- utiliser un pipeline ML scalable  
- distribuer l’encodage, l’imputation et les transformations  


# Spark MLlib  

Ce cours présente **Spark MLlib**, la bibliothèque de Machine Learning de Spark, et montre comment l’utiliser tout en utilisant des pipelines Spark, exactement comme en scikit-learn, mais en version distribuée.


## Introduction à MLlib

MLlib est le module Machine Learning de Spark.  
Contrairement à scikit-learn :

- Les données sont **distribuées** sur plusieurs machines  
- Les algorithmes sont conçus pour être **scalables**  
- Le préprocessing est géré via des **pipelines distribués**  
- Le format d’entrée est **VectorAssembler → features**  

MLlib est utilisé quand les datasets dépassent la capacité RAM d’un ordinateur.  
Parfait pour les données DPE nationales (plusieurs dizaines de millions de lignes).


## Les composants de MLlib

MLlib fournit quatre briques essentielles :

### Transformers  
Transforment les données sans apprentissage.  
Ex :
- `StringIndexer`
- `OneHotEncoder`
- `StandardScaler`
- `VectorAssembler`

### Estimators  
Ce sont les modèles.  
Ex :
- `RandomForestClassifier`
- `RandomForestRegressor`
- `LogisticRegression`
- `GBTClassifier`

### Pipelines  
Chaînent les étapes comme en scikit-learn.

### Evaluators  
Évaluent les performances :
- `MulticlassClassificationEvaluator`
- `RegressionEvaluator`


## Préparation des données DPE

Dans l'exemple, on suppose disposer d’un DataFrame Spark :

- `surface_habitable`
- `annee_construction`
- `type_batiment`
- `energie_principale`
- `etiquette_energie` (A–G → classification)
- `consommation_energie` (régression)


## Pipeline MLlib pour classification  

### Encodage des catégories

```python
from pyspark.ml.feature import StringIndexer, OneHotEncoder

indexer_bat = StringIndexer(inputCol="type_batiment", outputCol="bat_index")
indexer_ener = StringIndexer(inputCol="energie_principale", outputCol="ener_index")
indexer_label = StringIndexer(inputCol="etiquette_energie", outputCol="label")

encoder = OneHotEncoder(
    inputCols=["bat_index", "ener_index"],
    outputCols=["bat_vec", "ener_vec"]
)
```


### Assemblage des features

```python
from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(
    inputCols=[
        "surface_habitable",
        "annee_construction",
        "bat_vec",
        "ener_vec"
    ],
    outputCol="features"
)
```


### Modèle Random Forest (classification)

```python
from pyspark.ml.classification import RandomForestClassifier

rf_clf = RandomForestClassifier(
    featuresCol="features",
    labelCol="label",
    numTrees=100,
    maxDepth=10
)
```


### Pipeline complet

```python
from pyspark.ml import Pipeline

pipeline_clf = Pipeline(stages=[
    indexer_bat,
    indexer_ener,
    indexer_label,
    encoder,
    assembler,
    rf_clf
])
```


### Entraînement

```python
model_clf = pipeline_clf.fit(train_df)
pred_clf = model_clf.transform(test_df)
```


### Évaluation classification

```python
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy"
)

accuracy = evaluator.evaluate(pred_clf)
print("Accuracy :", accuracy)
```


## Pipeline MLlib pour régression  
## Objectif : prédire `consommation_energie`

Les étapes sont les mêmes sauf :

- `labelCol = "consommation_energie"`
- Le modèle est un `RandomForestRegressor`


### Préparation du pipeline

```python
from pyspark.ml.regression import RandomForestRegressor

rf_reg = RandomForestRegressor(
    featuresCol="features",
    labelCol="consommation_energie",
    numTrees=200,
    maxDepth=12
)

pipeline_reg = Pipeline(stages=[
    indexer_bat,
    indexer_ener,
    encoder,
    assembler,
    rf_reg
])
```


### Entraînement

```python
model_reg = pipeline_reg.fit(train_df)
pred_reg = model_reg.transform(test_df)
```


### Évaluation régression

```python
from pyspark.ml.evaluation import RegressionEvaluator

evaluator_rmse = RegressionEvaluator(
    labelCol="consommation_energie",
    predictionCol="prediction",
    metricName="rmse"
)

evaluator_r2 = RegressionEvaluator(
    labelCol="consommation_energie",
    predictionCol="prediction",
    metricName="r2"
)

print("RMSE :", evaluator_rmse.evaluate(pred_reg))
print("R2 :", evaluator_r2.evaluate(pred_reg))
```


## Importance des features dans Random Forest

MLlib permet d’extraire l’importance des attributs.

```python
rf_model = model_reg.stages[-1]  # dernier élément du pipeline

importances = rf_model.featureImportances
print(importances)
```

Attention :  
Le `VectorAssembler` compacte les features → les indices ne correspondent plus aux noms directement.

Il faut reconstituer :

```python
assembler_input = assembler.getInputCols()

for name, score in zip(assembler_input, importances):
    print(name, ":", score)
```


## Points forts de MLlib

| Fonctionnalité | Explication |
|----------------|-------------|
| **Distribué** | Préprocessing + apprentissage sur cluster |
| **Lazy execution** | Transformation sans exécution tant qu’aucune action |
| **Pipelines** | Même logique que scikit-learn mais scalable |
| **Scalable ML** | RandomForest et GBT pour des millions de lignes |
| **Compatibilité DataFrames** | Manipulation facile |


## Quand utiliser MLlib ?

MLlib est l’outil adapté si :

- ton dataset dépasse **1–2 millions de lignes**
- les fichiers sources sont très volumineux (DPE national)
- tu veux paralléliser le calcul sans effort
- tu veux entraîner des modèles rapides et robustes


## Résumé

| Tâche | Modèle MLlib | Objectif |
|------|--------------|----------|
| **Classification DPE (A–G)** | RandomForestClassifier | Prédiction qualitative |
| **Régression de la consommation** | RandomForestRegressor | Prédiction quantitative (kWh/m²/an) |

MLlib est la version **massive**, **distribuée**, **scalable** du machine learning, parfaitement adaptée aux données ADEME (DPE).



