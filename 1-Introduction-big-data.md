# Chapitre 1 : Big Data et Cloud

- [Chapitre 1 : Big Data et Cloud](#chapitre-1--big-data-et-cloud)
  - [Qu’est-ce que le Big Data ?](#quest-ce-que-le-big-data-)
    - [Les 3 V](#les-3-v)
    - [Les 5 V](#les-5-v)
    - [Les 42 V de Big Data et Data Science](#les-42-v-de-big-data-et-data-science)
  - [Cloud Computing : IaaS, PaaS et SaaS](#cloud-computing--iaas-paas-et-saas)
    - [IaaS — Infrastructure as a Service](#iaas--infrastructure-as-a-service)
    - [PaaS — Platform as a Service](#paas--platform-as-a-service)
    - [SaaS — Software as a Service](#saas--software-as-a-service)
  - [L’écosystème Hadoop](#lécosystème-hadoop)
  - [On-premise, Cloud et Hybrid Cloud](#on-premise-cloud-et-hybrid-cloud)
    - [On-premise](#on-premise)
    - [Full Cloud](#full-cloud)
    - [Hybrid Cloud](#hybrid-cloud)
    - [Résumé](#résumé)
  - [Introduction à la notion de data lake](#introduction-à-la-notion-de-data-lake)
    - [Acteurs du marché](#acteurs-du-marché)
    - [Concepts clés](#concepts-clés)
      - [Buckets](#buckets)
      - [Stockage de fichiers vs SGBD](#stockage-de-fichiers-vs-sgbd)
      - [Data Lake et couches Medaillon](#data-lake-et-couches-medaillon)
      - [Formats de fichiers](#formats-de-fichiers)
      - [Pourquoi le partitionnement est essentiel en Data Lake ?](#pourquoi-le-partitionnement-est-essentiel-en-data-lake-)
      - [Cardinalité : notion clé](#cardinalité--notion-clé)
      - [Bonne pratique](#bonne-pratique)
      - [Mauvais exemple](#mauvais-exemple)
      - [Pourquoi c’est stratégique en entreprise ?](#pourquoi-cest-stratégique-en-entreprise-)
      - [Astuces : Lecture d'un fichier parquet](#astuces--lecture-dun-fichier-parquet)
    - [Apache Iceberg](#apache-iceberg)
      - [Qu’est-ce qu’Apache Iceberg ?](#quest-ce-quapache-iceberg-)
      - [Pourquoi ACID est important ?](#pourquoi-acid-est-important-)
      - [Pourquoi Iceberg est intéressant](#pourquoi-iceberg-est-intéressant)
      - [Cas d’usage typique](#cas-dusage-typique)
    - [Conclusion](#conclusion)
  - [Liens utiles](#liens-utiles)


## Qu’est-ce que le Big Data ?

Le **Big Data** désigne des données dont la **quantité, la vitesse de production ou la diversité** rendent leur traitement difficile avec des outils classiques.

### Les 3 V

Les premiers modèles du Big Data reposent sur **3 caractéristiques principales** : **Volume, Velocity, Variety**.

| V | Définition | Exemple |
|---|---|---|
| **Volume** | Quantité très importante de données | Des milliards de transactions ou de fichiers |
| **Velocity** | Vitesse de production et de traitement des données | Données IoT produites chaque seconde |
| **Variety** | Diversité des formats de données | Texte, image, vidéo, JSON, données SQL |

**Exemple : Netflix**

- **Volume** → historique de visionnage de millions d'utilisateurs
- **Velocity** → événements générés en temps réel pendant le visionnage
- **Variety** → vidéos, profils utilisateurs, logs, avis, métadonnées

### Les 5 V

Le modèle a ensuite été enrichi avec deux nouveaux V :

| V | Définition | Exemple |
|---|---|---|
| **Volume** | Quantité de données | Plusieurs pétaoctets de logs |
| **Velocity** | Vitesse de génération / traitement | Flux de données en temps réel |
| **Variety** | Diversité des formats | Texte, image, vidéo, JSON |
| **Veracity** | Fiabilité / qualité des données | Données erronées, incomplètes ou bruitées |
| **Value** | Valeur que l'on peut tirer des données | Détecter une fraude ou recommander un produit |

👉 Les **5 V** permettent donc de décrire à la fois les **caractéristiques techniques** des données et leur **qualité / utilité**.

---

### Les 42 V de Big Data et Data Science

Les 5 V ne sont pas une définition universelle et définitive.

En 2017, **KDnuggets** a [publié une liste humoristique](https://www.kdnuggets.com/2017/04/42-vs-big-data-data-science.html) de **42 V** du Big Data et de la Data Science.

Elle reprend les V classiques et ajoute de nombreux termes comme :

- **Vagueness** → ambiguïté des données
- **Version Control** → gestion des versions
- **Visualization** → visualisation
- **Volume** → quantité
- **Voodoo** → aspect parfois "magique" attribué à la Data Science

## Cloud Computing : IaaS, PaaS et SaaS

Le **Cloud Computing** permet d'utiliser des ressources informatiques à distance, généralement à la demande et selon la consommation.

Les [trois modèles classiques](https://media.geeksforgeeks.org/wp-content/uploads/20260225110549301390/evolution_of_cloud_computing.webp) sont **IaaS, PaaS et SaaS**.

### IaaS — Infrastructure as a Service

Le fournisseur fournit l'**infrastructure** :

- machines virtuelles
- stockage
- réseau
- CPU / GPU

L'entreprise gère ensuite son système d'exploitation et ses applications.

**Exemples :**

- **AWS**
- **Microsoft Azure** 
- **Google Cloud (GCP)**
- **Alibaba Cloud**
- **OVHcloud**

<p align="center">
  <img src="https://srgresearch-public.s3.amazonaws.com/uploads/CIS_shares.jpg" alt="Source de l'image" width="600"/>
</p>
 

👉 **IaaS = on loue l'infrastructure.**

### PaaS — Platform as a Service

Le fournisseur gère davantage l'infrastructure et fournit une **plateforme prête à utiliser**.

Le développeur se concentre principalement sur son application ou son traitement de données.

**Exemples d'outil pour mettre en place une data platform :**

- **Snowflake**
- **Databricks**
- **Fabric**

<p align="center">
  <img src="https://substackcdn.com/image/fetch/$s_!uvxD!,w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ffa2b3cce-411b-4a96-8847-0e940c7fbf7c_3799x1895.png" alt="Source de l'image" width="600"/>
</p>

Plus d'info [ici](https://www.blog.datawithbaraa.com/p/the-big-three-databricks-vs-snowflake).

👉 **PaaS = on loue une plateforme prête à développer / traiter.**

### SaaS — Software as a Service

Le fournisseur fournit directement une **application complète**.

L'utilisateur n'a pas à gérer les serveurs, l'infrastructure ou l'installation du logiciel.

**Exemples :**

- Gmail
- Microsoft 365
- Salesforce

👉 **SaaS = on utilise directement le logiciel.**


## L’écosystème Hadoop

Le **Big Data moderne a été fortement influencé par l’écosystème Hadoop**.

Hadoop est un framework open source conçu pour le **stockage et le traitement distribué de grandes quantités de données**.

Ses composants historiques principaux sont :

<p align="center">
  <img src="https://editor.analyticsvidhya.com/uploads/11586Hadoop-Ecosystem-1.png" alt="Source de l'image" width="600"/>
</p>

D'autres technologies se sont ensuite développées autour de cet écosystème, notamment Spark.

## On-premise, Cloud et Hybrid Cloud

### On-premise

L'entreprise **possède et exploite elle-même** ses serveurs.

**Exemple :** une entreprise possède ses propres serveurs Hadoop dans ses datacenters.

**Avantages :**

- contrôle complet de l'infrastructure
- maîtrise des données
- personnalisation importante
- pas de dépendance directe à un fournisseur Cloud

**Inconvénients :**

- investissement initial important
- maintenance à la charge de l'entreprise
- capacité limitée par le matériel disponible
- montée en charge plus difficile

### Full Cloud

L'infrastructure est entièrement hébergée chez un fournisseur Cloud.

**Exemple :** une entreprise utilise AWS, Azure ou GCP pour son stockage, son calcul et ses services Data.

**Avantages :**

- pas besoin d'acheter les serveurs
- mise à l'échelle rapide
- nombreux services managés disponibles
- paiement selon l'utilisation

**Inconvénients :**

- coûts pouvant devenir importants à grande échelle
- dépendance au fournisseur
- dépendance au réseau / Internet
- contraintes de localisation et de gouvernance des données

### Hybrid Cloud

Une partie de l'infrastructure reste **on-premise** et une autre est dans le **Cloud**.

**Exemple :**

Une banque peut conserver certaines données sensibles **on-premise**, tout en utilisant **AWS / Azure / GCP** pour certaines capacités de calcul ou de Machine Learning.

**Avantages :**

- compromis entre contrôle et flexibilité
- possibilité de conserver certaines données en interne
- accès aux ressources Cloud pour absorber les pics de charge

**Inconvénients :**

- architecture plus complexe
- synchronisation des données
- sécurité et gouvernance plus difficiles
- compétences nécessaires sur les deux environnements

### Résumé

| Architecture | Avantage principal | Inconvénient principal |
|---|---|---|
| **On-premise** | Contrôle | Coût et maintenance |
| **Full Cloud** | Flexibilité / scalabilité | Dépendance au fournisseur |
| **Hybrid Cloud** | Contrôle + flexibilité | Complexité |


## Introduction à la notion de data lake

Un datalake est un **système de stockage d’objets** compatible.  
Il permet de stocker et récupérer des fichiers (objets) de manière simple et rapide, à petite ou grande échelle.

<p align="center">
  <img src="https://openmetal.io/wp-content/uploads/2025/05/Data-Lake-Data-Types-Diagram.png" alt="Source de l'image" width="600"/>
</p>

### Acteurs du marché

| Solution | Particularité |
|----------|---------------|
| AWS S3 | Stockage d’objets cloud leader, robuste, riche en fonctionnalités |
| Azure Storage Account | Stockage cloud Microsoft, intégration avec Azure Data Factory |
| Google Cloud Storage | Stockage cloud Google, compatible S3 via certains outils |
| MinIO | Open-source, léger, compatible S3, déployable localement ou en cloud souverain |


### Concepts clés

#### Buckets

- Un **bucket** est un conteneur logique pour stocker des objets (fichiers).  
- Chaque objet a une **clé** unique dans le bucket.  
- On peut avoir plusieurs buckets pour organiser les données par projet, type ou environnement.

Exemple :
- Pour Microsoft Azure :  le service de storage account
- Pour AWS : le service S3 (*Simple Storage Service*)

<p align="center">
  <img src="https://technomind.fr/wp-content/uploads/2025/11/Architecture-interne-du-stockage-S3-683x1024.png" alt="Source de l'image" width="600"/>
</p>

#### Stockage de fichiers vs SGBD

- Un datalake stocke des fichiers, contrairement aux SGBD qui stockent des **données structurées** dans des tables.  
- Idéal pour les **fichiers volumineux**, les logs, les images, vidéos, JSON, CSV, Parquet…  
- Les fichiers peuvent être lus par des outils de traitement de données, ETL, Spark, ou Pandas.

#### Data Lake et couches Medaillon

- MinIO est souvent utilisé pour constituer un **Data Lake**, c’est-à-dire un entrepôt de données centralisé qui accepte tous types de formats.  
- Exemple de couches Medaillon :
  - **Bronze** : données brutes, telles qu’elles sont collectées
  - **Silver** : données nettoyées, enrichies, prêtes pour analyse
  - **Gold** : données consolidées et optimisées pour reporting ou machine learning

<p align="center">
  <img src="https://www.databricks.com/sites/default/files/inline-images/building-data-pipelines-with-delta-lake-120823.png" alt="Source de l'image" width="600"/>
</p>

#### Formats de fichiers

| Format | Particularité | Usage recommandé |
|--------|---------------|----------------|
| JSON | Semi-structuré, flexible, mais volumineux | Données brutes ou API |
| CSV | Structuré, simple | Transformation et échange de données |
| Parquet | Colonnaire, compressé, partitionnable | Analytics, data lake, Big Data |
  
<p align="center">
  <img src="https://substackcdn.com/image/fetch/$s_!4vTf!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F7b4d4edc-20dd-4480-8b33-e11705fcd61f_2000x1428.png" alt="Source de l'image" width="600"/>
</p>

Le **partitionnement** consiste à **découper un jeu de données en plusieurs sous-dossiers** en fonction de la valeur d’une ou plusieurs colonnes.

Exemple :

```
final/parquet/
 ├── etiquette_dpe=A/
 ├── etiquette_dpe=B/
 ├── etiquette_dpe=C/
 └── etiquette_dpe=D/
```

Chaque dossier contient uniquement les lignes correspondant à la valeur de la partition.

#### Pourquoi le partitionnement est essentiel en Data Lake ?

Sans partitionnement :
- le moteur de calcul (Spark, Pandas, DuckDB…) doit lire **tout le dataset**

Avec partitionnement :
- seules les partitions utiles sont lues  --> Gain énorme en **temps de calcul** et en **coût**


#### Cardinalité : notion clé

La **cardinalité** correspond au **nombre de valeurs distinctes** d’une colonne.

Types de cardinalité

| Type | Exemple | Caractéristique |
|------|---------|-----------------|
| Faible cardinalité ✅ | `etiquette_dpe` (A, B, C, D, E) | Très peu de valeurs |
| Moyenne | `code_departement` (01 → 95) | Plusieurs dizaines |
| Haute cardinalité ❌ | `id_unique`, `numero_dpe` | Quasi unique par ligne |

#### Bonne pratique

On **partitionne de préférence sur des colonnes à faible cardinalité**, comme :

- catégories (`etiquette_dpe`)
- pays
- département
- type de produit
- date

#### Mauvais exemple

Partitionner sur :

- un identifiant unique
- un numéro de facture
- une clé primaire

Cela crée des **milliers de dossiers vides ou à une seule ligne**, ce qui dégrade les performances.


#### Pourquoi c’est stratégique en entreprise ?

Le partitionnement est utilisé massivement dans :

- Data Lakes AWS S3
- Azure Data Lake
- Google Cloud Storage
- Spark
- Hive
- Databricks

Il permet de :

- accélérer les requêtes  
- réduire les coûts de lecture  
- mieux organiser les données  

#### Astuces : Lecture d'un fichier parquet

Les fichiers **Parquet** sont **colonnaires et optimisés pour le traitement Big Data**, mais ils ne sont pas facilement lisibles dans un simple ou éditeur de texte classique.  

Pour visualiser ou explorer un fichier Parquet localement, on peut utiliser des outils **open source** comme [Parquet Viewer](https://github.com/mukunku/ParquetViewer). Une application graphique pour ouvrir et explorer les fichiers Parquet.

:bulb: [Télécharger le fichier `ParquetViewer.exe`](https://github.com/mukunku/ParquetViewer/releases/download/v3.5.0.2/ParquetViewer.exe)


<p align="center">
  <img src="https://github.com/mukunku/ParquetViewer/raw/main/wiki_images/main_screenshot5.png" alt="Source de l'image" width="600"/>
</p>


### Apache Iceberg

<p align="center">
  <img src="https://cdn.prod.website-files.com/60f955236a773f743298d63b/64ba8c146aed466249378cb1_AnyConv.com__image6.webp" alt="Source de l'image" width="600"/>
</p>


#### Qu’est-ce qu’Apache Iceberg ?

Apache Iceberg est un **format de table open source pour les Data Lakes** qui apporte :

- Une **gestion ACID complète** (Atomicité, Cohérence, Isolation, Durabilité) sur des fichiers stockés dans des systèmes distribués (S3, ADLS, GCS…).  
- La possibilité de **travailler sur des tables immuables avec des snapshots**, ce qui rend les opérations comme **update, delete, merge** possibles sur un Data Lake.  
- Une compatibilité avec des moteurs de calcul comme **Spark, Flink, Trino, Hive**.

#### Pourquoi ACID est important ?

ACID est un acronyme pour :

| Lettre | Signification | Pourquoi c’est utile |
|--------|---------------|--------------------|
| A | Atomicité | Chaque transaction est **tout ou rien** : aucune donnée partiellement écrite |
| C | Cohérence | Les données restent **cohérentes** après chaque transaction |
| I | Isolation | Plusieurs jobs ou utilisateurs peuvent écrire/lire simultanément **sans conflit** |
| D | Durabilité | Une fois une transaction validée, elle est **persistante**, même en cas de panne |

Sans ACID, les Data Lakes peuvent devenir **instables** lorsqu’on fait des mises à jour ou des suppressions, ce qui peut provoquer des **données corrompues ou incohérentes**.

#### Pourquoi Iceberg est intéressant

1. **Gestion efficace des fichiers**  
   - Iceberg maintient un **catalogue de métadonnées**.  
   - Permet d’éviter de scanner **des milliers de fichiers** à chaque requête.  
   - Compatible avec **partitionnement et clustering**.

2. **Support ACID sur un Data Lake**  
   - Les tables Iceberg permettent les **UPDATE, DELETE, MERGE** alors que des formats classiques comme Parquet ne le permettent pas nativement.  
   - Idéal pour des pipelines de données **modifiables**.

3. **Snapshots et time travel**  
   - On peut revenir à un **ancien état de la table** en quelques lignes de code.  
   - Très utile pour **audits, debugging, et pipelines reproductibles**.

4. **Performance et scalabilité**  
   - Partitionnement intelligent et indexation des fichiers.  
   - Requêtes plus rapides sur de grands datasets même dans le Cloud.

#### Cas d’usage typique

- **Data Lake d’entreprise** : stockage de données transactionnelles et analytiques avec possibilité de modifications.  
- **Tables événementielles** : logs, clics web, transactions financières.  
- **Machine Learning** : réentraîner des modèles avec des versions stables des datasets.


### Conclusion

- Iceberg apporte **robustesse et flexibilité** aux Data Lakes.  
- Avec ACID et snapshots, on peut faire du **traitement batch ou streaming** sur des données fiables.  
- Il combine **les avantages des Data Warehouses (fiabilité, cohérence)** et des **Data Lakes (scalabilité, coût réduit)**.


## Liens utiles

- [Data platform illustration](https://www.blog.datawithbaraa.com/p/the-big-three-databricks-vs-snowflake)