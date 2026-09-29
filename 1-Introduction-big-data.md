# Chapitre 2 : Big Data et Cloud

- [Chapitre 2 : Big Data et Cloud](#chapitre-2--big-data-et-cloud)
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

👉 **IaaS = on loue l'infrastructure.**

### PaaS — Platform as a Service

Le fournisseur gère davantage l'infrastructure et fournit une **plateforme prête à utiliser**.

Le développeur se concentre principalement sur son application ou son traitement de données.

**Exemples :**

- **Snowflake**
- **Databricks**

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