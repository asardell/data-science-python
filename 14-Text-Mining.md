# Chapitre 14 : Text Mining

- [Chapitre 14 : Text Mining](#chapitre-14--text-mining)
  - [Introduction](#introduction)
  - [Prétraitement des textes](#prétraitement-des-textes)
  - [Matrice Document-Terme (Bag of Words)](#matrice-document-terme-bag-of-words)
  - [TF-IDF (Term Frequency - Inverse Document Frequency)](#tf-idf-term-frequency---inverse-document-frequency)
  - [Analyse exploratoire](#analyse-exploratoire)
    - [Comptage de mots les plus fréquents](#comptage-de-mots-les-plus-fréquents)
    - [Clustering de documents](#clustering-de-documents)
    - [Topic Modeling (LDA)](#topic-modeling-lda)
    - [Visualisation par cluster](#visualisation-par-cluster)
  - [Méthodes avancées](#méthodes-avancées)

## Introduction

Le **Text Mining** consiste à exploiter des données textuelles pour en extraire de l’information utile.  
Dans le contexte des DPE, les champs qualitatifs comme :

- `description_installation_chauffage_n1`
- `type_generateur_chauffage_principal_ecs`
- `type_installation_ecs_n1`

contiennent beaucoup d’informations sur les systèmes de chauffage, l’énergie utilisée, et les installations.

**Objectifs possibles :**

- Extraire les mots les plus fréquents pour chaque type d’étiquette DPE  
- Créer des représentations vectorielles pour les modèles de ML  
- Faire du clustering de descriptions pour identifier des patterns  
- Réaliser un topic modeling pour regrouper les installations similaires  


## Prétraitement des textes

Avant toute analyse :

- **Nettoyage** : retirer les caractères spéciaux, majuscules, ponctuation, doublons.  
- **Tokenisation** : découper les phrases en mots (`Radiateur électrique → ["Radiateur", "électrique"]`).  
- **Suppression des stopwords** : mots très fréquents mais peu informatifs (`de`, `à`, `et`).  
- **Lemmatisation / Stemming** : réduire les mots à leur racine (`installations → installation`).  

```python
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

# Charger les données
texts = df['description_installation_chauffage_n1'].astype(str)

# Nettoyage
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

texts_clean = texts.apply(clean_text)

# Tokenisation, suppression stopwords, lemmatisation
stop_words = set(stopwords.words('french'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

texts_processed = texts_clean.apply(preprocess)
```


## Matrice Document-Terme (Bag of Words)

- Représenter chaque description par un vecteur de la taille du vocabulaire  
- Chaque cellule indique la fréquence d’un mot dans un document  

| Document | radiateur | électrique | réseau | collectif | ballon | ... |
|----------|-----------|------------|--------|-----------|--------|-----|
| Doc 1    | 1         | 0          | 2      | 2         | 0      | ... |
| Doc 2    | 1         | 2          | 0      | 0         | 1      | ... |

```python
vectorizer = CountVectorizer()
X_counts = vectorizer.fit_transform(texts_processed)
```


## TF-IDF (Term Frequency - Inverse Document Frequency)

- Permet de pondérer les mots selon leur importance  
- Formule : `TF-IDF(t,d) = TF(t,d) * log(N / DF(t))`  
  - `TF(t,d)` : fréquence du mot t dans le document d  
  - `DF(t)` : nombre de documents contenant t  
  - `N` : nombre total de documents  

Exemple ludique :

- Le mot **électrique** apparaît dans beaucoup de DPE → TF élevé mais IDF faible  
- Le mot **ballon** apparaît rarement → TF plus faible mais IDF plus élevé → important pour identifier certaines installations

```python
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(texts_processed)
```


## Analyse exploratoire

### Comptage de mots les plus fréquents

```python
import matplotlib.pyplot as plt
from wordcloud import WordCloud

all_text = " ".join(texts_processed)
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)

plt.figure(figsize=(15,7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
```

### Clustering de documents

- Objectif : regrouper les DPE similaires selon leur description  
- Méthodes : KMeans, Agglomerative Clustering sur la matrice TF-IDF  

```python
from sklearn.cluster import KMeans

k = 5
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(X_tfidf)
df['cluster'] = clusters
```

### Topic Modeling (LDA)

- Latent Dirichlet Allocation : identifie des sujets latents  
- Exemple :  
  - Sujet 1 → radiateur, électrique, accumulation  
  - Sujet 2 → réseau, collectif, urbain  

```python
from sklearn.decomposition import LatentDirichletAllocation

n_topics = 5
lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
lda.fit(X_tfidf)

def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print(f"Topic {topic_idx}:")
        print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))

display_topics(lda, tfidf_vectorizer.get_feature_names_out(), 10)
```


### Visualisation par cluster

```python
for i in range(k):
    cluster_text = " ".join(df[df['cluster'] == i]['description_installation_chauffage_n1'].astype(str))
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(cluster_text)
    plt.figure(figsize=(12,5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(f"Cluster {i}")
    plt.show()
```

Applications possibles

- **Classification DPE** : utiliser les vecteurs TF-IDF pour prédire `etiquette_dpe` ou `etiquette_ges`  
- **Détection d’installations atypiques** : identifier des descriptions rares via similarité  
- **Analyse comparative** : comparer mots et sujets entre collectif/individuel ou différentes étiquettes DPE  
- **Recommandation et prévision** : clustering + topic modeling → recommander améliorations énergétiques  


## Méthodes avancées
*Word Embeddings (Word2Vec, GloVe)** : vecteurs continus capturant le sens  
- **Doc2Vec** : représenter chaque document entier  
- **BERT / Transformer** : analyses fines pour classification ou extraction d’information  

