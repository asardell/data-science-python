# Chapitre 7 — Méthodes Factorielles (PCA, MCA, Kernel PCA, t-SNE)

Les **méthodes factorielles** sont des techniques statistiques permettant de **résumer**, **projeter** et **visualiser** des données multivariées dans un espace de dimension plus faible.  
Elles sont particulièrement utiles en **exploration** (EDA) mais aussi en **préparation de Machine Learning** :  
- pour simplifier un modèle,  
- réduire le bruit,  
- éliminer la colinéarité,  
- visualiser la structure intrinsèque des données.

Dans ce chapitre, nous présentons :

- **PCA (Analyse en Composantes Principales)** — numérique, linéaire (scikit-learn)  
- **MCA (Analyse des Correspondances Multiples)** — catégorielle (package `mca` ou `prince`)  
- **Kernel PCA** — non linéaire (scikit-learn)  
- **t-SNE** — réduction non linéaire pour visualisation (scikit-learn)

Nous illustrons l'intérêt de ces méthodes avec notre jeu de données ADEME (besoins de chauffage, consommations, surfaces, étiquettes DPE, etc.) sans inclure de code Python dans ce chapitre.

- [Chapitre 7 — Méthodes Factorielles (PCA, MCA, Kernel PCA, t-SNE)](#chapitre-7--méthodes-factorielles-pca-mca-kernel-pca-t-sne)
- [Analyse en Composantes Principales (PCA)](#analyse-en-composantes-principales-pca)
  - [Objectifs de la PCA](#objectifs-de-la-pca)
    - [Réduire la dimension](#réduire-la-dimension)
    - [Maximiser la variance expliquée](#maximiser-la-variance-expliquée)
    - [Créer des axes interprétables](#créer-des-axes-interprétables)
  - [Étapes méthodologiques](#étapes-méthodologiques)
    - [Sélection des variables](#sélection-des-variables)
    - [Standardisation des variables](#standardisation-des-variables)
    - [Calcul de la PCA](#calcul-de-la-pca)
    - [Analyse de la variance](#analyse-de-la-variance)
    - [Projection des individus](#projection-des-individus)
    - [Analyse des variables](#analyse-des-variables)
- [Analyse des Correspondances Multiples (MCA)](#analyse-des-correspondances-multiples-mca)
  - [Objectifs de la MCA](#objectifs-de-la-mca)
    - [Résumer l’information](#résumer-linformation)
    - [Détecter des relations et regroupements](#détecter-des-relations-et-regroupements)
    - [Visualiser les données](#visualiser-les-données)
  - [Étapes méthodologiques](#étapes-méthodologiques-1)
    - [Sélection des variables](#sélection-des-variables-1)
    - [Application de la MCA](#application-de-la-mca)
    - [Variance expliquée](#variance-expliquée)
    - [Projection des individus](#projection-des-individus-1)
    - [Analyse des modalités](#analyse-des-modalités)
    - [Visualisation](#visualisation)
- [Kernel PCA (KPCA)](#kernel-pca-kpca)
  - [Objectifs de la KPCA](#objectifs-de-la-kpca)
    - [Capturer les relations non linéaires](#capturer-les-relations-non-linéaires)
    - [Réduction de dimension](#réduction-de-dimension)
    - [Visualisation](#visualisation-1)
  - [Méthodologie](#méthodologie)
    - [Choix des variables](#choix-des-variables)
    - [Standardisation des données](#standardisation-des-données)
    - [Sélection du noyau](#sélection-du-noyau)
    - [Analyse des composantes](#analyse-des-composantes)
    - [Interprétation](#interprétation)
    - [Applications](#applications)
- [t-SNE (t-Distributed Stochastic Neighbor Embedding)](#t-sne-t-distributed-stochastic-neighbor-embedding)
  - [Objectifs du t-SNE](#objectifs-du-t-sne)
    - [Préserver les voisinages locaux](#préserver-les-voisinages-locaux)
    - [Réduction dimensionnelle](#réduction-dimensionnelle)
    - [Visualisation](#visualisation-2)
  - [Méthodologie](#méthodologie-1)
    - [Sélection et normalisation des variables](#sélection-et-normalisation-des-variables)
    - [Paramètres clés du t-SNE](#paramètres-clés-du-t-sne)
    - [Implémentation avec sklearn](#implémentation-avec-sklearn)
    - [Visualisation des résultats](#visualisation-des-résultats)
    - [Interprétation](#interprétation-1)
    - [Inconvénients de t-SNE](#inconvénients-de-t-sne)
    - [Applications](#applications-1)
- [En bref](#en-bref)

# Analyse en Composantes Principales (PCA)

L’Analyse en Composantes Principales (PCA) est une méthode d’analyse factorielle destinée à **résumer l’information contenue dans plusieurs variables numériques** en un nombre réduit de dimensions (axes), tout en conservant un maximum de variance.

Elle sert à :  
- explorer des données multivariées  
- comprendre la structure des variables  
- détecter des groupes ou anomalies  
- visualiser les données  
- préparer certaines étapes de machine learning (réduction dimensionnelle, élimination de redondances)


## Objectifs de la PCA

### Réduire la dimension
Transformer plusieurs variables initiales en composantes principales non corrélées. Par exemple, si l'on a `surface_habitable_logement`, `besoin_chauffage` et `conso_chauffage_ef`, la PCA peut créer 2 axes qui résument l'information.

### Maximiser la variance expliquée
PC1 porte l’information la plus importante, PC2 la suivante, etc. Par exemple :
```python
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(df_scaled)
print(pca.explained_variance_ratio_)
```
Cela montre la proportion de variance capturée par chaque composante.

### Créer des axes interprétables
Chaque axe est une combinaison linéaire des variables initiales, pondérées par leur contribution. Par exemple :
```python
for i, col in enumerate(colonnes_num):
    print(f'{col} contribution PC1:', pca.components_[0,i])
``` 
On voit quelles variables influencent le plus PC1.


## Étapes méthodologiques

### Sélection des variables
Choisir des colonnes numériques pertinentes du dataset : `surface_habitable_logement`, `besoin_chauffage`, `conso_chauffage_ef`, `conso_ecs_ef`.

### Standardisation des variables
Les variables doivent être standardisées pour éviter que l’échelle fausse la PCA.
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[colonnes_num])
```

### Calcul de la PCA
```python
pca = PCA(n_components=4)
pca_result = pca.fit_transform(df_scaled)
```

### Analyse de la variance

Vérifier la proportion de variance expliquée pour choisir le nombre de composantes.
```python
explained_var = pca.explained_variance_ratio_
print(explained_var)
```

- **Scree plot** :
```python
import matplotlib.pyplot as plt
plt.bar(range(1, len(explained_var)+1), explained_var*100)
plt.xlabel('Composantes principales')
plt.ylabel('Pourcentage de variance expliquée')
plt.title('Scree plot')
plt.show()
```

### Projection des individus
Visualiser la distribution des individus sur les axes principaux :
```python
plt.scatter(pca_result[:,0], pca_result[:,1], c='skyblue', edgecolor='k')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Projection des individus')
plt.show()
```

### Analyse des variables
Étudier les contributions de chaque variable à chaque composante :
```python
for i, var in enumerate(colonnes_num):
    plt.arrow(0, 0, pca.components_[0,i], pca.components_[1,i], head_width=0.05, color='r')
    plt.text(pca.components_[0,i]*1.15, pca.components_[1,i]*1.15, var, color='g')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Cercle des corrélations')
plt.grid()
plt.show()
```

Ainsi, cette méthodologie permet de combiner la **réduction dimensionnelle**, la **visualisation** et l’**interprétation des variables et individus** sur un jeu de données réel comme celui des logements et de leur consommation énergétique.

# Analyse des Correspondances Multiples (MCA)

L’Analyse des Correspondances Multiples (MCA) est une méthode d’analyse factorielle adaptée aux **données qualitatives** ou catégorielles. Elle est utile pour :

- Explorer les relations entre plusieurs variables catégorielles.
- Réduire la dimensionnalité tout en conservant l’information principale.
- Visualiser la distribution des individus et des modalités.
- Préparer des étapes ultérieures de machine learning.


## Objectifs de la MCA

### Résumer l’information
Transformer plusieurs variables catégorielles en un nombre réduit de dimensions (axes) représentant les principales associations entre modalités.

### Détecter des relations et regroupements
Identifier des clusters d’individus ayant des profils similaires selon les variables catégorielles.

### Visualiser les données
Produire des graphes permettant de voir la proximité entre individus et entre modalités.


## Étapes méthodologiques

### Sélection des variables
Choisir uniquement les colonnes catégorielles pertinentes pour l’analyse, par exemple `etiquette_dpe`, `type_batiment`, `code_postal_ban`.

### Application de la MCA
Utiliser le package `mca` pour calculer les composantes principales.
```python
import mca
# Sélection des colonnes catégorielles
colonnes_cat = ['etiquette_dpe', 'type_batiment', 'code_postal_ban']
df_cat = df[colonnes_cat]

mca_bruite = mca.MCA(df_cat, benzecri=True)
mca_result = mca_bruite.fs_r()
```

### Variance expliquée
Évaluer la part de l’inertie expliquée par chaque dimension :
```python
inertia = mca_bruite.frac_var()
print("Variance expliquée par dimension:", inertia)
```

### Projection des individus
Visualiser la distribution des individus selon les premières dimensions :
```python
import matplotlib.pyplot as plt
plt.scatter(mca_result[:,0], mca_result[:,1], c='skyblue', edgecolor='k')
plt.xlabel('Dim1')
plt.ylabel('Dim2')
plt.title('Projection des individus MCA')
plt.grid()
plt.show()
```

### Analyse des modalités
Voir quelles modalités contribuent le plus à chaque dimension :
```python
mod_contrib = mca_bruite.var_cor()
print(mod_contrib)
```

### Visualisation
- Cercle des corrélations (ou cartes factorielle) pour les modalités
```python
for i, col in enumerate(colonnes_cat):
    plt.text(mca_bruite.V[0,i], mca_bruite.V[1,i], col, color='g')
plt.xlabel('Dim1')
plt.ylabel('Dim2')
plt.title('Cercle des corrélations MCA')
plt.grid()
plt.show()
```

Cette méthodologie permet de combiner la **réduction dimensionnelle**, la **visualisation des individus et modalités** et l’**interprétation des relations entre variables catégorielles**, ce qui est essentiel pour l’exploration de données avant toute modélisation.


# Kernel PCA (KPCA)

La Kernel Principal Component Analysis (KPCA) est une extension non linéaire de la PCA classique. Elle permet de **capturer des structures complexes dans les données** qui ne peuvent pas être expliquées par des combinaisons linéaires.

KPCA est particulièrement utile lorsque les relations entre les variables sont **non linéaires** et que la PCA classique ne permet pas de bien séparer les informations.


## Objectifs de la KPCA

### Capturer les relations non linéaires
Contrairement à la PCA classique, KPCA applique une transformation non linéaire (via un noyau) pour projeter les données dans un espace de dimension supérieure où les structures linéaires peuvent être extraites.

### Réduction de dimension
Comme la PCA classique, KPCA réduit la dimensionnalité, mais elle peut mieux conserver la variance des données non linéaires.

### Visualisation
Permet de projeter les individus sur les axes principaux non linéaires pour détecter des clusters ou des motifs complexes.


## Méthodologie

### Choix des variables
Sélectionner des colonnes numériques pertinentes. Par exemple : `surface_habitable_logement`, `besoin_chauffage`, `conso_chauffage_ef`, `conso_ecs_ef`.

### Standardisation des données
Comme pour la PCA classique, standardiser les variables pour éviter que l’échelle fausse l’analyse :
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[colonnes_num])
```

### Sélection du noyau
KPCA utilise différents types de noyaux :
- `linear` : équivalent à la PCA classique
- `poly` : polynomial
- `rbf` : radial basis function, capture des relations très flexibles
- `sigmoid` : fonction sigmoïde

Exemple de configuration :
```python
from sklearn.decomposition import KernelPCA
kpca = KernelPCA(n_components=2, kernel='rbf', gamma=0.1)
kpca_result = kpca.fit_transform(df_scaled)
```

### Analyse des composantes
Visualiser la variance projetée ou la distribution des individus :
```python
import matplotlib.pyplot as plt
plt.scatter(kpca_result[:,0], kpca_result[:,1], c='skyblue', edgecolor='k')
plt.xlabel('KPCA1')
plt.ylabel('KPCA2')
plt.title('Projection des individus avec KPCA')
plt.grid()
plt.show()
```

### Interprétation
- Contrairement à la PCA classique, **les composantes de KPCA ne sont pas des combinaisons linéaires directes des variables originales**, donc l’interprétation des coefficients est plus complexe.
- On se concentre sur **l’exploration visuelle et la détection de clusters** ou d’anomalies.
- L’effet du noyau (`gamma` pour rbf, `degree` pour poly) est crucial pour la qualité de la projection.

### Applications
- Détection de groupes non linéaires dans des datasets complexes.
- Prétraitement pour du ML supervisé : réduction dimensionnelle avant clustering, classification.
- Visualisation de patterns cachés dans les données.


Cette méthodologie permet d’appliquer une approche factorielle non linéaire aux données, offrant **plus de flexibilité que la PCA classique**, et est particulièrement adaptée à l’exploration de datasets comportant des relations complexes entre variables.


# t-SNE (t-Distributed Stochastic Neighbor Embedding)

Le t-SNE est une méthode de réduction de dimension **non linéaire**, particulièrement adaptée pour **visualiser des données complexes en 2 ou 3 dimensions**. Il est souvent utilisé pour explorer des datasets avant d’appliquer du machine learning.

## Objectifs du t-SNE

### Préserver les voisinages locaux
t-SNE conserve principalement les relations de proximité entre individus proches dans l’espace original. Il permet de détecter **clusters ou sous-groupes**.

### Réduction dimensionnelle
Réduit les données à 2 ou 3 dimensions pour visualisation. Contrairement à la PCA, il ne conserve pas la variance globale mais plutôt la structure locale.

### Visualisation
Permet de visualiser des regroupements naturels dans les données, souvent difficile à détecter avec PCA si les relations sont non linéaires.


## Méthodologie

### Sélection et normalisation des variables
Sélectionner des colonnes numériques ou transformer les variables catégorielles en numériques (via One-Hot Encoding si nécessaire).
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[colonnes_num])
```

### Paramètres clés du t-SNE
- `n_components` : nombre de dimensions finales (2 ou 3)
- `perplexity` : équilibre entre la prise en compte des voisinages locaux et globaux (souvent entre 5 et 50)
- `learning_rate` : vitesse d’apprentissage
- `n_iter` : nombre d’itérations pour la convergence

### Implémentation avec sklearn
```python
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=1000, random_state=42)
tsne_result = tsne.fit_transform(df_scaled)
```

### Visualisation des résultats
```python
import matplotlib.pyplot as plt
plt.scatter(tsne_result[:,0], tsne_result[:,1], c='skyblue', edgecolor='k')
plt.xlabel('t-SNE1')
plt.ylabel('t-SNE2')
plt.title('Projection des individus avec t-SNE')
plt.grid()
plt.show()
```

### Interprétation
- Chaque point représente un individu dans l’espace réduit.
- Les **points proches** dans le graphique sont similaires dans les données originales.
- La taille de la perplexité influence la granularité des clusters détectés.
- t-SNE ne fournit pas de coefficients de variables, donc **l’interprétation est principalement visuelle**.

### Inconvénients de t-SNE
- **Computation lourde** sur de grands datasets.
- **Non reproductible** sans fixer `random_state` et dépendant de l’initialisation.
- **Pas de projection pour de nouvelles données** (l’algorithme ne génère pas de fonction d’encodage simple).
- **Difficulté à interpréter les distances relatives** entre clusters (t-SNE préserve surtout les voisinages locaux).
- Sensible au choix des hyperparamètres (`perplexity`, `learning_rate`).

### Applications
- Détection de clusters ou sous-groupes dans les données.
- Visualisation des données pour exploration avant ML supervisé ou non supervisé.
- Comparaison des distributions ou identification d’outliers.


Cette méthodologie permet de projeter les données complexes dans un espace 2D ou 3D en conservant les relations locales, ce qui en fait un outil puissant pour **l’exploration et la visualisation avant modélisation**, tout en gardant à l’esprit ses limites et inconvénients.

# En bref

:warning:

**PCA**, produit des composantes linéaires des variables originales.
- Ces composantes conservent la variance globale et sont cohérentes avec l’espace original.
- On peut donc les utiliser comme features pour un modèle ML supervisé ou non supervisé (classification, clustering, régression, etc.).
**t-SNE** est conçu uniquement pour la visualisation et l’exploration des données.
- Il préserve les voisinages locaux, mais pas la structure globale ni les distances exactes
- Les coordonnées produites ne sont pas linéairement liées aux variables originales.
- Il ne doit pas être utilisé comme entrée pour un modèle ML, car il peut créer des artefacts ou fausser les relations.
**KPCA** applique une transformation non linéaire des données à l’aide d’un kernel (RBF, polynomial, etc.), puis fait une PCA dans cet espace transformé.
- Les composantes extraites peuvent être utilisées comme features pour du ML, contrairement à t-SNE.
- Cependant : Comme c’est basé sur un kernel, les nouvelles données doivent être transformées avec le même kernel pour produire les composantes.
- Les résultats peuvent être moins interprétables que la PCA classique.
- Si le dataset est très grand, la KPCA peut devenir très coûteuse en mémoire et temps.

