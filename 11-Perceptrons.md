# Chapitre 11 : Perceptron & Réseau de Neurones Multicouches (MLP)

- [Chapitre 11 : Perceptron \& Réseau de Neurones Multicouches (MLP)](#chapitre-11--perceptron--réseau-de-neurones-multicouches-mlp)
- [Introduction générale aux réseaux de neurones](#introduction-générale-aux-réseaux-de-neurones)
- [Le Perceptron (modèle simple)](#le-perceptron-modèle-simple)
  - [Qu'est-ce que c'est ?](#quest-ce-que-cest-)
    - [Avantages](#avantages)
    - [Limites](#limites)
- [Perceptron Multicouche (MLP)](#perceptron-multicouche-mlp)
    - [Pourquoi c’est puissant ?](#pourquoi-cest-puissant-)
- [Classification avec Perceptron](#classification-avec-perceptron)
  - [Quand utiliser un réseau de neurones pour classifier ?](#quand-utiliser-un-réseau-de-neurones-pour-classifier-)
  - [Fonctionnement en classification](#fonctionnement-en-classification)
  - [Hyperparamètres importants (MLPClassifier)](#hyperparamètres-importants-mlpclassifier)
  - [Exemple de code Python (classification)](#exemple-de-code-python-classification)
- [Les fonctions d'activation (ReLU, tanh, logistic/sigmoid, softmax)](#les-fonctions-dactivation-relu-tanh-logisticsigmoid-softmax)
  - [La fonction ReLU](#la-fonction-relu)
  - [La fonction tanh](#la-fonction-tanh)
  - [La fonction sigmoïde (logistic)](#la-fonction-sigmoïde-logistic)
  - [La fonction Softmax](#la-fonction-softmax)
  - [Résumé global](#résumé-global)
  - [Cas des fonctions d'activation en régression](#cas-des-fonctions-dactivation-en-régression)
    - [Régression](#régression)
    - [Activation linéaire (aucune activation)](#activation-linéaire-aucune-activation)
  - [Exceptions possibles](#exceptions-possibles)
    - [Cible strictement positive](#cible-strictement-positive)
    - [Cible bornée entre 0 et 1](#cible-bornée-entre-0-et-1)
    - [Cible bornée entre -1 et 1](#cible-bornée-entre--1-et-1)
  - [Exemples appliqués au contexte DPE](#exemples-appliqués-au-contexte-dpe)
  - [Tableau récapitulatif](#tableau-récapitulatif)
  - [Règle simple](#règle-simple)
- [Régression](#régression-1)
  - [Fonctionnement en régression](#fonctionnement-en-régression)
  - [Hyperparamètres](#hyperparamètres)
  - [Métriques d'évaluation](#métriques-dévaluation)
  - [Exemple code Python (régression)](#exemple-code-python-régression)
- [Avantages \& Inconvénients](#avantages--inconvénients)
  - [Avantages](#avantages-1)
  - [Inconvénients](#inconvénients)
  - [Quand utiliser ces modèles ?](#quand-utiliser-ces-modèles-)
- [Conclusion](#conclusion)


# Introduction générale aux réseaux de neurones
Les réseaux de neurones artificiels s’inspirent du fonctionnement du cerveau humain :  
des "neurones" (unités de calcul) reliés entre eux apprennent à reconnaître des motifs dans les données.

Un réseau de neurones peut servir à :
- classifier des objets (chat / chien, bon / mauvais, A/B/C…)
- prédire des valeurs continues (prix, température, consommation d'énergie)

Il existe deux grandes familles :
1. **Le perceptron simple couche** → modèle très basique
2. **Les perceptrons multicouches (MLP)** → modèle puissant pour capturer des relations complexes


# Le Perceptron (modèle simple)
## Qu'est-ce que c'est ?
Le perceptron est l’un des tout premiers algorithmes de machine learning (1957).  
C’est un classifieur **linéaire**, c’est-à-dire qu’il sépare les classes par une **ligne**, un **plan** ou un **hyperplan** selon la dimension.

Il fonctionne en apprenant une combinaison linéaire des variables :

\[
y = w_1 x_1 + w_2 x_2 + ... + w_n x_n + b
\]

Puis applique une fonction seuil :
- si la somme est positive → classe 1  
- sinon → classe 0  

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/31/Perceptron.svg/1200px-Perceptron.svg.png" alt="Source de l'image" width="600"/>
</p>

### Avantages
- Très simple à comprendre
- Très rapide
- Peu de paramètres

### Limites
- Ne marche que pour des données **linéairement séparables**
- Ne gère pas les relations complexes
- Ne donne pas de probabilités
- **Ne peut pas faire de régression !**


# Perceptron Multicouche (MLP)
Le MLP, ou réseau de neurones à plusieurs couches, est une extension du perceptron.

Il contient :
- **une couche d’entrée**
- **une ou plusieurs couches cachées**
- **une couche de sortie**

Chaque couche applique une transformation non linéaire, ce qui permet au modèle d’apprendre des motifs complexes.
Cet lien utile permet de jouer avec la configuration d'un réseau de neurones : [vers l'application](https://playground.tensorflow.org/)

### Pourquoi c’est puissant ?
Parce qu'un MLP peut approximer **n’importe quelle fonction** si on lui donne assez de neurones.  
C’est ce qu’on appelle un **approximateur universel**.


<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/a/a9/Perceptron_4layers.png" alt="Source de l'image" width="600"/>
</p>


# Classification avec Perceptron

## Quand utiliser un réseau de neurones pour classifier ?
- lorsque les relations entre variables sont **non linéaires**
- lorsque les modèles simples (logistique, KNN…) sont insuffisants
- pour apprendre des structures complexes

## Fonctionnement en classification
La dernière couche utilise :
- **sigmoid** → classification binaire  
- **softmax** → classification multiclasse  

Les sorties sont des probabilités.

## Hyperparamètres importants (MLPClassifier)
- `hidden_layer_sizes` : nombre de neurones dans les couches cachées  
  → ex : (50, 50) = 2 couches de 50 neurones  
- `activation` : type de non-linéarité  
  - relu (le plus courant)  
  - tanh  
  - logistic  
- `solver` : méthode d’optimisation  
  - adam (par défaut, robuste)  
  - lbfgs (bon pour petits datasets)  
  - sgd (descente de gradient)  
- `learning_rate` : contrôle la vitesse d’apprentissage  
- `max_iter` : nombre maximal d’itérations  
- `alpha` : régularisation (évite le surapprentissage)

## Exemple de code Python (classification)
```python
from sklearn.neural_network import Perceptron, MLPClassifier
from sklearn.metrics import classification_report, accuracy_score

# MODELE 1 - Perceptron simple
perc = Perceptron()
perc.fit(X_train, y_train)
pred_perc = perc.predict(X_test)

print("Perceptron Accuracy :", accuracy_score(y_test, pred_perc))
print(classification_report(y_test, pred_perc))

# MODELE 2 - MLP Classifier
mlp = MLPClassifier(hidden_layer_sizes=(50,50),
                    activation='relu',
                    max_iter=500)
mlp.fit(X_train, y_train)

pred_mlp = mlp.predict(X_test)

print("MLP Accuracy :", accuracy_score(y_test, pred_mlp))
print(classification_report(y_test, pred_mlp))
```

# Les fonctions d'activation (ReLU, tanh, logistic/sigmoid, softmax)  

Les **fonctions d’activation** permettent d’introduire de la **non-linéarité** dans les modèles de Machine Learning, en particulier dans les réseaux de neurones.  
Sans elles, un réseau serait simplement une **régression linéaire**, incapable de capturer des relations complexes.

Dans le contexte du **DPE**, les relations entre variables (isolation, surface, ventilation, année de construction…) ne sont pas linéaires.  
Les fonctions d’activation permettent de modéliser ces comportements.

Pourquoi avons-nous besoin de non-linéarité ?

Dans les données DPE :

- la **consommation énergétique** n’augmente pas de façon linéaire avec la surface ;  
- l’**année de construction** introduit des ruptures (changements de normes) ;  
- la **ventilation** peut avoir un effet positif OU négatif ;  
- la **performance** dépend d’interactions entre plusieurs facteurs.

➡️ **Sans non-linéarité, ces effets ne peuvent pas être capturés.**

Les fonctions d’activation permettent justement d’introduire cette flexibilité.


## La fonction ReLU

Définition :

\[
ReLU(x) = \max(0, x)
\]

Exemple :

Imagine une mesure appelée **"impact de l’isolation sur la consommation"**.

- Si l’isolation est mauvaise → impact POSITIF (hausse de consommation)  
- Si elle est excellente → l’impact ne peut pas devenir négatif au point “d’économiser plus que 0”  

**La consommation ne peut pas être négative.**  
ReLU coupe donc tous les effets négatifs à zéro.

Ce que ReLU apporte :

- simplicité  
- efficacité  
- modèle “linéaire par morceaux”  
- activation la plus utilisée dans les réseaux modernes


## La fonction tanh

Définition :
\[
tanh(x) \in [-1, 1]
\]

Exemple:
Supposons une variable interne du modèle représentant **"impact de la ventilation"** :

- trop de ventilation → pertes → consommation augmente → valeur positive  
- pas assez de ventilation → stagnation de l’humidité → consommation diminue (chauffage plus efficace) → valeur négative  
- ventilation correcte → effet neutre → proche de 0

**tanh permet de modéliser des effets positifs ET négatifs.**

Ce que tanh apporte :

- utile dans les couches cachées  
- centrée autour de 0  
- utile quand des effets opposés doivent être représentés


## La fonction sigmoïde (logistic)

Définition  :
\[
sigmoid(x)=\frac{1}{1+e^{-x}}
\]

Sortie entre **0 et 1**, interprétable comme une **probabilité**.

Exemple :

Objectif : prédire si un logement est une **passoire énergétique** (Oui / Non).

La sigmoïde convertit un score interne en probabilité :

- x = +6 → 0,997 → "presque sûr que c’est une passoire"  
- x = 0 → 0,5 → "50/50, je ne sais pas"  
- x = –6 → 0,002 → "très peu probable"

Parfait pour les **problèmes binaires**.

Ce qu’elle apporte :

- transforme n’importe quel score en probabilité  
- utilisée dans la sortie d’un réseau pour classification binaire


## La fonction Softmax

Définition  :
Transforme une liste de scores en **probabilités qui totalisent 100 %**.

Exemple :

Objectif : prédire l'**étiquette DPE** (A → G).

Scores bruts du modèle :

| Classe | Score |
|--------|--------|
| A      | 1.2    |
| B      | 0.8    |
| C      | 3.3    |
| D      | 0.4    |

La softmax convertit ces scores en probabilités :

- A → 15 %  
- B → 9 %  
- C → **72 %**  
- D → 4 %

Le modèle choisit **C**, la classe avec la probabilité la plus élevée.

Ce qu’elle apporte :

- indispensable pour la classification **multiclasse**  
- garantit une distribution propre (somme = 100 %)


## Résumé global

| Activation | Domaines d’utilisation | Avantage clé | Exemple DPE |
|-----------|-------------------------|--------------|--------------|
| **ReLU** | couches cachées | simple, efficace | impact isolation |
| **tanh** | couches cachées | valeurs de -1 à 1 | ventilation, orientation |
| **sigmoïde** | sortie binaire | probabilité | passoire (Oui/Non) |
| **Softmax** | sortie multiclasse | distribution 100 % | étiquette A → G |


Métaphore pour retenir :

- **ReLU** → il ignore tout ce qui réduit trop la consommation (ne peut pas être négatif)  
- **tanh** → il gère les effets *positifs ET négatifs*  
- **sigmoïde** → il répond *oui ou non* à “est-ce une passoire ?"  
- **softmax** → il choisit l’étiquette finale *A/B/C/D/E/F/G*

## Cas des fonctions d'activation en régression

La grande différence concerne **l’activation en sortie** du modèle.

### Régression
La cible est une **valeur continue**, souvent sans borne, comme :

- consommation énergétique (kWh/m²/an)  
- déperditions thermiques  
- coûts énergétiques  

Dans ce cas, on utilise :

### Activation linéaire (aucune activation)
\[
f(x) = x
\]

Cela permet au modèle de prédire n’importe quelle valeur réelle.


## Exceptions possibles

### Cible strictement positive
Certaines variables ne peuvent pas être négatives, comme :

- consommation  
- déperditions  
- pertes thermiques  
- émissions CO₂  

Dans ce cas, on peut utiliser :

- ReLU  
- Softplus  

Ces fonctions empêchent la sortie d’être négative.

### Cible bornée entre 0 et 1
Exemples :
- scores normalisés  
- probabilités  
- indices de performance standardisés  

Activation recommandée : **sigmoïde**.

### Cible bornée entre -1 et 1
Exemple :
- indice standardisé autour de 0  

Activation recommandée : **tanh**.


## Exemples appliqués au contexte DPE

Exemple 1 : prédiction de la consommation énergétique
- couche cachée 1 : ReLU  
- couche cachée 2 : ReLU  
- sortie : activation **linéaire**

La consommation peut être 70, 150, 350, 600 kWh/m²/an : il faut une sortie non bornée.

Exemple 2 : prédiction des déperditions thermiques (> 0) -->  Sortie ReLU ou Softplus possible.

Exemple 3 : prédiction d’un score entre 0 et 1 --< Sortie sigmoïde.


## Tableau récapitulatif

| Contexte cible | Activation cachée | Activation sortie | Justification |
|----------------|-------------------|------------------|---------------|
| Régression classique | ReLU / tanh | Linéaire | Valeur réelle continue |
| Cible positive | ReLU / tanh | ReLU ou Softplus | Sortie toujours > 0 |
| Cible entre 0 et 1 | ReLU / tanh | Sigmoïde | Sortie bornée |
| Cible entre -1 et 1 | ReLU / tanh | tanh | Sortie bornée symétrique |


## Règle simple

- Les **couches cachées** utilisent généralement **ReLU** (ou d’autres non-linéarités).  
- La **couche de sortie** utilise une activation **linéaire** pour la régression.  
- On applique une activation spécifique en sortie uniquement si la cible a des limites naturelles.

# Régression

:warning: le perceptron simple **ne peut pas faire de régression**.  
On utilise donc directement le **MLPRegressor**.

## Fonctionnement en régression
La dernière couche contient :
- **1 neurone**
- activation = **linéaire**

Le modèle apprend à approximer une fonction continue :
\[
\hat{y} = f_\theta(x)
\]

## Hyperparamètres
(similaires au classifieur)
- `hidden_layer_sizes`
- `activation` (relu conseillé)
- `solver` (adam conseillé)
- `alpha` (régularisation)
- `max_iter`

## Métriques d'évaluation
- **MSE** : erreur quadratique moyenne  
- **RMSE** : racine MSE (interprétation réaliste)  
- **MAE** : erreur absolue moyenne  
- **R²** : proportion de variance expliquée  

## Exemple code Python (régression)
```python
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

mlp_reg = MLPRegressor(hidden_layer_sizes=(80, 40),
                       activation='relu',
                       max_iter=800)

mlp_reg.fit(X_train, y_train)

pred_reg = mlp_reg.predict(X_test)

mse = mean_squared_error(y_test, pred_reg)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, pred_reg)
r2 = r2_score(y_test, pred_reg)

print("MSE :", mse)
print("RMSE :", rmse)
print("MAE :", mae)
print("R² :", r2)
```


# Avantages & Inconvénients

## Avantages
- Très flexible et puissant
- Capable de modéliser des relations non linéaires
- Peut s’adapter à la fois à classification et régression
- Performant sur des grands volumes de données

## Inconvénients
- Peu interprétable (boîte noire)
- Peut facilement surapprendre
- Demande un certain réglage (hyperparamètres)
- Temps d’entraînement parfois long


##  Quand utiliser ces modèles ?

Très bon choix si :
- les relations sont complexes
- les autres modèles (logistique, KNN, arbres…) sous-performent
- données suffisamment nombreuses

Moins bon choix si :
- on veut expliquer le modèle
- peu de données
- on cherche rapidité + simplicité


# Conclusion
Les réseaux de neurones (perceptron et MLP) sont des outils puissants pour résoudre des problèmes de **classification** et de **régression**.

- Le **perceptron simple** est limité mais utile pour comprendre les bases.  
- Les **MLP** offrent une grande flexibilité et permettent d’apprendre des relations **non linéaires** difficiles pour les modèles classiques.

Ils demandent cependant une bonne configuration et un contrôle rigoureux du surapprentissage.

En recherche, on peut retrouver plusieurs variantes de réseaux de neurones.

<p align="center">
  <img src="https://miro.medium.com/1*cuTSPlTq0a_327iTPJyD-Q.png" alt="Source de l'image" width="600"/>
</p>