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
- [Régression](#régression)
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
  <img src="https://www.researchgate.net/publication/286404296/figure/fig1/AS:334969059725313@1456874255502/Le-modele-de-perceptrons-multicouches-MLP.png" alt="Source de l'image" width="600"/>
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