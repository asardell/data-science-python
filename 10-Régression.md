# Chapitre 9 : Apprentissage supervisé - Classification

## Qu’est-ce que l’apprentissage supervisé ?

L'apprentissage supervisé est une approche du Machine Learning où l'on dispose d'un **jeu de données annoté**, c’est-à-dire que chaque observation a une **valeur cible** connue. Le but est d’apprendre une **fonction de prédiction** capable de généraliser à de nouvelles données.  

## Classification vs Régression

### Classification
- **Variable cible** : catégorielle (ex. Oui / Non, A / B / C / D / E / F / G).  
- **Objectif** : assigner chaque observation à une **classe**.  

**Exemple :**  

| Logement | Surface (m²) | Conso Chauffage (kWh) | Passoire énergétique |
|-----------|--------------|----------------------|--------------------|
| 1         | 85           | 250                  | Oui                |
| 2         | 120          | 150                  | Non                |
| 3         | 60           | 300                  | Oui                |

- **Métriques typiques** :  
  - Matrice de confusion  
  - Accuracy (taux de bonnes classifications)  
  - Précision et rappel (pour détecter les classes minoritaires)  
  - F1-score (équilibre entre précision et rappel)

---

### 2.2 Régression
- **Variable cible** : continue (ex. nombre, quantité, température).  
- **Objectif** : prédire une **valeur numérique**.  

**Exemple :**  

| Logement | Surface (m²) | Type Chauffage | Besoin énergétique (kWh) |
|-----------|--------------|----------------|--------------------------|
| 1         | 85           | Gaz            | 250                      |
| 2         | 120          | Électrique     | 150                      |
| 3         | 60           | Fioul          | 300                      |

- **Métriques typiques** :  
  - MSE (Mean Squared Error) : moyenne des carrés des écarts entre valeurs observées et prédites  
  - MAE (Mean Absolute Error) : moyenne des écarts absolus  
  - R² : proportion de variance expliquée par le modèle

---

## 3. Préparer l’échantillon : train / test split

Pour évaluer la performance du modèle, on sépare le jeu de données en **ensemble d’entraînement** et **ensemble de test**.  

**Exemple de répartition pour le projet DPE :**  

| Ensemble | Nombre de logements | Proportion de passoires énergétiques |
|----------|-------------------|------------------------------------|
| Train    | 700               | 25%                                |
| Test     | 300               | 25%                                |

**Objectif :**  
- L’ensemble d’entraînement sert à **apprendre le modèle**.  
- L’ensemble de test sert à **évaluer sa capacité de généralisation**.  
- On peut utiliser la stratification pour conserver la proportion de classes dans les deux ensembles.

---

## 4. Évaluer un modèle simple

### 4.1 Pour la classification : Matrice de confusion

**Exemple de matrice de confusion sur le projet DPE :**  

|               | Prédit Oui | Prédit Non |
|---------------|------------|------------|
| Observé Oui   | 40         | 10         |
| Observé Non   | 15         | 235        |

- **Interprétation :**  
  - Diagonale : bonnes prédictions  
  - Hors diagonale : erreurs  
  - Accuracy = (40 + 235) / 300 = 91,7%

---

### 4.2 Pour la régression : MSE et MAE

**Exemple sur le besoin énergétique :**  

| Logement | Réel (kWh) | Prévu (kWh) | Erreur absolue | Erreur au carré |
|----------|------------|-------------|----------------|----------------|
| 1        | 250        | 260         | 10             | 100            |
| 2        | 150        | 140         | 10             | 100            |
| 3        | 300        | 310         | 10             | 100            |

- **MSE** = moyenne des erreurs au carré = (100 + 100 + 100)/3 = 100  
- **MAE** = moyenne des erreurs absolues = (10 + 10 + 10)/3 = 10  
- Plus ces valeurs sont faibles, meilleure est la performance.

---

## 5. Points clés

- L’apprentissage supervisé repose sur des données **étiquetées**.  
- **Classification** → variable cible catégorielle, prédire des classes.  
- **Régression** → variable cible continue, prédire des valeurs numériques.  
- La séparation **train/test** est essentielle pour évaluer la généralisation du modèle.  
- Les **métriques simples** (matrice de confusion, MSE) permettent un premier diagnostic avant d’utiliser des métriques plus complexes.
