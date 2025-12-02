# Chapitre 15 : Introduction au Reinforcement Learning

## Qu’est-ce que le Reinforcement Learning ?

Le **Reinforcement Learning** est une branche de l’apprentissage automatique où un **agent** apprend à prendre des décisions dans un **environnement** pour maximiser une **récompense cumulée**.

- Contrairement à la classification ou la régression : pas de “label” fixe à prédire.
- L’agent agit, observe les conséquences, et s’ajuste pour améliorer sa stratégie.

**Concepts clés :**

| Concept          | Description |
|-----------------|-------------|
| Agent            | Celui qui prend des décisions (ex : un système de gestion énergétique). |
| Environnement    | L’univers dans lequel l’agent évolue (ex : logement avec son système de chauffage). |
| Action           | Ce que l’agent peut faire (ex : augmenter ou réduire la température, changer de mode de chauffage). |
| État             | Description de la situation actuelle (ex : consommation énergétique, température, type de chauffage). |
| Récompense       | Score reçu après une action (ex : réduction de consommation, gain énergétique). |
| Politique        | Stratégie de l’agent pour choisir ses actions selon l’état. |


## Pourquoi RL pour les données DPE ?

Les données DPE contiennent beaucoup d’informations sur les logements, l’énergie utilisée et les installations. On peut imaginer un **agent virtuel** qui apprend à **optimiser la consommation énergétique d’un logement** en fonction de différents paramètres.

**Exemple concret :**

- **État (state)** :  
  - Surface du logement  
  - Type d’installation chauffage (`individuel` / `collectif`)  
  - Étiquette DPE actuelle  
  - Température intérieure  
  - Énergie utilisée (kWh/m²)

- **Actions (action)** :  
  - Changer le type d’émetteur (radiateur → plancher chauffant)  
  - Ajuster la température de consigne  
  - Installer un ballon ECS plus performant

- **Récompense (reward)** :  
  - Réduction de la consommation énergétique (kWh/m²/an)  
  - Amélioration de l’étiquette DPE  
  - Réduction des émissions GES

- **Objectif** : apprendre une **politique optimale** qui minimise la consommation énergétique tout en respectant le confort et le budget.


## Types de Reinforcement Learning

1. **Apprentissage par valeur (Value-based)**  
   - Exemple : Q-learning  
   - L’agent apprend une **fonction Q(s,a)** qui estime la “qualité” d’une action dans un état donné.  
   - Politique = choisir l’action avec le Q le plus élevé.

2. **Apprentissage par politique (Policy-based)**  
   - L’agent apprend directement la **politique** π(a|s) qui indique la probabilité de prendre chaque action dans un état donné.

3. **Méthodes hybrides (Actor-Critic)**  
   - Combinaison de valeur et politique  
   - Actor = propose l’action, Critic = évalue la qualité


## Exemple avec DPE

On peut simuler un environnement simple :  

- Un logement est défini par : `surface`, `type_chauffage`, `consommation`.  
- L’agent peut **choisir d’augmenter ou réduire la température**, ou **changer le type de chauffage**.  
- La **récompense** est négative si la consommation augmente, positive si elle diminue.

```python
import numpy as np
import random

# Etats simplifiés : [température, consommation]
states = [[20, 200], [22, 180], [24, 220]]

# Actions : 0 = baisser temp, 1 = garder temp, 2 = augmenter temp
actions = [0, 1, 2]

# Récompense simple
def reward(state, action):
    temp, cons = state
    if action == 0:  # baisser température
        return 10 - cons/50
    elif action == 1:  # garder
        return 5 - cons/50
    else:  # augmenter
        return -5 - cons/50

# Politique aléatoire
state = random.choice(states)
action = random.choice(actions)
r = reward(state, action)
print(f"Etat: {state}, Action: {action}, Récompense: {r}")
```


## Applications possibles sur DPE

- **Simulation d’amélioration énergétique** : tester différentes actions (isolation, chauffage, ECS) et mesurer impact sur consommation et GES  
- **Optimisation multi-objectifs** : confort vs consommation vs coût  
- **Recommandation personnalisée** : politique optimale pour chaque logement selon son état et historique de consommation


## Champs d’application

- **Smart home / bâtiments intelligents** : ajustement automatique des systèmes de chauffage pour réduire la consommation énergétique  
- **Gestion de réseaux urbains de chaleur** : optimiser la distribution de chaleur entre bâtiments collectifs  
- **Politiques publiques** : simuler l’impact de différentes mesures (subventions, isolation, changement de chauffage) sur la consommation globale et les émissions  
- **Simulation et formation** : outils pédagogiques pour sensibiliser aux économies d’énergie


## Conclusion

- Le RL est très adapté aux problèmes où **les décisions successives influencent le résultat final**  
- Dans le contexte DPE, il permet d’apprendre **stratégies optimales d’amélioration énergétique**  
- Nécessite la **simulation d’un environnement** et des **mesures fiables de récompense**  
- Peut être combiné avec d’autres méthodes ML pour enrichir les états ou améliorer les prévisions

