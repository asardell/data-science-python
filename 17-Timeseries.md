# Chapitre 17 : Times series

- [Chapitre 17 : Times series](#chapitre-17--times-series)
  - [Introduction aux séries temporelles](#introduction-aux-séries-temporelles)
  - [Simulation d'une série temporelle pour illustration](#simulation-dune-série-temporelle-pour-illustration)
  - [Lissage des séries temporelles](#lissage-des-séries-temporelles)
    - [Moyenne mobile](#moyenne-mobile)
      - [Paramètres principaux](#paramètres-principaux)
      - [Avantages](#avantages)
      - [Inconvénients](#inconvénients)
    - [Lissage exponentiel](#lissage-exponentiel)
      - [Explication](#explication)
      - [Paramètres principaux](#paramètres-principaux-1)
      - [Avantages](#avantages-1)
      - [Inconvénients](#inconvénients-1)
  - [ARIMA / SARIMA](#arima--sarima)
      - [Explication](#explication-1)
      - [Paramètres principaux](#paramètres-principaux-2)
      - [Avantages](#avantages-2)
      - [Inconvénients](#inconvénients-2)
  - [SARIMAX avec variable exogène](#sarimax-avec-variable-exogène)
      - [Explication](#explication-2)
      - [Paramètres principaux](#paramètres-principaux-3)
      - [Avantages](#avantages-3)
      - [Inconvénients](#inconvénients-3)
  - [Prophet](#prophet)
      - [Explication](#explication-3)
      - [Paramètres principaux](#paramètres-principaux-4)
      - [Avantages](#avantages-4)
      - [Inconvénients](#inconvénients-4)
  - [Time series avec scikit-learn](#time-series-avec-scikit-learn)
  - [Points clés](#points-clés)
  - [Aller plus loin avec LSTM (Long Short-Term Memory)](#aller-plus-loin-avec-lstm-long-short-term-memory)

## Introduction aux séries temporelles

Une **série temporelle** est une suite chronologique de valeurs d'une variable observée à intervalles réguliers.  
Exemple : consommation énergétique mensuelle d’un logement ou d’un bâtiment.

**Objectifs principaux :**
- Identifier les **tendances** (augmentation ou diminution sur le long terme)
- Détecter la **saisonnalité** (variations périodiques)
- Capturer le **bruit** et prévoir les valeurs futures


## Simulation d'une série temporelle pour illustration

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Fixer le seed pour reproductibilité
np.random.seed(42)

# Générer une série temporelle simulée
n_years = 10
dates = pd.date_range(start='2010-01-01', periods=n_years*12, freq='M')

# Tendance linéaire
trend = np.linspace(200, 250, n_years*12)

# Saisonnalité mensuelle (12 mois)
seasonal = 20 * np.sin(2 * np.pi * np.arange(n_years*12)/12)

# Bruit aléatoire
noise = np.random.normal(0, 5, n_years*12)

# Consommation totale
consommation = trend + seasonal + noise

# Créer DataFrame
df = pd.DataFrame({'date': dates, 'consommation': consommation})
df.set_index('date', inplace=True)

# Visualisation
df['consommation'].plot(figsize=(12,4), title='Consommation énergétique simulée')
plt.ylabel('kWh/m²')
plt.show()
```

## Lissage des séries temporelles

### Moyenne mobile

La **moyenne mobile** est une méthode simple de lissage d’une série temporelle qui consiste à calculer la moyenne des valeurs sur une fenêtre glissante.  
Dans l’exemple ci-dessus, `window=12` signifie que l’on calcule la moyenne sur les 12 derniers mois pour chaque point de la série, ce qui permet de lisser les variations mensuelles et de visualiser la tendance générale.

#### Paramètres principaux
- `window` : taille de la fenêtre glissante (ici 12 mois pour un lissage annuel)
- `min_periods` (optionnel) : nombre minimum de valeurs non-nulles pour calculer la moyenne
- `center` (optionnel) : si `True`, la moyenne est centrée sur la fenêtre, sinon elle est alignée à la fin de la fenêtre

#### Avantages
- Très simple à comprendre et à implémenter
- Permet de visualiser facilement la **tendance générale** de la série
- Utile pour détecter des changements de niveau ou des cycles lents

#### Inconvénients
- **Ne capture pas la saisonnalité complexe** ni les variations non linéaires
- **Retarde les valeurs réelles** (décalage) : la moyenne reflète les valeurs passées, pas instantanées
- Sensible au choix de la taille de la fenêtre : trop petite → bruit persistant, trop grande → tendance trop lissée

```python
df['MA_12'] = df['consommation'].rolling(window=12).mean()
df[['consommation','MA_12']].plot(figsize=(12,4), title='Lissage par moyenne mobile (12 mois)')
plt.show()
```

### Lissage exponentiel

#### Explication
Le **lissage exponentiel** est une méthode qui attribue des **poids décroissants aux observations passées**, plus récentes ayant plus d’influence.  
- **Simple Exponential Smoothing (SES)** : lisse la série sans tenir compte de tendance ou saisonnalité.  
- **Holt-Winters** : permet d’intégrer **la tendance** (`trend='add'`) et **la saisonnalité** (`seasonal='add'`) dans le lissage.

Dans l’exemple, `smoothing_level=0.2` indique que 20% de la valeur actuelle est prise en compte et 80% de l’historique est conservé.

#### Paramètres principaux
- `smoothing_level` (`alpha`) : importance donnée à la dernière observation (0 < alpha < 1)
- `trend` : type de tendance à modéliser (`add` pour additive, `mul` pour multiplicative)
- `seasonal` : type de saisonnalité (`add` ou `mul`)
- `seasonal_periods` : nombre de périodes pour la saisonnalité (ex. 12 pour des données mensuelles annuelles)
- `optimized` : si True, les paramètres alpha/beta/gamma sont automatiquement optimisés

#### Avantages
- Plus réactif que la moyenne mobile aux changements récents
- Peut intégrer tendance et saisonnalité (Holt-Winters)
- Convient bien pour des séries régulières et saisonnières
- Moins de retard que la moyenne mobile

#### Inconvénients
- Nécessite un paramétrage (alpha, beta, gamma) qui peut influencer fortement les résultats
- Moins robuste aux **changements brusques ou outliers**
- Ne capture pas les relations exogènes avec d’autres variables

```python
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing

# Simple Exponential Smoothing
ses_model = SimpleExpSmoothing(df['consommation']).fit(smoothing_level=0.2, optimized=False)
df['SES'] = ses_model.fittedvalues

df[['consommation','SES']].plot(figsize=(12,4), title='Lissage exponentiel simple')
plt.show()

# Holt-Winters (trend + seasonality)
hw_model = ExponentialSmoothing(df['consommation'], trend='add', seasonal='add', seasonal_periods=12).fit()
df['HW'] = hw_model.fittedvalues
df[['consommation','HW']].plot(figsize=(12,4), title='Holt-Winters')
plt.show()
```

## ARIMA / SARIMA

#### Explication
Les modèles **ARIMA (AutoRegressive Integrated Moving Average)** et **SARIMA (Seasonal ARIMA)** sont des modèles statistiques pour les séries temporelles.  
- **ARIMA(p,d,q)** :
  - `p` : nombre de termes auto-régressifs (AR) – dépendance sur les valeurs passées
  - `d` : degré de différenciation – nombre de fois qu’on différencie la série pour la rendre stationnaire
  - `q` : nombre de termes de moyenne mobile (MA) – dépendance sur les erreurs passées
- **SARIMA(p,d,q)(P,D,Q,s)** ajoute la **saisonnalité** :
  - `P,D,Q` : mêmes concepts que ARIMA mais sur le cycle saisonnier
  - `s` : longueur de la saison (ex. 12 pour données mensuelles)

#### Paramètres principaux
- `order=(p,d,q)` : paramètres ARIMA
- `seasonal_order=(P,D,Q,s)` : paramètres SARIMA pour la saisonnalité
- `trend` : permet d’ajouter une tendance (`'n'`, `'c'`, `'t'`, `'ct'`)
- `exog` : variable(s) exogènes optionnelle(s) (pour SARIMAX)

#### Avantages
- Capture **la tendance et l’autocorrélation** des séries temporelles
- SARIMA permet d’intégrer **la saisonnalité**
- Modèle probabiliste, fournit des intervalles de confiance pour les prédictions
- Robuste pour des séries longues et régulières

#### Inconvénients
- Nécessite que la série soit **stationnaire** (dépend souvent de transformations)
- Sélection des paramètres `p,d,q,P,D,Q` souvent complexe
- Ne gère pas directement les variables exogènes multiples dans ARIMA (SARIMAX le permet)
- Moins flexible pour des relations non linéaires ou séries très bruitées
- 
```python
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

# ARIMA simple
arima_model = ARIMA(df['consommation'], order=(1,1,1)).fit()
df['ARIMA'] = arima_model.fittedvalues
df[['consommation','ARIMA']].plot(figsize=(12,4), title='ARIMA(1,1,1)')
plt.show()

# SARIMA (avec saisonnalité 12 mois)
sarima_model = SARIMAX(df['consommation'], order=(1,1,1), seasonal_order=(1,1,1,12)).fit()
df['SARIMA'] = sarima_model.fittedvalues
df[['consommation','SARIMA']].plot(figsize=(12,4), title='SARIMA(1,1,1)x(1,1,1,12)')
plt.show()
```

## SARIMAX avec variable exogène

#### Explication
Le modèle **SARIMAX (Seasonal ARIMA with eXogenous variables)** est une extension de SARIMA qui permet d'intégrer des **variables explicatives externes** (exogènes) dans la modélisation.  
- Cela permet de capturer l'effet de facteurs extérieurs sur la série temporelle.  
- Exemple : la consommation énergétique peut être influencée par la **température**, les **prix de l'énergie**, ou d'autres indicateurs climatiques ou économiques.

#### Paramètres principaux
- `order=(p,d,q)` : paramètres ARIMA (auto-régressif, différenciation, moyenne mobile)
- `seasonal_order=(P,D,Q,s)` : paramètres saisonniers
- `exog` : tableau des variables exogènes à intégrer
- `trend` : optionnel, permet d'ajouter une tendance

#### Avantages
- Intègre la **saisonnalité** et les **facteurs externes**
- Fournit des prédictions plus précises lorsque la série est influencée par des variables exogènes
- Modèle probabiliste avec intervalles de confiance
- Compatible avec les séries mensuelles, trimestrielles, annuelles, etc.

#### Inconvénients
- Comme ARIMA/SARIMA, la série doit être **stationnaire** ou différenciée
- La qualité des prédictions dépend fortement des variables exogènes choisies
- Sélection des paramètres ARIMA et saisonniers reste complexe
- Plus lourd à ajuster et à calculer que SARIMA simple

```python
# Simulons une variable exogène : température moyenne mensuelle
temperature = 10 + 10 * np.sin(2*np.pi*np.arange(n_years*12)/12) + np.random.normal(0,2,n_years*12)

sarimax_model = SARIMAX(df['consommation'], order=(1,1,1), seasonal_order=(1,1,1,12), exog=temperature).fit()
df['SARIMAX'] = sarimax_model.fittedvalues
df[['consommation','SARIMAX']].plot(figsize=(12,4), title='SARIMAX avec température')
plt.show()
```

## Prophet

#### Explication
**Prophet** est une librairie développée par Facebook pour la prévision de séries temporelles.  
Elle est particulièrement adaptée aux séries avec **tendance, saisonnalité** et **jours fériés ou événements spécifiques**.  
- Elle décompose la série en **tendance + saisonnalité + jours fériés**.
- Très pratique pour des séries irrégulières ou avec des valeurs manquantes.

#### Paramètres principaux
- `yearly_seasonality=True/False` : inclure ou non la saisonnalité annuelle
- `weekly_seasonality`, `daily_seasonality` : ajout d'autres composantes saisonnières
- `holidays` : tableau de jours fériés à prendre en compte
- `changepoint_prior_scale` : sensibilité aux changements de tendance

#### Avantages
- Facile à utiliser, nécessite peu de prétraitement
- Gère les **séries non stationnaires** sans différenciation
- Fournit des **intervalles de confiance** sur les prévisions
- Permet d’inclure des **événements externes** (holidays, promotions…)

#### Inconvénients
- Moins précis que ARIMA/SARIMA pour des séries très régulières et stationnaires
- Hyperparamètres peuvent nécessiter un ajustement manuel pour améliorer la précision
- Moins transparent sur les détails internes du modèle par rapport à ARIMA/SARIMA

```python
from prophet import Prophet

# Préparer le dataframe pour Prophet
df_prophet = df.reset_index().rename(columns={'date':'ds', 'consommation':'y'})

prophet_model = Prophet(yearly_seasonality=True)
prophet_model.fit(df_prophet)

future = prophet_model.make_future_dataframe(periods=12, freq='M')
forecast = prophet_model.predict(future)

prophet_model.plot(forecast)
plt.title('Prévision avec Prophet')
plt.show()
```


## Time series avec scikit-learn

Pour utiliser des modèles classiques (Random Forest, Gradient Boosting, etc.) sur des séries temporelles, on transforme la série en **features lag** :

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Créer des features lag
df_ml = df.copy()
for lag in range(1,13):
    df_ml[f'lag_{lag}'] = df_ml['consommation'].shift(lag)
df_ml.dropna(inplace=True)

X = df_ml.drop(columns='consommation')
y = df_ml['consommation']

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=12)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

print("RMSE :", np.sqrt(mean_squared_error(y_test, y_pred)))

# Visualisation
plt.figure(figsize=(12,4))
plt.plot(y_test.index[-24:], y_test[-24:], label='Vrai')
plt.plot(y_test.index[-24:], y_pred[-24:], label='Prévision RF')
plt.legend()
plt.show()
```

## Points clés

- Les séries temporelles peuvent se décomposer en **tendance, saisonnalité et bruit**
- **Lissage** et **décomposition** aident à visualiser les patterns
- ARIMA/SARIMA/SARIMAX sont adaptés pour séries stationnaires et saisonnières
- Prophet est facile à utiliser et robuste aux irrégularités
- Avec scikit-learn, il faut créer des **lag features** pour appliquer des modèles classiques

Cette simulation peut servir de base pour analyser la **consommation énergétique réelle des logements** et tester différents modèles de prévision.

## Aller plus loin avec LSTM (Long Short-Term Memory)

Les **LSTM** sont un type de réseau de neurones récurrents (RNN) capables de capturer les **dépendances à long terme** dans les séries temporelles.  
- Idéal pour des séries avec **tendances complexes, saisonnalités multiples et effets de mémoire longue**.
