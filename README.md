# Titanic Survival Prediction

A machine learning project predicting passenger survival on the Titanic, built for the [Kaggle Titanic competition](https://www.kaggle.com/competitions/titanic).

## Overview
This project uses a Random Forest classifier to predict whether a passenger survived the Titanic disaster, based on features like class, sex, age, and family relationships. Beyond the standard dataset features, I engineered several additional ones to improve prediction accuracy.

## Approach

**Feature Engineering**
- **Title extraction** — extracted passenger titles (Mr, Mrs, Miss, Master, etc.) from name strings using regex, grouping rare titles (Countess, Capt, Don, etc.) into a single "Rare" category
- **Ticket-group size** — grouped passengers sharing a ticket number to estimate travel-group size beyond official family relationships (SibSp/Parch)
- **Family size & IsAlone** — combined sibling/spouse and parent/child counts into a total family size, plus a binary "travelling alone" flag
- **Title-based age imputation** — filled missing ages using the median age within each title group (e.g. "Master" → young age) rather than a single global median, for more accurate estimates

**Model**
- Random Forest Classifier (scikit-learn), trained with deliberately constrained hyperparameters (`max_depth=5`, `min_samples_leaf=5`) to prioritize generalization over overfitting to the leaderboard, rather than using grid search

## Result
Achieved **78.7% accuracy** on the Kaggle public leaderboard.

## Tools
Python, pandas, NumPy, scikit-learn

## Files
- `titanic-survival-prediction.ipynb` — full notebook with data cleaning, feature engineering, model training, and prediction
