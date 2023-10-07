# %
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from collections import Counter
from pathlib import Path
from matplotlib import pyplot
import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from xgboost import XGBClassifier
import xgboost
import shap
# %
master_dir = Path.cwd().parents[0]
data_dir = master_dir/'data'/'H_M-Sales-2019'
data = pd.read_csv(
    data_dir/'customers.csv').sample(n=1000).dropna(subset='age')
trans = pd.read_csv(data_dir/'transactions_train-002.csv', parse_dates=True)
# %
# understand whether your data is balanced or not
transactions = trans.merge(data, on='customer_id', how='right').dropna(
    subset=['age', 'article_id', 'sales_channel_id'])
# %
X = data.set_index('customer_id')[['age', 'FN']].fillna(0).dropna(subset='age')
# %
X = X.join(transactions.groupby('customer_id').count()
           ['t_dat']).dropna(subset=['age'])
# %
y = pd.pivot_table(transactions, index='customer_id', columns='sales_channel_id',
                   values='article_id', aggfunc='count').iloc[:, 0].fillna(0)
X = X.loc[y.index]
# %
y = np.where(y > 0, 1, 0)
# %
# summarize class distribution
counter = Counter(y)
# %
# # estimate scale_pos_weight value
estimate = counter[0] / counter[1]
# %
# define model
# in this case they're pretty balanced we wouldn't need it
# model = XGBClassifier(scale_pos_weight=0.85)
model = XGBClassifier(method='isotonic')
# define evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
# summarize performance
print('Mean ROC AUC: %.5f' % np.mean(scores))
# %

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1)

# Define the search space
space = {
    'n_estimators': hp.quniform('n_estimators', 50, 1000, 25),
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.5),
    'max_depth':  hp.choice('max_depth', range(1, 14, 1)),
    'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
    'subsample': hp.uniform('subsample', 0.1, 1),
    'gamma': hp.uniform('gamma', 0.1, 0.5),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.1, 1)
}

# Define the objective function


def objective(space):
    model = XGBClassifier(
        n_estimators=int(space['n_estimators']),
        learning_rate=space['learning_rate'],
        max_depth=int(space['max_depth']),
        min_child_weight=space['min_child_weight'],
        subsample=space['subsample'],
        gamma=space['gamma'],
        colsample_bytree=space['colsample_bytree']
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    auc = roc_auc_score(y_test, preds)

    return {'loss': -auc, 'status': STATUS_OK}


# Run the algorithm
trials = Trials()
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=100,
            trials=trials)
# %
model = XGBClassifier(n_estimators=int(best['n_estimators']),
                      learning_rate=best['learning_rate'],
                      max_depth=int(best['max_depth']),
                      min_child_weight=best['min_child_weight'],
                      subsample=best['subsample'],
                      gamma=best['gamma'],
                      colsample_bytree=best['colsample_bytree']
                      )
# define evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
# summarize performance
print('Mean ROC AUC: %.5f' % np.mean(scores))
# %
# for more model evalation https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/
#! look in sklearn as well you can plot confidence matrix, think of also of the best metrics to use etc..

# train an XGBoost model
X, y = shap.datasets.boston()
model = xgboost.XGBRegressor().fit(X, y)

# explain the model's predictions using SHAP
# (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)
explainer = shap.Explainer(model)
shap_values = explainer(X)

# visualize the first prediction's explanation
shap.plots.waterfall(shap_values[0])
# visualize all the training set predictions
shap.plots.force(shap_values)
# summarize the effects of all the features
shap.plots.beeswarm(shap_values)
