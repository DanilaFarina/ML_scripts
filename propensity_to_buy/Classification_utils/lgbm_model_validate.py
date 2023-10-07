### - The main approaches that come into my mind are:
# 1) Build a tree based model (using boosting or bagging) using features built in etl pipeline
# 2) Use survival analysis technique to predict the end time series
# I am going to use a tree based approach because I am more familiar with the code
# but some research on the second would be interesting

#%%
import pandas as pd
import numpy as np
import pathlib as p
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import lightgbm as lgb
import matplotlib.pyplot as plt
import utils

# if we are then let's change the parameter
# look at feat importance

master_path = p.Path.cwd()
gold_path = master_path / "gold"
data = pd.read_parquet(gold_path / "features_bikes_stations_weather.parquet")
data.hpcp_rank.fillna(0, inplace=True)
features = [
    x for x in data.columns.tolist() if x not in ["start_date", "duration"]
]
X = data[features]
y = data["duration"]

# simple train test split to have a first idea
x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
# create validation set of 1%
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.9)
# then oof set up to see if we are overfitting
#%%
## LighGBM
########################################################
params = {
    "task": "train",
    "boosting": "gbdt",
    "random_state": 4590,
    "objective": "regression",
    "feature_pre_filter": "false",
    "feature_fraction": 0.6,
    "bagging_fraction": 1.0,
    "num_leaves": 50,
    "bagging_freq": 0,
    "min_child_samples": 5,
    "metric": "MSE",
}

#######################################################
# first test - lgb training

#########################################################
#%%
#########################################################
# prediction
lgb_clf = utils.lgb_training(x_train, x_test, y_train, y_test, p=params)
y_pred = lgb_clf.predict(x_test, num_iteration=lgb_clf.best_iteration)
utils.regression_report(y_test, y_pred)
# MSE is really high
# MSE: 3540887.32
# RMSE: 1881.72
# no tuning
# we could try and tune the model with optuna
#############################################
# lgb feature importance
lgb.plot_importance(lgb_clf)
#%%
to_drop = ['registered','hpcp_rank','hpcp_bi','year']
x_train, x_test, y_train, y_test = train_test_split(X.drop(to_drop,axis=1), y, train_size=0.8)
# create validation set of 1%
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.9)
# %%
lgb_clf = utils.lgb_training(x_train, x_test, y_train, y_test, p=params)
y_pred = lgb_clf.predict(x_test, num_iteration=lgb_clf.best_iteration)
utils.regression_report(y_test, y_pred)
# %%
