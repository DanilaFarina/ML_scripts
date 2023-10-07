### - The main approaches that come into my mind are:
# 1) Build a tree based model (using boosting or bagging) using features built in etl pipeline
# 2) Use survival analysis techniques to predict the end time series because the survival
# function is interpreted as time to an event
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
import shap 

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
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# then oof set up to see if we are overfitting
#%%
rf_clf = utils.rf_training(x_train, y_train)
y_pred = rf_clf.predict(x_test)
utils.regression_report(y_test, y_pred)

#%%
rf_importances = utils.compute_forest_importance(
    rf_clf, x_test, y_test, features, permutation="on"
)
utils.plot_forest_importance(rf_importances)

#for better feat importance: shapley values
# takes a long time to run
shap.initjs() # JavaScript plots

# Generate the Tree SHAP estimator of Shapley values that corresponds to the Random Forest we built
explainer = shap.TreeExplainer(rf_clf, feature_perturbation='interventional')
shap_values = explainer.shap_values(x_test)
#shap.summary_plot(shap_values, x_test)
cmap = plt.get_cmap("Dark2")
shap.summary_plot(shap_values, x_test, plot_type="bar", color=cmap.colors[2])
#%%
# could use hyperopt to do the hyperparameters tuning
# remove features that have no importance and look at OOF set up
to_drop = ['registered','hpcp_rank','hpcp_bi','year_2012','year_2011','year_2013']
# an improvement in the performance would be to fit an anova to see which feat carry more info
mean_s, std_s, scores = utils.OOF_Predictions(
    np.array(X.drop(to_drop,axis=1)),
    np.array(y),
    model=RandomForestRegressor(n_estimators=100, random_state=222),
)
########################################################

#%%
#For better understanding feat importance

# features importance?

# %%
