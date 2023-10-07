# %
from sklearn.model_selection import cross_val_score
from sklearn.utils import resample
from sklearn.model_selection import KFold
from functools import partial
from pathlib import Path
from tkinter import Y

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from tqdm import tqdm

# %%
tqdm.pandas()
master_dir = Path.cwd().parents[0]
data_dir = master_dir/'data'/'H_M-Sales-2019'
trans = pd.read_csv(data_dir/'transactions_train-002.csv', parse_dates=True)
trans['t_dat'] = pd.to_datetime(trans.t_dat)
# articles = pd.read_csv(data_dir/'articles.csv')
# %
daily_price = trans[['t_dat', 'price']].groupby('t_dat').mean()
# daily_price.plot(style='k.')
daily_price.plot()
# %
# timeseries = trans.set_index(pd.to_datetime(daily_price['t_dat']))['price']
trans = trans.assign(year=trans.t_dat.dt.year)
yearly_price = trans[['year', 'price']].groupby('year').mean()
# %
trans = trans.assign(month=trans.t_dat.dt.month)
monthly_price = trans[['year', 'month', 'price']
                      ].groupby(['year', 'month']).mean()
# %
window_size = 30
windowed = daily_price.rolling(window=window_size)
price_smooth = windowed.mean()
# %
# to fill missing values we can use interpolations or we can calculate the window or calculating the %change
# to find outliers a common way is to find points that are mopre than 3 dtandard deviations away from the mean of the dayaset
#! Interpolation


def interpolate_and_plot(prices, interpolation):

    # Create a boolean mask for missing values
    missing_values = prices.isna()

    # Interpolate the missing values
    prices_interp = prices.interpolate(interpolation)

    # Plot the results, highlighting the interpolated values in black
    fig, ax = plt.subplots(figsize=(10, 5))
    prices_interp.plot(color='k', alpha=.6, ax=ax, legend=False)

    # Now plot the interpolated values on top in red
    prices_interp[missing_values].plot(ax=ax, color='r', lw=3, legend=False)
    plt.show()


interpolation_type = 'linear'  # look at pandas type
interpolate_and_plot(daily_price, interpolation_type)
# %%


def percent_change(series):
    # Collect all *but* the last value of this window, then the final value
    previous_values = series[:-1]
    last_value = series[-1]

    # Calculate the % difference between the last value and the mean of earlier values
    percent_change = (last_value - np.mean(previous_values)
                      ) / np.mean(previous_values)
    return percent_change


# % # we could pivot the data to observe the time series by product or by platforms
# %%
prices_perc = daily_price.rolling(30).pipe(percent_change)
# careful you might want to use only recent data! so you'd subset one year. Or you might want to subset by company, by store, by city, by product etc..
# %


def replace_outliers(series):
    # Calculate the absolute difference of each timepoint from the series mean
    absolute_differences_from_mean = np.abs(series - np.mean(series))

    # Calculate a mask for the differences that are > 3 standard deviations from zero
    this_mask = absolute_differences_from_mean > (np.std(series) * 3)

    # Replace these values with the median accross the data
    series[this_mask] = np.nanmedian(series)
    return series


# Apply your preprocessing function to the timeseries and plot the results
prices_perc = prices_perc.pipe(replace_outliers)
prices_perc.plot()
plt.show()
# %
# # EXTRACT FEATURES
feats = prices_perc.rolling(30).aggregate([np.std, np.max, np.mean]).dropna()
# we could use the percentiles!!! they're each a feature, remember to use the partial function for that
# then date based features: what day is it, is it a holiday, is it weekend
# %
#! the biggest thing that wil impact your model is the autocorrelation, how smooth your data is-> how correlated your time point is
#! with its neighboring ones. this is the extent to which the previous data point will be predictive of the future ones
# we start creating lagged features (shifting the data from the past)
shifts = [0, 1, 2, 3, 4, 5, 6, 7]
lagged_data = {f'lag_{ii}': daily_price['price'].shift(ii) for ii in shifts}
# %%
lagged_data = pd.DataFrame.from_records(lagged_data)
# %
X = lagged_data.fillna(np.nanmedian(daily_price))
y = daily_price['price'].fillna(np.nanmedian(daily_price))

# Fit the model
# you've created time-shifted versions of a single time series, you can fit an auto-regressive model.
# This is a regression model where the input features are time-shifted versions of the output time series data.
# # You are using previous values of a timeseries to predict current values of the same timeseries (thus, it is auto-regressive).
# By investigating the coefficients of this model, you can explore any repetitive patterns that exist in a timeseries, and get an idea for how far in the past a data point is predictive of the future.

model = Ridge()
model.fit(X, y)

# %
# Visualize the output data up to "2011-01"
fig, axs = plt.subplots(2, 1, figsize=(20, 10))
y.plot(ax=axs[0])


def visualize_coefficients(coefs, names, ax):
    # Make a bar plot for the coefficients, including their names on the x-axis
    ax.bar(names, coefs)
    ax.set(xlabel='Coefficient name', ylabel='Coefficient value')

    # Set formatting so it looks nice
    plt.setp(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
    return ax


# Run the function to visualize model's coefficients
visualize_coefficients(model.coef_, X.columns, ax=axs[1])
plt.show()
# how do I interpret the coefficients?
# %
# in crossvalidation we could do a sanity check -> predicted time series have the same time structure as per the previous ones
cv = TimeSeriesSplit(n_splits=10)
fig, ax = plt.subplots()
for ii, (tr, tt) in enumerate(cv.split(X, y)):
    # Plot the training data on each iteration, to see the behavior of the CV
    ax.plot(tr, ii + y[tr])

ax.set(title='Training data on each CV iteration', ylabel='CV iteration')
plt.show()
# %

cv = KFold(n_splits=10, shuffle=False)

# Iterate through CV splits
results = []
# y=y.values.reshape(-1,1)
# X=X.values
for tr, tt in cv.split(X, y):
    # Fit the model on training data
    model.fit(X[tr], y[tr])

    # Generate predictions on the test data and collect
    prediction = model.predict(X[tt])
    results.append((prediction, tt))


# %
# how do we quantify variability in our model and how is this related to time series data?
# A stationary signal is one that does not change their statistical properties over time - most time series are non stationary
daily_price.plot()


def bootstrap_interval(data, percentiles=(2.5, 97.5), n_boots=100):
    """Bootstrap a confidence interval for the mean of columns of a 2-D dataset."""
    # Create empty array to fill the results
    bootstrap_means = np.zeros([n_boots, data.shape[-1]])
    for ii in range(n_boots):
        # Generate random indices for data *with* replacement, then take the sample mean
        random_sample = resample(data)
        bootstrap_means[ii] = random_sample.mean(axis=0)

    # Compute the percentiles of choice for the bootstrapped means
    percentiles = np.percentile(bootstrap_means, percentiles, axis=0)
    return percentiles


# %%
# Iterate through CV splits
n_splits = 100
cv = TimeSeriesSplit(n_splits=n_splits)

# Create empty array to collect coefficients
coefficients = np.zeros([n_splits, X.shape[1]])

for ii, (tr, tt) in enumerate(cv.split(X, y)):
    # Fit the model on training data and collect the coefficients
    model.fit(X[tr], y[tr])
    coefficients[ii] = model.coef_
# Calculate a confidence interval around each coefficient
bootstrapped_interval = bootstrap_interval(coefficients)

# Plot it
fig, ax = plt.subplots()
ax.scatter(lagged_data.columns, bootstrapped_interval[0], marker='_', lw=3)
ax.scatter(lagged_data.columns, bootstrapped_interval[1], marker='_', lw=3)
ax.set(title='95% confidence interval for model coefficients')
plt.setp(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
plt.show()
# %
# cross validation is a good way to assess whether a time series is stationary or not, if the coeffiecients vary too much between the samples then the mdoel has noise and is non stationary
# a method is bootstraping the mean: you make an array of bootstraped means and then you can calculate the percentiles of each (95% confidence interval)


def my_pearsonr(est, X, y):
    # Generate predictions and convert to a vector
    y_pred = est.predict(X).squeeze()
    # Use the numpy "corrcoef" function to calculate a correlation matrix
    my_corrcoef_matrix = np.corrcoef(y_pred, y.squeeze())
    # Return a single correlation value from the matrix
    my_corrcoef = my_corrcoef_matrix[1, 0]
    return my_corrcoef


# Generate scores for each split to see how the model performs over time
scores = cross_val_score(model, X, y, cv=cv, scoring=my_pearsonr)
# %%
# Convert to a Pandas Series object
times_scores = [lagged_data.reset_index().index[tt[0]]
                for tr, tt in cv.split(X, y)]
scores_series = pd.Series(scores, index=times_scores, name='score')

# Bootstrap a rolling confidence interval for the mean score
scores_lo = scores_series.rolling(20).aggregate(
    partial(bootstrap_interval, percentiles=2.5))
scores_hi = scores_series.rolling(20).aggregate(
    partial(bootstrap_interval, percentiles=97.5))
# %


# if we see that there is a problem, how can we fix it? ONLY USE THE LATEST DATAPOINTS for training so restrict the training window when initialising the TimeSeriesSplit with smaller window of points
smaller_window = 100
cv = TimeSeriesSplit(n_splits=10, max_train_size=smaller_window)
# %
