# %
from sklearn.metrics import mean_squared_error
from scipy.sparse.linalg import svds
import numpy as np
from fastdist import fastdist
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from itertools import permutations
import random

tqdm.pandas()
master_dir = Path.cwd()
data_dir = master_dir/'data'/'H_M-Sales-2019'
trans = pd.read_csv(data_dir/'transactions_train-002.csv')
articles = pd.read_csv(data_dir/'articles.csv')
unique_customers = trans['customer_id'].unique()
sampled_customers = random.sample(list(unique_customers), 200)
sampled_transactions = trans[trans['customer_id'].isin(sampled_customers)]
# %
merged = sampled_transactions.drop('sales_channel_id', axis=1).merge(
    articles[['article_id', 'product_type_name']], on='article_id', how='inner')
del trans
# %


def clean_text(df, col):
    df.loc[:, col] = df.loc[:, col].str.lower(
    ).str.strip().str.replace('  ', ' ')
    return df


def clean_headers(df):
    # Lowercase all headers
    df.columns = df.columns.str.lower()
    # Remove symbols and replace spaces with underscores
    df.columns = df.columns.str.replace(' ', '_', regex=True)
    df.columns = df.columns.str.replace(r'\W', '', regex=True)
    return df


merged = clean_text(merged, 'prod_name')
# %

# Now you can apply your 'create_pairs' function


def create_pairs(x, list_cols):
    return pd.DataFrame(list(permutations(x.values, 2)),
                        columns=list_cols)


grouped = merged.groupby('customer_id')['prod_name']
valid_ids = grouped.size()[grouped.size() > 1].index

merged = merged[merged['customer_id'].isin(valid_ids)]

# Now you can apply your 'create_pairs' function
grouped = merged.groupby('customer_id')['prod_name']
pairs = grouped.progress_apply(create_pairs, list_cols=['prod_1', 'prod_2'])
pairs = pairs.query('prod_1 != prod_2')
# grouped.swifter.apply(create_pairs)
# %  #! COUNT HOW MANY TIMES THEY HAVE BEEN BOUGHT TOGETHER
# %
pairs_counts = pairs.groupby(['prod_1', 'prod_2']).size(
).reset_index().sort_values(by=0, ascending=False)
# % #! ask the dataset what is the item that has been bought the most with hudson shorts:
print(
    f"""You might also like: {pairs_counts.query('prod_1 == "hudson shorts"').iloc[0,1]}""")

# %
#! CONTENT BASED !!!
articles.columns
# %
# prepare the df
# Select the relevant columns for the recommendation system
selected_columns = [
    'article_id',
    'colour_group_name',
    'perceived_colour_value_name',
    'index_name',
    'garment_group_name',
]

# Create a new DataFrame with the selected columns
selected_df = articles[selected_columns]

# Use pd.get_dummies to create binary (dummy) variables for categorical columns
dummy_df = pd.get_dummies(selected_df, columns=[
    'colour_group_name',
    'perceived_colour_value_name',
    'index_name',
    'garment_group_name',
])

# Group by 'article_id' and aggregate the dummy variables using max to get binary values
binary_df = dummy_df.groupby('article_id').max().reset_index().astype(int)

# If you want to reset the index to 'article_id', you can use the following line
# binary_df.set_index('article_id', inplace=True)
# %%
# Display the resulting binary DataFrame
binary_df.astype('category').describe()

# %
# this is way too slow
# from scipy.spatial.distance import pdist, squareform
# jaccard_distances= pdist(binary_df.values, metric='jaccard')
# sqjacc= squareform((jaccard_distances))
# distance_df = pd.DataFrame(1- sqjacc, index=selected_df['article_id'],
#                           columns=selected_df['article_id'])
# %

# Compute Jaccard similarities # we can only do this if we don't have a big dataset!!
jaccard_similarities = fastdist.matrix_pairwise_distance(
    a=binary_df.values, metric=fastdist.hamming, metric_name='jaccard')

# %
# we can work also with descriptions, we can embed or use tfid to create a similar dataset then do recommendations using cos similarity.
# we can then build a userprofile taking the average of the columns
# then it's very useful because we can create a list of all the books that the user has not yet read and then calculate the cosine distance between the user profile and that matrix
# if you sort this, you will get the most similar items to the user preferences based on the user's full history

#! COLLABORATIVE FILTERING with matrix factorisation !!!
sampled_transactions
# % deal with sparsity
users_purch_df = merged.pivot_table(
    index='customer_id', columns='product_type_name', values='t_dat', aggfunc='count')
# users_purch_df_centered= users_purch_df.sub(users_purch_df.median(axis=1),axis=0)
users_purch_df.fillna(0, inplace=True)
# users_purch_df_centered.reset_index()
# if it's price we could just fill it up with 0
# %
U, sigma, Vt = svds(users_purch_df.values)
sigma = np.diag(sigma)
# % # recreate the full matrix and give it the original scale
recalculated_ratings = np.dot(
    np.dot(U, sigma), Vt) + users_purch_df.mean(axis=1).values.reshape(-1, 1)
pd.DataFrame(recalculated_ratings, columns=users_purch_df.columns,
             index=users_purch_df.index)
# % #if you would like to evaluate the predictions we need to hold out a set and repeat
# %
actual_values = users_purch_df.iloc[:20, :100].values
users_purch_df.iloc[:20, :100] = np.nan
# %
users_purch_df.fillna(0, inplace=True)
# users_purch_df_centered.reset_index()
# if it's price we could just fill it up with 0
# %
U, sigma, Vt = svds(users_purch_df.values)
sigma = np.diag(sigma)
# % # recreate the full matrix and give it the original scale
recalculated_ratings = np.dot(
    np.dot(U, sigma), Vt) + users_purch_df.mean(axis=1).values.reshape(-1, 1)
predicted_values = recalculated_ratings[:20, :100]
# %
mask = ~np.isnan(actual_values)
# %
print(mean_squared_error(
    actual_values[mask], predicted_values[mask], squared=False))
# %
# check out fastFM and LightFM for recommendations!
