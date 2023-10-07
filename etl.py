# %
import pandas as pd
import numpy as np
from pathlib import Path
import sweetviz as sv
# %
master_dir = Path.cwd()
data_dir = master_dir/'data'/'H_M-Sales-2019'
# %
articles = pd.read_csv(data_dir/'articles.csv')
customers = pd.read_csv(data_dir/'customers.csv')
trans = pd.read_csv(data_dir/'transactions_train-002.csv')

# %
# #! cleaning and analysing utils
my_report_c = sv.analyze(customers)
# my_report.show_html()
my_report_a = sv.analyze(articles)
# my_report.show_html()
my_report_t = sv.analyze(trans)
# my_report.show_html()
# my_report.show_notebook(  w=None,
#                h=None,
#                scale=None,
#                layout='widescreen',
#                filepath=None)
# %
# articles = clean_headers(articles)
# customers = clean_headers(customers)
# trans = clean_headers(trans)
# %
#! make sure all the data types are correct before doing this!! IDs should be objects
articles.dtypes
articles.loc[:, ['article_id', 'product_code']] = articles.loc[:,
                                                               ['article_id', 'product_code']].astype('object')
# %
# articles = clean_df(articles, report=False)
# customers = clean_df(customers, report=False)
# trans = clean_df(trans, report=False)
# % dummy util
# %
#! MORE CLEANING FOR MODELLING
# add the potential predictors to a df X_tmp
X_tmp = pd.DataFrame()

#! feat engeneering


def add_dummies_to_X(X: pd.DataFrame, cat_var: str, prefix_str: str) -> pd.DataFrame:
    """
    This function creates dummy variables for a specific categorical variable in a DataFrame and then drops the original categorical variable.

    Arguments:
        X: The DataFrame to which the dummy variables will be added.
        cat_var: The categorical variable in the DataFrame X for which the dummy variables will be created.
        prefix_str: The prefix for the dummy variable column names.

    Returns:
        The DataFrame X with the dummy variables added and the original categorical variable dropped.
    """
    X = pd.concat([X, pd.get_dummies(X[cat_var],
                  prefix=prefix_str)], axis=1).drop(cat_var, axis=1)
    return X


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
