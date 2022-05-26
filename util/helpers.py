from collections import Counter
from surprise import Dataset, Reader
# from datetime import datetime as dt
import pandas as pd
import os


# Function to return the correct dataframe form for line/box plots
def melt_df(dataframe, nbhd_pair):
    return dataframe[nbhd_pair].reset_index().melt(id_vars='index').rename(columns=str.title)


# Fnction to group elements based on frequency
def group_list(lst):
    return list(zip(Counter(lst).keys(), Counter(lst).values()))


def load_train_test_surpriselib(x, y):
    # load trainset from the "dataset" varible which we eliminated ratings from
    train_data = Dataset.load_from_df(
        x[['user_id', 'item_id', 'rating']],
        Reader(rating_scale=(x.rating.min(), x.rating.max()))
    )
    trainset = train_data.build_full_trainset()
    testset = list(y[['user_id', 'item_id', 'rating']].to_records(index=False))
    print("trainset and testset successfully created.")

    return trainset, testset


def load_dataset_explicit(ds_name, ds_path, total_users=10000):
    ratings_file_name = 'ratings.csv'
    ratings_path = os.path.join(ds_path, ratings_file_name)

    ratings = 0
    colnames = ['user_id', 'item_id', 'rating', 'timestamp']

    if ds_name == 'ml-1m':
        ratings_file_name = 'ratings.dat'
        ratings_path = os.path.join(ds_path, ratings_file_name)
        ratings = pd.read_csv(ratings_path, delimiter='::', names=colnames)

    elif ds_name == 'ml-latest-small':
        ratings = pd.read_csv(ratings_path).rename({'movieId': 'item_id', 'userId': 'user_id'}, axis=1)
        # ratings['date'] = ratings['timestamp'].apply(lambda x: dt.fromtimestamp(x).date())

    elif ds_name == 'ml-latest':
        ratings = pd.read_csv(ratings_path).rename({'movieId': 'item_id', 'userId': 'user_id'}, axis=1)
        # define the users to extract from the dataset (1 -> n)
        target_users = list(range(1, total_users))
        ratings_small = ratings[ratings.user_id.isin(target_users)]
        ratings = ratings_small

    elif ds_name == 'personality':
        ratings = pd.read_csv(ratings_path)\
            .rename({' movie_id': 'item_id', 'useri': 'user_id', ' rating': 'rating'}, axis=1)
        ratings["user_id"] = ratings["user_id"].astype('category')
        ratings["user_cat"] = ratings["user_id"].cat.codes
        # clean the ratings dataframe
        ratings = ratings\
            .drop(['user_id'], axis=1)\
            .rename({'user_cat': 'user_id'}, axis=1)

    # encoding reference: https://pbpython.com/categorical-encoding.html
    elif ds_name == 'amazon-2':
        ratings = pd.read_csv(ratings_path, names=colnames)
        ratings["user_id"] = ratings["user_id"].astype('category')
        ratings["user_cat"] = ratings["user_id"].cat.codes
        ratings["item_id"] = ratings["item_id"].astype('category')
        ratings["item_cat"] = ratings["item_id"].cat.codes
        # clean the ratings dataframe
        ratings = ratings\
            .drop(['user_id', 'item_id'], axis=1)\
            .rename({'user_cat': 'user_id', 'item_cat': 'item_id'}, axis=1)

    return ratings
