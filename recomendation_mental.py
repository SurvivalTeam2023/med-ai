import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse as sp
from sklearn.metrics import pairwise as pw
from popular_recomendation import popularity_recommender
from lightfm import LightFM
from fastapi import FastAPI, HTTPException

sns.set()
pd.set_option("display.max_columns", None)


def create_mental_dict(interactions):
    mental_id = list(interactions.index)
    mental_dict = {}
    counter = 0

    for i in mental_id:
        mental_dict[i] = counter
        counter += 1

    new_dict = dict([(value, key) for key, value in mental_dict.items()])

    return new_dict


# Function to create an item dictionary based on their item_id and item name
def create_item_dict(df, id_col, name_col):
    item_dict = {}

    for i in range(df.shape[0]):
        item_dict[(df.loc[i, id_col])] = df.loc[i, name_col]

    return item_dict


# Function to produce mental recommendations
def find_key_by_value(dictionary, value):
    for key, val in dictionary.items():
        if str(val) == str(value):
            return key
    raise HTTPException(status_code=404, detail="No data")


def sample_recommendation_mental(
    model,
    interactions,
    mental_id,
    mental_dict,
    item_dict,
    threshold=0,
    nrec_items=10,
    show=True,
):
    try:
        n_mentals, n_items = interactions.shape
        mental_x = find_key_by_value(mental_dict, mental_id)
        scores = pd.Series(model.predict(mental_x, np.arange(n_items)))
        scores.index = interactions.columns
        scores = list(pd.Series(scores.sort_values(ascending=False).index))
        known_items = list(
            pd.Series(
                interactions.loc[mental_id, :][
                    interactions.loc[mental_id, :] > threshold
                ].index
            ).sort_values(ascending=False)
        )

        scores = [x for x in scores if x not in known_items]
        return_score_list = scores[0:nrec_items]
        known_items = list(pd.Series(known_items).apply(lambda x: item_dict[x]))
        scores = list(pd.Series(return_score_list).apply(lambda x: item_dict[x]))

        if show == True:
            print("Recommended songs for mentalID:", mental_id)
            counter = 1
            for i in scores:
                print(str(counter) + "- " + str(i))
                counter += 1

        return return_score_list
    except Exception as e:
        raise HTTPException(status_code=404, detail="No data for this mental")


def create_item_emdedding_distance_matrix(model, interactions):
    df_item_norm_sparse = sp.csr_matrix(model.item_embeddings)
    similarities = pw.cosine_similarity(df_item_norm_sparse)
    item_emdedding_distance_matrix = pd.DataFrame(similarities)
    item_emdedding_distance_matrix.columns = interactions.columns
    item_emdedding_distance_matrix.index = interactions.columns

    return item_emdedding_distance_matrix


# Function to create item-item recommendation
def item_item_recommendation(
    item_emdedding_distance_matrix, item_id, item_dict, n_items=10, show=True
):
    recommended_items = list(
        pd.Series(
            item_emdedding_distance_matrix.loc[find_key_by_value(item_dict, item_id), :]
            .sort_values(ascending=False)
            .head(n_items + 1)
            .index[1 : n_items + 1]
        )
    )
    print(item_dict)
    if show == True:
        print("Song of interest: {0}".format(item_id))
        print("Song(s) similar to the above item are as follows:-")
        counter = 1

        for i in recommended_items:
            print(str(counter) + ". " + str(i))
            counter += 1

    return recommended_items
