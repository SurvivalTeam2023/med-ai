from popular_mental_recomendation import popularity_mental_recommender
from recomendation import (
    create_user_dict,
    create_item_dict,
    sample_recommendation_user,
    item_item_recommendation,
    create_item_emdedding_distance_matrix,
    create_genre_dict,
)
import seaborn as sns
import pandas as pd
import os
import glob
from lightfm.evaluation import precision_at_k, recall_at_k, auc_score, reciprocal_rank
from scipy import sparse as sp
from popular_recomendation import popularity_recommender
from lightfm import LightFM

from recomendation_mental import sample_recommendation_mental
from recomendation_mental_health_degree import (
    sample_recommendation_mental_health_degree,
)
from popular_mental_health_degree_recomendation import (
    popularity_mental_health_degree_recommender,
)

sns.set()
pd.set_option("display.max_columns", None)
PREFIX_TRAIN_MUSIC_MODEL = "train_model_music_"
PREFIX_TRAIN_GENRE_MODEL = "train_model_mental_"


def load_model_latest_version(prefix):
    current_path = os.getcwd()
    file_pattern = os.path.join(current_path + "/train_model", prefix + "*.txt")
    files = glob.glob(file_pattern)
    if not files:
        print("No files found with the specified prefix:", prefix)
        exit()
    files.sort(key=os.path.getmtime, reverse=True)
    latest_file = files[0]
    return pd.read_csv(latest_file, delimiter="\t")


def get_audio_ids_recommend_by_user_id(user_id):
    data_model = load_model_latest_version(PREFIX_TRAIN_MUSIC_MODEL)
    if data_model is None:
        return
    raw_data = data_model.head(50000)
    data = raw_data.groupby(["audio_name"]).agg({"count": "count"}).reset_index()
    data["percentage"] = raw_data["count"].div(raw_data["count"].sum()) * 100
    pop_model = popularity_recommender()
    pop_model.create(raw_data, "user_id", "audio_name")
    x = raw_data.pivot_table(index="user_id", columns="audio_id", values="count")
    x_nan = x.fillna(0)
    interaction = sp.csr_matrix(x_nan.values)
    hybrid_model = LightFM(loss="warp-kos", n=20, k=20, learning_schedule="adadelta")
    hybrid_model.fit(interaction, epochs=30, num_threads=6)
    user_dict = create_user_dict(interactions=x)
    song_dict = create_item_dict(df=raw_data, id_col="audio_id", name_col="audio_id")
    return (
        sample_recommendation_user(
            model=hybrid_model,
            interactions=x,
            user_id=user_id,
            user_dict=user_dict,
            item_dict=song_dict,
            threshold=5,
            nrec_items=50,
            show=True,
        ),
    )


def get_audio_similar_with_song_id(audio_id):
    data_model = load_model_latest_version(PREFIX_TRAIN_MUSIC_MODEL)
    raw_data = data_model.head(50000)
    data = raw_data.groupby(["audio_name"]).agg({"count": "count"}).reset_index()
    data["percentage"] = raw_data["count"].div(raw_data["count"].sum()) * 100
    pop_model = popularity_recommender()
    pop_model.create(raw_data, "user_id", "audio_name")
    x = raw_data.pivot_table(index="user_id", columns="audio_id", values="count")
    x_nan = x.fillna(0)
    interaction = sp.csr_matrix(x_nan.values)
    hybrid_model = LightFM(loss="warp-kos", n=20, k=20, learning_schedule="adadelta")
    hybrid_model.fit(interaction, epochs=30, num_threads=6)
    song_dict = create_item_dict(df=raw_data, id_col="audio_id", name_col="audio_id")
    song_item_dist = create_item_emdedding_distance_matrix(
        model=hybrid_model, interactions=x
    )
    return (
        item_item_recommendation(
            item_emdedding_distance_matrix=song_item_dist,
            item_id=int(audio_id),
            item_dict=song_dict,
            n_items=10,
        ),
    )


def get_audio_ids_recommend_by_mental_id(mental_id):
    data_model = load_model_latest_version(PREFIX_TRAIN_GENRE_MODEL)
    raw_data = data_model.head(50000)
    data = raw_data.groupby(["audio_id"]).agg({"audio_count": "count"}).reset_index()
    data["percentage"] = (
        raw_data["audio_count"].div(raw_data["audio_count"].sum()) * 100
    )
    pop_model = popularity_mental_recommender()
    pop_model.create(raw_data, "mental_id", "audio_id")
    x = raw_data.pivot_table(
        index="mental_id", columns="audio_id", values="audio_count"
    )
    x_nan = x.fillna(0)
    interaction = sp.csr_matrix(x_nan.values)
    hybrid_model = LightFM(loss="warp-kos", n=20, k=20, learning_schedule="adadelta")
    hybrid_model.fit(interaction, epochs=30, num_threads=6)
    mental_dict = create_genre_dict(interactions=x)
    song_dict = create_item_dict(df=raw_data, id_col="audio_id", name_col="audio_id")
    return (
        sample_recommendation_mental(
            model=hybrid_model,
            interactions=x,
            mental_id=int(mental_id),
            mental_dict=mental_dict,
            item_dict=song_dict,
            threshold=50,
            nrec_items=50,
            show=True,
        ),
    )


def get_audio_ids_recommend_by_mental_health_degree(mental_health_degree_id, mental_id):
    print(mental_health_degree_id, mental_id)
    data_model = load_model_latest_version(PREFIX_TRAIN_GENRE_MODEL)
    raw_data = data_model.head(100000)
    data_filter = raw_data[(raw_data["mental_id"] == mental_id)].reset_index(drop=True)
    data = data_filter.groupby(["audio_id"]).agg({"audio_count": "count"}).reset_index()
    data["percentage"] = (
        data_filter["audio_count"].div(data_filter["audio_count"].sum()) * 100
    )
    pop_model = popularity_mental_health_degree_recommender()
    pop_model.create(data_filter, "mental_health_degree_id", "audio_id")
    x = data_filter.pivot_table(
        index="mental_health_degree_id", columns="audio_id", values="audio_count"
    )
    x_nan = x.fillna(0)
    interaction = sp.csr_matrix(x_nan.values)
    hybrid_model = LightFM(loss="warp-kos", n=20, k=20, learning_schedule="adadelta")
    hybrid_model.fit(interaction, epochs=30, num_threads=6)
    mental_dict = create_genre_dict(interactions=x)
    song_dict = create_item_dict(df=data_filter, id_col="audio_id", name_col="audio_id")
    return (
        sample_recommendation_mental_health_degree(
            model=hybrid_model,
            interactions=x,
            mental_health_degree_id=int(mental_health_degree_id),
            mental_dict=mental_dict,
            item_dict=song_dict,
            threshold=30,
            nrec_items=50,
            show=True,
        ),
    )
