from recomendation import (
    create_user_dict,
    create_item_dict,
    sample_recommendation_user,
    item_item_recommendation,
    create_item_emdedding_distance_matrix,
)
import seaborn as sns
import pandas as pd
import os
import glob
from lightfm.evaluation import precision_at_k, recall_at_k, auc_score, reciprocal_rank
from scipy import sparse as sp
from popular_recomendation import popularity_recommender
from lightfm import LightFM

sns.set()
pd.set_option("display.max_columns", None)


def load_model_latest_version():
    current_path = os.getcwd()
    prefix = "model_"
    file_pattern = os.path.join(current_path + "/train_model", prefix + "*.txt")
    files = glob.glob(file_pattern)
    if not files:
        print("No files found with the specified prefix:", prefix)
        exit()
    files.sort(key=os.path.getmtime, reverse=True)
    latest_file = files[0]
    return pd.read_csv(latest_file, delimiter="\t")


def get_audio_ids_recommend_by_user_id(user_id):
    data_model = load_model_latest_version()
    raw_data = data_model.head(50000)
    data = raw_data.groupby(["audio_name"]).agg({"count": "count"}).reset_index()
    data["percentage"] = raw_data["count"].div(raw_data["count"].sum()) * 100
    # users = raw_data["user_id"].unique()
    pop_model = popularity_recommender()
    pop_model.create(raw_data, "user_id", "audio_name")
    x = raw_data.pivot_table(index="user_id", columns="audio_id", values="count")
    x_nan = x.fillna(0)
    # print("xNan",xNan)
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


get_audio_ids_recommend_by_user_id(user_id="95291")

# Create a subset of top fifty thousand observations to work with, as the entire dataset is TOO expensive to compute on!!!
# print(rawData.head())
# print("\n", rawData.tail())
# print("\n", rawData.describe(include="all"))
# print(data["count"].hist(bins=80))
# print("\n", data.sort_values(by=["count"], ascending=False).head(10))

# popModel.create(trainData, 'user_id', 'artist_name') for popularity based recommendations by artists
# print("\n", popModel.recommend(users[342]))

# Create pivot table (interaction matrix) from the original dataset
# print(interaction)

# """Evaluation of the trained model"""

# print(
#     "\nPrecision at K:",
#     precision_at_k(hybridModel, interaction, k=15).mean().round(3) * 100,
# )
# print(
#     "Recall at K:", recall_at_k(hybridModel, interaction, k=500).mean().round(3) * 100
# )
# print(
#     "Area under ROC curve:", auc_score(hybridModel, interaction).mean().round(3) * 100
# )
# print(
#     "Reciprocal Rank:", reciprocal_rank(hybridModel, interaction).mean().round(3) * 100
# )
# Recommend songs similar to a given songID
# songItemDist = create_item_emdedding_distance_matrix(model=hybridModel, interactions=x)
# print(
#     item_item_recommendation(
#         item_emdedding_distance_matrix=songItemDist,
#         item_id="4962",
#         item_dict=songDict,
#         n_items=10,
#     ),
# )
