from popular_recomendation import create_user_dict, create_item_dict, sample_recommendation_user, item_item_recommendation, create_item_emdedding_distance_matrix
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
    prefix = "model_"
    file_pattern = prefix + "*.txt"
    files = glob.glob(file_pattern)
    
    if not files:
        print("No files found with the specified prefix:", prefix)
        exit()
    
    files.sort(key=os.path.getmtime, reverse=True)
    latest_file = files[0]
    return pd.read_csv(latest_file)

# Create a subset of top fifty thousand observations to work with, as the entire dataset is TOO expensive to compute on!!!
triplets = "./train_model/file_20230703173728.txt"
rawData1 = pd.read_csv(triplets, delimiter="\t")
rawData = rawData1.head(50000)

# print(rawData.head())
# print("\n", rawData.tail())
# print("\n", rawData.describe(include="all"))

data = rawData.groupby(["audio_name"]).agg({"count": "count"}).reset_index()
data["percentage"] = rawData["count"].div(rawData['count'].sum()) * 100
# print("\n", data.sort_values(by=["count"], ascending=False).head(10))

# print(data["count"].hist(bins=80))

users = rawData["user_id"].unique()
popModel = popularity_recommender()
popModel.create(rawData, "user_id", "audio_name")
# popModel.create(trainData, 'user_id', 'artist_name') for popularity based recommendations by artists
# print("\n", popModel.recommend(users[342]))

# Create pivot table (interaction matrix) from the original dataset
x = rawData.pivot_table(index="user_id", columns="audio_id", values="count")
xNan = x.fillna(0)
# print("xNan",xNan)
interaction = sp.csr_matrix(xNan.values)
# print(interaction)

"""Personlized hybrid model"""

hybridModel = LightFM(loss="warp-kos", n=20, k=20, learning_schedule="adadelta")
hybridModel.fit(interaction, epochs=30, num_threads=6)

"""Evaluation of the trained model"""

print(
    "\nPrecision at K:",
    precision_at_k(hybridModel, interaction, k=15).mean().round(3) * 100,
)
print(
    "Recall at K:", recall_at_k(hybridModel, interaction, k=500).mean().round(3) * 100
)
print(
    "Area under ROC curve:", auc_score(hybridModel, interaction).mean().round(3) * 100
)
print(
    "Reciprocal Rank:", reciprocal_rank(hybridModel, interaction).mean().round(3) * 100
)

"""Recommendaing songs personally based on the user"""

# Creating user dictionary based on their index and number in the interaction matrix using recsys library
userDict = create_user_dict(interactions=x)
print('\n', userDict)

# Creating a song dictionary based on their songID and artist name
songDict = create_item_dict(df=rawData, id_col="audio_id", name_col="audio_id")
print('\n', songDict)

# Recommend songs using lightfm library
print(
    sample_recommendation_user(
        model=hybridModel,
        interactions=x,
        user_id=95291,
        user_dict=userDict,
        item_dict=songDict,
        threshold=5,
        nrec_items=50,
        show=True,
    ),
)

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
