class popularity_genre_recommender:
    def __init__(self):
        self.train_data = None
        self.genre_id = None
        self.item_id = None
        self.popularity_recommendations = None

    # Create the popularity based recommender system model
    def create(self, train_data, genre_id, item_id):
        self.train_data = train_data
        self.genre_id = genre_id
        self.item_id = item_id

        # Get a count of genre_ids for each unique song as recommendation score
        train_data_grouped = (
            train_data.groupby([self.item_id])
            .agg({self.genre_id: "count"})
            .reset_index()
        )
        train_data_grouped.rename(columns={"genre_id": "score"}, inplace=True)

        # Sort the songs based upon recommendation score
        train_data_sort = train_data_grouped.sort_values(
            ["score", self.item_id], ascending=[0, 1]
        )

        # Generate a recommendation rank based upon score
        train_data_sort["Rank"] = train_data_sort["score"].rank(
            ascending=0, method="first"
        )

        # Get the top 10 recommendations
        self.popularity_recommendations = train_data_sort.head(10)

    # Use the popularity based recommender system model to
    # make recommendations
    def recommend(self, genre_id):
        user_recommendations = self.popularity_recommendations

        # Add genre_id column for which the recommendations are being generated
        user_recommendations["genre_id"] = genre_id

        # Bring genre_id column to the front
        cols = user_recommendations.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        user_recommendations = user_recommendations[cols]

        return user_recommendations
