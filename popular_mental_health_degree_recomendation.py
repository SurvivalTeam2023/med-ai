class popularity_mental_health_degree_recommender:
    def __init__(self):
        self.train_data = None
        self.mental_health_degree_id = None
        self.item_id = None
        self.popularity_recommendations = None

    # Create the popularity based recommender system model
    def create(self, train_data, mental_health_degree_id, item_id):
        self.train_data = train_data
        self.mental_health_degree_id = mental_health_degree_id
        self.item_id = item_id

        # Get a count of mental_health_degree_ids for each unique song as recommendation score
        train_data_grouped = (
            train_data.groupby([self.item_id])
            .agg({self.mental_health_degree_id: "count"})
            .reset_index()
        )
        train_data_grouped.rename(
            columns={"mental_health_degree_id": "score"}, inplace=True
        )

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
    def recommend(self, mental_health_degree_id):
        mental_recommendations = self.popularity_recommendations

        # Add mental_health_degree_id column for which the recommendations are being generated
        mental_recommendations["mental_health_degree_id"] = mental_health_degree_id

        # Bring mental_health_degree_id column to the front
        cols = mental_recommendations.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        mental_recommendations = mental_recommendations[cols]

        return mental_recommendations
