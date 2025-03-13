"""
Implements a recommendation engine using collaborative filtering with cosine similarity.
"""

import numpy as np
import pandas as pd
import logging
from sklearn.metrics.pairwise import cosine_similarity

class RecommendationEngine:
    def __init__(self):
        """
        Initialize the RecommendationEngine.
        """
        self.logger = logging.getLogger(__name__)
        self.user_item_matrix = None

    def load_user_item_matrix(self, matrix):
        """
        Load the user-item interaction matrix.
        :param matrix: DataFrame or numpy array representing user-item interactions.
        """
        if isinstance(matrix, pd.DataFrame):
            self.user_item_matrix = matrix.drop(columns=["user_id"]).values
        else:
            self.user_item_matrix = matrix
        self.logger.info("User-item matrix loaded with shape {}.".format(self.user_item_matrix.shape))

    def compute_similarity(self):
        """
        Compute cosine similarity between users based on their interactions.
        :return: Similarity matrix.
        """
        if self.user_item_matrix is None:
            self.logger.error("User-item matrix not loaded.")
            raise ValueError("Load user-item matrix first.")
        similarity = cosine_similarity(self.user_item_matrix)
        self.logger.info("Computed user-user cosine similarity matrix.")
        return similarity

    def recommend(self, user_index, top_n=5):
        """
        Recommend products to a given user based on similar users.
        :param user_index: Index of the user in the matrix.
        :param top_n: Number of products to recommend.
        :return: List of recommended product indices.
        """
        similarity = self.compute_similarity()
        # Get the similarity scores for the target user
        user_sim = similarity[user_index]
        # Get weighted sum of ratings from similar users
        weighted_sum = np.dot(self.user_item_matrix.T, user_sim)
        # Normalize by sum of similarities
        sim_sum = np.sum(user_sim) + 1e-9
        score = weighted_sum / sim_sum
        # Exclude products the user has already interacted with (non-zero values)
        user_interactions = self.user_item_matrix[user_index]
        score[user_interactions > 0] = -np.inf
        recommended_indices = np.argsort(score)[::-1][:top_n]
        self.logger.info("Generated recommendations for user index {}.".format(user_index))
        return recommended_indices.tolist()

if __name__ == '__main__':
    # Quick test for recommendation engine
    import matplotlib.pyplot as plt
    logging.basicConfig(level=logging.INFO)
    # Generate a random user-item matrix with 10 users and 8 products
    np.random.seed(42)
    matrix = np.random.randint(0, 6, (10, 8))
    df_matrix = pd.DataFrame(matrix, columns=["product_" + str(i+1) for i in range(8)])
    df_matrix.insert(0, "user_id", np.arange(1, 11))
    rec_engine = RecommendationEngine()
    rec_engine.load_user_item_matrix(df_matrix)
    recommendations = rec_engine.recommend(user_index=0, top_n=3)
    print("Recommendations for user 0:", recommendations)