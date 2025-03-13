"""
Performs user segmentation using PCA for dimensionality reduction and KMeans clustering.
"""

import numpy as np
import pandas as pd
import logging
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class UserProfiler:
    def __init__(self, n_clusters=5, n_components=2):
        """
        Initialize the UserProfiler.
        :param n_clusters: Number of clusters to segment users.
        :param n_components: Number of PCA components for visualization and reduction.
        """
        self.n_clusters = n_clusters
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=self.n_components)
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        self.logger = logging.getLogger(__name__)
        self.fitted = False

    def fit(self, user_data):
        """
        Fit the scaler, PCA, and KMeans clustering on user data.
        :param user_data: DataFrame of user behavior features.
        :return: Cluster labels.
        """
        self.logger.info("Starting user profiling on data with shape {}.".format(user_data.shape))
        # Standardize data
        scaled_data = self.scaler.fit_transform(user_data.select_dtypes(include=[np.number]))
        # Dimensionality reduction
        pca_result = self.pca.fit_transform(scaled_data)
        # Clustering
        clusters = self.kmeans.fit_predict(pca_result)
        self.fitted = True
        self.logger.info("User profiling complete. Number of clusters: {}.".format(self.n_clusters))
        return clusters

    def predict(self, user_data):
        """
        Predict cluster labels for new user data.
        :param user_data: DataFrame of new user behavior features.
        :return: Predicted cluster labels.
        """
        if not self.fitted:
            self.logger.error("The model has not been fitted yet. Call fit() first.")
            raise ValueError("UserProfiler not fitted.")
        scaled_data = self.scaler.transform(user_data.select_dtypes(include=[np.number]))
        pca_result = self.pca.transform(scaled_data)
        clusters = self.kmeans.predict(pca_result)
        self.logger.info("Predicted clusters for {} users.".format(len(clusters)))
        return clusters

    def get_cluster_centers(self):
        """
        Retrieve the centers of clusters in the PCA-transformed space.
        :return: Array of cluster centers.
        """
        if not self.fitted:
            self.logger.error("The model has not been fitted yet.")
            raise ValueError("UserProfiler not fitted.")
        centers = self.kmeans.cluster_centers_
        return centers

if __name__ == '__main__':
    # Quick test for user profiling
    import matplotlib.pyplot as plt
    logging.basicConfig(level=logging.INFO)
    # Generate synthetic user behavior data
    df = pd.DataFrame({
        "page_views": np.random.poisson(lam=10, size=300),
        "clicks": np.random.poisson(lam=3, size=300),
        "time_spent": np.random.exponential(scale=5, size=300),
        "purchases": np.random.binomial(n=5, p=0.2, size=300)
    })
    profiler = UserProfiler(n_clusters=4, n_components=2)
    clusters = profiler.fit(df)
    df["cluster"] = clusters
    plt.figure()
    plt.scatter(df["page_views"], df["clicks"], c=df["cluster"], cmap="viridis", alpha=0.7)
    plt.xlabel("Page Views")
    plt.ylabel("Clicks")
    plt.title("User Profiling: Clusters Visualization")
    plt.show()