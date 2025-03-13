"""
Generates synthetic datasets for sales, competitor pricing, and user behavior.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

class DataGenerator:
    def __init__(self, config):
        """
        Initialize DataGenerator with configuration.
        """
        self.config = config
        np.random.seed(self.config.get("random_seed", 42))
        self.output_dir = self.config.get("output_dir", os.getcwd())
        self.logger = logging.getLogger(__name__)
    
    def generate_date_range(self):
        """
        Generate a date range based on start and end dates.
        """
        start_date = pd.to_datetime(self.config["start_date"])
        end_date = pd.to_datetime(self.config["end_date"])
        frequency = self.config.get("frequency", "D")
        dates = pd.date_range(start=start_date, end=end_date, freq=frequency)
        return dates

    def generate_sales_data(self):
        """
        Generate synthetic sales data with trend, seasonality, and noise.
        Returns a DataFrame with columns: date, sales, feature1, feature2, price.
        """
        dates = self.generate_date_range()
        n = len(dates)
        trend = np.linspace(1, 1 + self.config.get("sales_trend", 0.05), n)
        seasonality = self.config.get("seasonality_amplitude", 20) * np.sin(np.linspace(0, 3.14 * 2, n))
        noise = np.random.normal(0, self.config.get("noise_level", 5), n)
        sales = trend * (50 + seasonality) + noise
        # Create additional synthetic features for pricing model
        feature1 = np.random.uniform(0, 100, n)
        feature2 = np.random.uniform(0, 50, n)
        # Simulate price data (base price with random fluctuations)
        base_price = 20
        price = base_price + 0.1 * sales + np.random.normal(0, 2, n)
        data = pd.DataFrame({
            "date": dates,
            "sales": sales,
            "feature1": feature1,
            "feature2": feature2,
            "price": price
        })
        self.logger.info("Sales data generated with {} records.".format(len(data)))
        return data

    def generate_competitor_pricing(self):
        """
        Generate synthetic competitor pricing data.
        Returns a DataFrame with columns: date, competitor_price.
        """
        dates = self.generate_date_range()
        n = len(dates)
        # Competitor pricing fluctuates around a base with added noise and seasonality
        base_price = 22
        seasonality = 3 * np.sin(np.linspace(0, 2 * np.pi, n))
        noise = np.random.normal(0, 1.5, n)
        competitor_price = base_price + seasonality + noise
        data = pd.DataFrame({
            "date": dates,
            "competitor_price": competitor_price
        })
        self.logger.info("Competitor pricing data generated with {} records.".format(len(data)))
        return data

    def generate_user_behavior(self):
        """
        Generate synthetic user behavior data.
        Returns a DataFrame with user features such as page_views, clicks, time_spent, and purchases.
        """
        n_users = self.config.get("n_users", 500)
        features = self.config.get("user_behavior_features", ["page_views", "clicks", "time_spent", "purchases"])
        data = {}
        data["user_id"] = np.arange(1, n_users + 1)
        # Generate behavior metrics with different distributions
        data["page_views"] = np.random.poisson(lam=10, size=n_users)
        data["clicks"] = np.random.poisson(lam=3, size=n_users)
        data["time_spent"] = np.random.exponential(scale=5, size=n_users)
        data["purchases"] = np.random.binomial(n=5, p=0.2, size=n_users)
        df = pd.DataFrame(data)
        self.logger.info("User behavior data generated for {} users.".format(n_users))
        return df

    def generate_user_item_matrix(self):
        """
        Generate a synthetic user-item interaction matrix.
        Returns a DataFrame where rows represent users and columns represent products.
        """
        n_users = self.config.get("n_users", 500)
        n_products = self.config.get("n_products", 50)
        density = self.config.get("user_item_interaction_density", 0.1)
        # Initialize matrix with zeros
        matrix = np.zeros((n_users, n_products))
        # Fill with random interactions based on density
        for i in range(n_users):
            for j in range(n_products):
                if np.random.rand() < density:
                    # Simulate interaction score (e.g., purchase count or rating)
                    matrix[i, j] = np.random.randint(1, 6)
        df = pd.DataFrame(matrix, columns=["product_" + str(i+1) for i in range(n_products)])
        df.insert(0, "user_id", np.arange(1, n_users + 1))
        self.logger.info("User-item interaction matrix generated with shape {}.".format(df.shape))
        return df

if __name__ == '__main__':
    # Quick testing if module is run directly
    import logging
    logging.basicConfig(level=logging.INFO)
    sample_config = {
        "random_seed": 42,
        "start_date": "2024-01-01",
        "end_date": "2024-12-31",
        "frequency": "D",
        "sales_trend": 0.05,
        "seasonality_amplitude": 20,
        "noise_level": 5,
        "n_users": 500,
        "n_products": 50,
        "user_behavior_features": ["page_views", "clicks", "time_spent", "purchases"],
        "user_item_interaction_density": 0.1,
        "output_dir": "./data"
    }
    dg = DataGenerator(sample_config)
    sales = dg.generate_sales_data()
    comp = dg.generate_competitor_pricing()
    user_behavior = dg.generate_user_behavior()
    ui_matrix = dg.generate_user_item_matrix()
    print("Data generation test complete.")