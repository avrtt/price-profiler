"""
Contains global configuration parameters for data generation, forecasting, pricing optimization and user profiling.
"""

import os

class Config:
    def __init__(self):
        self.random_seed = 42

        # data generation configurations
        self.data_generation = {
            "start_date": "2024-01-01",
            "end_date": "2025-12-31",
            "frequency": "D", # daily data
            "sales_trend": 0.05, # underlying daily trend factor
            "seasonality_amplitude": 20,
            "noise_level": 5,
            "n_users": 500,
            "n_products": 50,
            "user_behavior_features": ["page_views", "clicks", "time_spent", "purchases"],
            "user_item_interaction_density": 0.1,
            "output_dir": os.path.join(os.getcwd(), "data")
        }

        # forecasting configurations
        self.forecasting = {
            "order": (1, 1, 1), # ARIMA order
            "grid_search": {
                "p_values": [0, 1, 2],
                "d_values": [0, 1],
                "q_values": [0, 1, 2]
            }
        }

        # pricing optimization configurations
        self.pricing = {
            "model_params": {
                "rf": {
                    "n_estimators": 100,
                    "max_depth": 10
                }
            }
        }

        # user profiling configurations
        self.user_profiling = {
            "n_clusters": 5,
            "pca_components": 2
        }

        # recommendation configurations
        self.recommendation = {
            "top_n": 5
        }

        # logging and output directories
        self.logging = {
            "log_file": os.path.join(os.getcwd(), "logs", "app.log")
        }

        # ensure output directories exist
        self._ensure_directories()

    def _ensure_directories(self):
        output_dir = self.data_generation.get("output_dir", os.getcwd())
        log_dir = os.path.dirname(self.logging["log_file"])
        for d in [output_dir, log_dir]:
            if not os.path.exists(d):
                os.makedirs(d)

if __name__ == '__main__':
    # for quick config testing
    config = Config()
    print("Configuration loaded successfully.")