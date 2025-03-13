"""
Provides a dynamic pricing optimizer that uses multiple machine learning models to determine optimal pricing.
"""

import numpy as np
import pandas as pd
import logging
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from joblib import dump, load

class PricingOptimizer:
    def __init__(self, rf_params=None):
        """
        Initialize PricingOptimizer with specified parameters.
        :param rf_params: Dictionary of parameters for the Random Forest model.
        """
        self.logger = logging.getLogger(__name__)
        self.model_lr = LinearRegression()
        self.rf_params = rf_params if rf_params is not None else {"n_estimators": 100, "max_depth": 10, "random_state": 42}
        self.model_rf = RandomForestRegressor(**self.rf_params)
        self.trained = False

    def train(self, features, prices):
        """
        Train both Linear Regression and Random Forest models.
        :param features: DataFrame or array of input features.
        :param prices: Array or Series of target prices.
        """
        self.logger.info("Training Linear Regression model.")
        self.model_lr.fit(features, prices)
        self.logger.info("Training Random Forest model with parameters: {}.".format(self.rf_params))
        self.model_rf.fit(features, prices)
        self.trained = True

    def evaluate(self, features, prices):
        """
        Evaluate both models and return their RMSE.
        :param features: Features for evaluation.
        :param prices: True prices.
        :return: Dictionary with RMSE for linear regression and random forest.
        """
        if not self.trained:
            self.logger.error("Models have not been trained yet.")
            raise ValueError("Call train() before evaluate().")
        pred_lr = self.model_lr.predict(features)
        pred_rf = self.model_rf.predict(features)
        rmse_lr = np.sqrt(mean_squared_error(prices, pred_lr))
        rmse_rf = np.sqrt(mean_squared_error(prices, pred_rf))
        self.logger.info("Linear Regression RMSE: {:.3f}, Random Forest RMSE: {:.3f}".format(rmse_lr, rmse_rf))
        return {"LinearRegression": rmse_lr, "RandomForest": rmse_rf}

    def predict_optimal_price(self, features, method="ensemble"):
        """
        Predict optimal prices using the specified method.
        :param features: Features to predict prices.
        :param method: 'lr', 'rf', or 'ensemble' (default).
        :return: Array of predicted prices.
        """
        if not self.trained:
            self.logger.error("Models have not been trained yet.")
            raise ValueError("Call train() before predict_optimal_price().")
        pred_lr = self.model_lr.predict(features)
        pred_rf = self.model_rf.predict(features)
        if method == "lr":
            self.logger.info("Using Linear Regression for prediction.")
            return pred_lr
        elif method == "rf":
            self.logger.info("Using Random Forest for prediction.")
            return pred_rf
        elif method == "ensemble":
            self.logger.info("Using ensemble of LR and RF for prediction.")
            return (pred_lr + pred_rf) / 2.0
        else:
            self.logger.error("Invalid prediction method specified: {}.".format(method))
            raise ValueError("Method must be 'lr', 'rf', or 'ensemble'.")

    def save_models(self, path):
        """
        Save both models to disk.
        :param path: Directory path where models will be saved.
        """
        dump(self.model_lr, path + "_lr.joblib")
        dump(self.model_rf, path + "_rf.joblib")
        self.logger.info("Models saved to disk at {}_lr.joblib and {}_rf.joblib.".format(path, path))

    def load_models(self, path):
        """
        Load models from disk.
        :param path: Directory path where models are saved.
        """
        self.model_lr = load(path + "_lr.joblib")
        self.model_rf = load(path + "_rf.joblib")
        self.trained = True
        self.logger.info("Models loaded from disk.")

if __name__ == '__main__':
    # Quick test with synthetic data
    import matplotlib.pyplot as plt
    import pandas as pd
    logging.basicConfig(level=logging.INFO)
    # Create a synthetic dataset
    np.random.seed(42)
    n_samples = 200
    features = pd.DataFrame({
        "feature1": np.random.uniform(0, 100, n_samples),
        "feature2": np.random.uniform(0, 50, n_samples)
    })
    # Simulate price as a function of features with some noise
    prices = 10 + 0.2 * features["feature1"] + 0.5 * features["feature2"] + np.random.normal(0, 3, n_samples)
    optimizer = PricingOptimizer()
    optimizer.train(features, prices)
    preds = optimizer.predict_optimal_price(features, method="ensemble")
    evaluation = optimizer.evaluate(features, prices)
    print("Evaluation metrics:", evaluation)
    plt.figure()
    plt.scatter(range(n_samples), prices, label="Actual Prices", alpha=0.7)
    plt.scatter(range(n_samples), preds, label="Predicted Prices", alpha=0.7)
    plt.legend()
    plt.title("Pricing Optimization: Actual vs Predicted")
    plt.show()