"""
Main script to orchestrate data generation, forecasting, pricing optimization, user profiling and recommendations.
"""

import os
import argparse
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from config import Config
from data_generation import DataGenerator
from demand_forecast import DemandForecaster
from pricing_optimizer import PricingOptimizer
from user_profiling import UserProfiler
from recommendation import RecommendationEngine
from utils import setup_logging, save_dataframe, plot_series

def run_data_generation(config):
    """
    Generate and save synthetic data.
    """
    dg = DataGenerator(config.data_generation)
    sales_data = dg.generate_sales_data()
    competitor_data = dg.generate_competitor_pricing()
    user_behavior = dg.generate_user_behavior()
    user_item_matrix = dg.generate_user_item_matrix()

    # Save generated data to CSV files
    save_dataframe(sales_data, os.path.join(config.data_generation["output_dir"], "sales_data.csv"))
    save_dataframe(competitor_data, os.path.join(config.data_generation["output_dir"], "competitor_data.csv"))
    save_dataframe(user_behavior, os.path.join(config.data_generation["output_dir"], "user_behavior.csv"))
    save_dataframe(user_item_matrix, os.path.join(config.data_generation["output_dir"], "user_item_matrix.csv"))

    return sales_data, competitor_data, user_behavior, user_item_matrix

def run_forecasting(sales_data, forecasting_config):
    """
    Run demand forecasting on sales data.
    """
    forecaster = DemandForecaster(order=forecasting_config["order"])
    # Fit model on sales data (assuming sales_data['sales'] is our target)
    model = forecaster.fit(sales_data["sales"])
    # Forecast next 30 days
    forecast = forecaster.predict(steps=30)
    logging.getLogger(__name__).info("Forecast generated for next 30 days.")
    return forecast

def run_pricing_optimization(sales_data, pricing_config):
    """
    Train pricing optimizer and predict optimal prices.
    """
    optimizer = PricingOptimizer(rf_params=pricing_config.get("model_params", {}).get("rf"))
    # For simplicity, using feature1 and feature2 from sales_data
    features = sales_data[["feature1", "feature2"]]
    prices = sales_data["price"]
    optimizer.train(features, prices)
    optimal_prices = optimizer.predict_optimal_price(features, method="ensemble")
    eval_metrics = optimizer.evaluate(features, prices)
    logging.getLogger(__name__).info("Pricing optimization complete.")
    return optimal_prices, eval_metrics

def run_user_profiling(user_behavior, profiling_config):
    """
    Perform user profiling on user behavior data.
    """
    profiler = UserProfiler(n_clusters=profiling_config.get("n_clusters", 5),
                            n_components=profiling_config.get("pca_components", 2))
    clusters = profiler.fit(user_behavior)
    user_behavior["cluster"] = clusters
    logging.getLogger(__name__).info("User profiling complete.")
    return user_behavior, clusters

def run_recommendations(user_item_matrix, recommendation_config):
    """
    Generate product recommendations for a sample user.
    """
    rec_engine = RecommendationEngine()
    rec_engine.load_user_item_matrix(user_item_matrix)
    recommendations = rec_engine.recommend(user_index=0, top_n=recommendation_config.get("top_n", 5))
    logging.getLogger(__name__).info("Recommendations generated for user index 0.")
    return recommendations

def visualize_results(sales_data, forecast, optimal_prices):
    """
    Visualize the sales data and forecast.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(sales_data["date"], sales_data["sales"], label="Historical Sales", marker="o")
    # Extend dates for forecast
    last_date = pd.to_datetime(sales_data["date"].iloc[-1])
    forecast_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=len(forecast), freq='D')
    plt.plot(forecast_dates, forecast, label="Forecast", marker="x")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.title("Sales Forecast")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(sales_data["date"], sales_data["price"], label="Actual Price", marker="o")
    plt.plot(sales_data["date"], optimal_prices, label="Optimal Price", marker="x")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title("Pricing Optimization")
    plt.legend()
    plt.tight_layout()
    plt.show()

def main(args):
    # Set up logging
    logger = setup_logging()
    logger.info("Starting price-profiler pipeline.")

    # Load configuration
    config = Config()
    # Data Generation
    sales_data, competitor_data, user_behavior, user_item_matrix = run_data_generation(config)
    # Forecasting
    forecast = run_forecasting(sales_data, config.forecasting)
    # Pricing Optimization
    optimal_prices, pricing_eval = run_pricing_optimization(sales_data, config.pricing)
    # User Profiling
    user_behavior, clusters = run_user_profiling(user_behavior, config.user_profiling)
    # Recommendations
    recommendations = run_recommendations(user_item_matrix, config.recommendation)

    # Log outputs
    logger.info("Forecast (first 5): {}".format(forecast.head().to_dict()))
    logger.info("Pricing Evaluation Metrics: {}".format(pricing_eval))
    logger.info("User Clusters Distribution: \n{}".format(user_behavior["cluster"].value_counts().to_dict()))
    logger.info("Recommendations for user 0: {}".format(recommendations))

    # Visualize results
    visualize_results(sales_data, forecast, optimal_prices)
    logger.info("price-profiler pipeline complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the price-profiler dynamic pricing & profiling engine.")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode with verbose logging.")
    args = parser.parse_args()
    main(args)