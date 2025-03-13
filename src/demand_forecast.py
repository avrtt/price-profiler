"""
Implements demand forecasting using ARIMA models with grid search for hyperparameter tuning.
"""

import numpy as np
import pandas as pd
import itertools
import logging
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

class DemandForecaster:
    def __init__(self, order=(1, 1, 1)):
        """
        Initialize the DemandForecaster with a default ARIMA order.
        """
        self.order = order
        self.model = None
        self.logger = logging.getLogger(__name__)

    def fit(self, sales_series):
        """
        Fit an ARIMA model to the sales_series.
        :param sales_series: Pandas Series of sales data.
        :return: Fitted ARIMA model.
        """
        self.logger.info("Fitting ARIMA model with order {}.".format(self.order))
        try:
            self.model = ARIMA(sales_series, order=self.order).fit()
        except Exception as e:
            self.logger.error("Error fitting ARIMA model: {}".format(e))
            raise e
        return self.model

    def predict(self, steps=30):
        """
        Forecast future demand for a given number of steps.
        :param steps: Number of time steps to forecast.
        :return: Forecasted values as a Pandas Series.
        """
        if self.model is None:
            self.logger.error("Model not fitted. Call fit() before predict().")
            raise ValueError("Model not fitted.")
        self.logger.info("Forecasting {} steps ahead.".format(steps))
        forecast = self.model.forecast(steps=steps)
        return forecast

    def evaluate(self, sales_series, test_size=0.2):
        """
        Evaluate the forecasting model by splitting the data into training and test sets.
        :param sales_series: Full Pandas Series of sales data.
        :param test_size: Fraction of data to use for testing.
        :return: RMSE of the forecast on the test set.
        """
        n = len(sales_series)
        split = int(n * (1 - test_size))
        train, test = sales_series.iloc[:split], sales_series.iloc[split:]
        self.logger.info("Training on {} records, testing on {} records.".format(len(train), len(test)))
        try:
            model = ARIMA(train, order=self.order).fit()
            forecast = model.forecast(steps=len(test))
            rmse = np.sqrt(mean_squared_error(test, forecast))
            self.logger.info("RMSE on test set: {:.3f}".format(rmse))
            return rmse
        except Exception as e:
            self.logger.error("Error during evaluation: {}".format(e))
            raise e

    def grid_search_arima(self, sales_series, p_values, d_values, q_values):
        """
        Perform a grid search to find the best ARIMA order based on RMSE.
        :param sales_series: Pandas Series of sales data.
        :param p_values: List of candidate p values.
        :param d_values: List of candidate d values.
        :param q_values: List of candidate q values.
        :return: Best (p, d, q) tuple and corresponding RMSE.
        """
        best_score, best_cfg = float("inf"), None
        self.logger.info("Starting grid search for ARIMA parameters.")
        for p, d, q in itertools.product(p_values, d_values, q_values):
            order = (p, d, q)
            try:
                rmse = self.evaluate(sales_series, test_size=0.2)
                self.logger.info("Tested ARIMA order {} with RMSE {:.3f}".format(order, rmse))
                if rmse < best_score:
                    best_score, best_cfg = rmse, order
            except Exception as e:
                self.logger.warning("Skipping order {} due to error: {}".format(order, e))
                continue
        self.logger.info("Best ARIMA order: {} with RMSE {:.3f}".format(best_cfg, best_score))
        return best_cfg, best_score

if __name__ == '__main__':
    # Quick test using synthetic data
    import matplotlib.pyplot as plt
    logging.basicConfig(level=logging.INFO)
    dates = pd.date_range(start="2024-01-01", periods=365, freq='D')
    sales = pd.Series(50 + 0.05*np.arange(365) + 10*np.sin(np.linspace(0, 3.14*2, 365)) + np.random.normal(0, 5, 365), index=dates)
    forecaster = DemandForecaster(order=(1,1,1))
    forecaster.fit(sales)
    forecast = forecaster.predict(steps=30)
    print("Forecasted values:")
    print(forecast.head())
    plt.figure()
    plt.plot(sales.index, sales, label="Historical Sales")
    plt.plot(pd.date_range(sales.index[-1], periods=31, freq='D')[1:], forecast, label="Forecast")
    plt.legend()
    plt.title("Sales Forecast")
    plt.show()