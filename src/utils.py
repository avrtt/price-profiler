"""
Helper functions for logging, data I/O, plotting and evaluation metrics.
"""

import os
import logging
import pandas as pd
import matplotlib.pyplot as plt

def setup_logging(log_file="logs/app.log", level=logging.INFO):
    """
    Set up logging configuration.
    """
    log_dir = os.path.dirname(log_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logging.basicConfig(filename=log_file,
                        level=level,
                        format='%(asctime)s %(levelname)s %(name)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    console = logging.StreamHandler()
    console.setLevel(level)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    return logging.getLogger()

def save_dataframe(df, filename):
    """
    Save a Pandas DataFrame to CSV.
    """
    df.to_csv(filename, index=False)
    logging.getLogger(__name__).info("DataFrame saved to {}.".format(filename))

def load_dataframe(filename):
    """
    Load a CSV file into a Pandas DataFrame.
    """
    df = pd.read_csv(filename)
    logging.getLogger(__name__).info("DataFrame loaded from {}.".format(filename))
    return df

def plot_series(series, title="Time Series", xlabel="Time", ylabel="Value", save_path=None):
    """
    Plot a time series.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(series.index, series.values, marker='o', linestyle='-')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if save_path:
        plt.savefig(save_path)
        logging.getLogger(__name__).info("Plot saved to {}.".format(save_path))
    else:
        plt.show()

def calculate_rmse(true, predicted):
    """
    Calculate Root Mean Squared Error.
    """
    import numpy as np
    rmse = np.sqrt(((true - predicted) ** 2).mean())
    return rmse

if __name__ == '__main__':
    # Quick test for utils functions
    logger = setup_logging()
    import numpy as np
    import pandas as pd
    # Create a simple DataFrame
    df = pd.DataFrame({"a": np.arange(10), "b": np.arange(10, 20)})
    save_dataframe(df, "test.csv")
    df_loaded = load_dataframe("test.csv")
    logger.info("Loaded DataFrame:\n{}".format(df_loaded.head()))
    # Plot a dummy time series
    series = pd.Series(np.sin(np.linspace(0, 3.14, 50)), index=pd.date_range("2021-01-01", periods=50))
    plot_series(series, title="Sine Wave")