This is a tool for demand forecasting, regression and ensemble‐based dynamic pricing optimization; ML-driven user profiling with personalized recommendation systems for real‐time sales and engagement optimization.

The project was developed as a part of my real freelance work and published with the client's approval. For privacy purposes, all client data has been replaced with synthetic examples.  

The mini-app is designed as PoC to help e-commerce platforms optimize product pricing and personalize user experiences.  

It can predict predict future sales based on historical data, competitor pricing and seasonality using time series models, dynamically compute real-time optimal pricing. Profiling is done by clustering users based on behavior tracking (clicks, time on page, purchase history). Users are segmented for personalized recommendations using collaborative filtering techniques.

## Main files
- **config.py** - configuration parameters for all modules  
- **data_generation.py** - contains the `DataGenerator` class; synthetic data generation for sales, competitor pricing and user behavior  
- **demand_forecast.py** - `DemandForecaster` class for time series demand forecasting using ARIMA models and grid search
- **pricing_optimizer.py** - `PricingOptimizer` class for dynamic pricing optimizer using regression and random forest models
- **user_profiling.py** - user segmentation using PCA and k-means clustering
- **recommendation.py** - collaborative filtering based recommendation system; implements the `RecommendationEngine` class that leverages cosine similarity
- **utils.py** - helper functions for logging, plotting, data I/O, etc.
- **main.py** - main script to run the complete pipeline

## Setup
1. Clone:
   ```bash
   git clone git@github.com:avrtt/price-profiler.git
   cd price-profiler
   ```
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Run the main script:
   ```bash
   python src/main.py
   ```

## License
MIT
