import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple
import logging

logger = logging.getLogger(__name__)

class DemandForecastingModule:
    def __init__(self):
        self.models = {}
        self.model_dir = 'models'
        self.scaler = StandardScaler()
        os.makedirs(self.model_dir, exist_ok=True)

    def load_and_preprocess_data(self, sales_data_path: str, weather_data_path: str = None) -> pd.DataFrame:
        """Load and preprocess sales data with external factors"""
        df = pd.read_csv(sales_data_path)
        df['date'] = pd.to_datetime(df['date'])

        # Feature engineering
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['week_of_year'] = df['date'].dt.isocalendar().week
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
        df['is_month_start'] = df['date'].dt.is_month_start.astype(int)

        # Lag features
        for lag in [1, 7, 14, 30]:
            df[f'lag_{lag}'] = df.groupby('product')['quantity'].shift(lag)

        # Rolling statistics
        for window in [7, 14, 30]:
            df[f'rolling_mean_{window}'] = df.groupby('product')['quantity'].rolling(window=window, min_periods=1).mean().reset_index(0, drop=True)
            df[f'rolling_std_{window}'] = df.groupby('product')['quantity'].rolling(window=window, min_periods=1).std().reset_index(0, drop=True)

        # Add weather data if available
        if weather_data_path and os.path.exists(weather_data_path):
            weather_df = pd.read_csv(weather_data_path)
            weather_df['date'] = pd.to_datetime(weather_df['date'])
            df = df.merge(weather_df, on=['date', 'location'], how='left')

            # Fill missing weather data
            weather_columns = ['temperature', 'humidity', 'precipitation', 'is_holiday']
            for col in weather_columns:
                if col in df.columns:
                    df[col] = df[col].fillna(df[col].mean())

        # Fill NaN values
        df = df.fillna(method='bfill').fillna(method='ffill')

        return df

    def train_models(self, data_path: str, weather_data_path: str = None) -> Dict[str, Any]:
        """Train demand forecasting models for each product"""
        df = self.load_and_preprocess_data(data_path, weather_data_path)

        # Get unique products
        products = df['product'].unique()

        training_results = {}

        for product in products:
            logger.info(f"Training model for product: {product}")
            product_df = df[df['product'] == product].copy()

            # Prepare features and target
            feature_columns = [
                'day_of_week', 'month', 'quarter', 'week_of_year', 'is_weekend',
                'is_month_end', 'is_month_start', 'lag_1', 'lag_7', 'lag_14', 'lag_30',
                'rolling_mean_7', 'rolling_std_7', 'rolling_mean_14', 'rolling_std_14',
                'rolling_mean_30', 'rolling_std_30'
            ]

            # Add weather features if available
            weather_features = ['temperature', 'humidity', 'precipitation', 'is_holiday']
            for feature in weather_features:
                if feature in product_df.columns:
                    feature_columns.append(feature)

            X = product_df[feature_columns]
            y = product_df['quantity']

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # Train Random Forest
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model.fit(X_train_scaled, y_train)

            # Train Gradient Boosting
            gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            gb_model.fit(X_train_scaled, y_train)

            # Evaluate models
            rf_pred = rf_model.predict(X_test_scaled)
            gb_pred = gb_model.predict(X_test_scaled)

            rf_mae = mean_absolute_error(y_test, rf_pred)
            rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
            rf_r2 = r2_score(y_test, rf_pred)

            gb_mae = mean_absolute_error(y_test, gb_pred)
            gb_rmse = np.sqrt(mean_squared_error(y_test, gb_pred))
            gb_r2 = r2_score(y_test, gb_pred)

            # Choose best model
            if rf_mae < gb_mae:
                best_model = rf_model
                best_model_name = 'RandomForest'
                best_metrics = {'MAE': rf_mae, 'RMSE': rf_rmse, 'R2': rf_r2}
            else:
                best_model = gb_model
                best_model_name = 'GradientBoosting'
                best_metrics = {'MAE': gb_mae, 'RMSE': gb_rmse, 'R2': gb_r2}

            # Save model
            model_path = os.path.join(self.model_dir, f'demand_forecast_{product}.pkl')
            joblib.dump(best_model, model_path)

            # Store results
            training_results[product] = {
                'model_type': best_model_name,
                'metrics': best_metrics,
                'feature_importance': dict(zip(feature_columns, best_model.feature_importances_)),
                'training_samples': len(X_train),
                'test_samples': len(X_test)
            }

            logger.info(f"Model trained for {product}: {best_model_name} - MAE: {best_metrics['MAE']:.2f}")

        return training_results

    def predict_demand(self, product: str, forecast_horizon: int = 30) -> List[Dict[str, Any]]:
        """Generate demand forecast for a specific product"""
        model_path = os.path.join(self.model_dir, f'demand_forecast_{product}.pkl')

        if not os.path.exists(model_path):
            raise ValueError(f"No trained model found for product: {product}")

        model = joblib.load(model_path)

        # Generate future dates
        last_date = datetime.now()
        future_dates = [last_date + timedelta(days=i) for i in range(1, forecast_horizon + 1)]

        # Create feature DataFrame for future dates
        future_features = []

        for date in future_dates:
            features = {
                'day_of_week': date.weekday(),
                'month': date.month,
                'quarter': (date.month - 1) // 3 + 1,
                'week_of_year': date.isocalendar().week,
                'is_weekend': 1 if date.weekday() in [5, 6] else 0,
                'is_month_end': 1 if date.day == (date + timedelta(days=1)).day else 0,
                'is_month_start': 1 if date.day == 1 else 0,
                'lag_1': 0,  # Will be filled with recent data
                'lag_7': 0,
                'lag_14': 0,
                'lag_30': 0,
                'rolling_mean_7': 0,
                'rolling_std_7': 0,
                'rolling_mean_14': 0,
                'rolling_std_14': 0,
                'rolling_mean_30': 0,
                'rolling_std_30': 0
            }
            future_features.append(features)

        # Scale features
        feature_df = pd.DataFrame(future_features)
        scaled_features = self.scaler.transform(feature_df)

        # Make predictions
        predictions = model.predict(scaled_features)

        # Create forecast results
        forecast_results = []
        for i, (date, pred) in enumerate(zip(future_dates, predictions)):
            # Calculate confidence interval (simple approximation)
            confidence = max(0.7, 1 - (i * 0.02))  # Decreasing confidence over time

            forecast_results.append({
                'date': date.strftime('%Y-%m-%d'),
                'predicted_demand': max(0, int(pred)),  # Ensure non-negative
                'confidence_score': confidence,
                'upper_bound': max(0, int(pred * 1.2)),
                'lower_bound': max(0, int(pred * 0.8))
            })

        return forecast_results

    def get_forecast_summary(self, product: str, forecast_horizon: int = 30) -> Dict[str, Any]:
        """Get summary statistics for demand forecast"""
        forecast = self.predict_demand(product, forecast_horizon)

        demands = [item['predicted_demand'] for item in forecast]

        return {
            'product': product,
            'forecast_horizon': forecast_horizon,
            'total_predicted_demand': sum(demands),
            'average_daily_demand': np.mean(demands),
            'max_daily_demand': max(demands),
            'min_daily_demand': min(demands),
            'demand_volatility': np.std(demands),
            'forecast_details': forecast
        }

    def update_model(self, product: str, new_data_path: str) -> Dict[str, Any]:
        """Update model with new data"""
        # This would implement incremental learning or retraining
        # For now, just retrain the model
        return self.train_models(new_data_path)
