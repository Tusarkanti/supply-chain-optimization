import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, RobustScaler
import joblib
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple
import logging
import requests
from dataclasses import dataclass
import holidays
from sklearn.model_selection import GridSearchCV
try:
    import xgboost as xgb
    from xgboost import XGBRegressor
    XGBoost_AVAILABLE = True
except ImportError:
    XGBoost_AVAILABLE = False
    XGBRegressor = None

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class MarketIntelligence:
    competitor_price_index: float
    market_demand_index: float
    economic_sentiment: float
    consumer_confidence: float
    promotional_activity: float
    seasonal_trend: float

@dataclass
class ExternalFactors:
    gdp_growth: float
    inflation_rate: float
    unemployment_rate: float
    fuel_price: float
    exchange_rate: float
    weather_severity: float

class EnhancedDemandForecastingModule:
    def __init__(self):
        self.models = {}
        self.model_dir = 'models/enhanced_forecasting'
        self.scaler = RobustScaler()
        os.makedirs(self.model_dir, exist_ok=True)

        # Initialize external data sources
        self.holiday_calendars = {
            'US': holidays.US(),
            'CA': holidays.CA(),
            'UK': holidays.UK(),
            'DE': holidays.DE(),
            'FR': holidays.FR()
        }

    def fetch_economic_data(self, country: str = 'US') -> ExternalFactors:
        """Fetch real-time economic indicators"""
        try:
            # In a real implementation, this would connect to APIs like:
            # - FRED API (Federal Reserve Economic Data)
            # - World Bank API
            # - OECD API
            # - Alpha Vantage for financial data

            # Mock data for demonstration
            return ExternalFactors(
                gdp_growth=np.random.normal(2.5, 0.5),
                inflation_rate=np.random.normal(2.1, 0.3),
                unemployment_rate=np.random.normal(4.2, 0.4),
                fuel_price=np.random.normal(3.50, 0.20),
                exchange_rate=np.random.normal(1.0, 0.05),
                weather_severity=np.random.beta(2, 5)  # 0-1 scale
            )
        except Exception as e:
            logger.warning(f"Could not fetch economic data: {e}")
            return ExternalFactors(2.5, 2.1, 4.2, 3.50, 1.0, 0.3)

    def fetch_market_intelligence(self, product_category: str) -> MarketIntelligence:
        """Fetch market intelligence data"""
        try:
            # Mock market intelligence data
            return MarketIntelligence(
                competitor_price_index=np.random.normal(1.0, 0.1),
                market_demand_index=np.random.normal(1.0, 0.15),
                economic_sentiment=np.random.beta(5, 2),  # 0-1 scale
                consumer_confidence=np.random.beta(4, 3),  # 0-1 scale
                promotional_activity=np.random.beta(2, 8),  # 0-1 scale
                seasonal_trend=np.sin(2 * np.pi * datetime.now().month / 12) * 0.3 + 0.7
            )
        except Exception as e:
            logger.warning(f"Could not fetch market intelligence: {e}")
            return MarketIntelligence(1.0, 1.0, 0.5, 0.5, 0.2, 1.0)

    def load_and_preprocess_data(self, sales_data_path: str, weather_data_path: str = None,
                               economic_data: Dict = None, market_data: Dict = None) -> pd.DataFrame:
        """Enhanced data loading and preprocessing with external factors"""
        df = pd.read_csv(sales_data_path)
        df['date'] = pd.to_datetime(df['date'])

        # Enhanced feature engineering
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['week_of_year'] = df['date'].dt.isocalendar().week
        df['day_of_month'] = df['date'].dt.day
        df['day_of_year'] = df['date'].dt.dayofyear
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
        df['is_month_start'] = df['date'].dt.is_month_start.astype(int)

        # Cyclical encoding for temporal features
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        # Holiday features
        for country, holiday_cal in self.holiday_calendars.items():
            df[f'is_holiday_{country}'] = df['date'].dt.date.isin(holiday_cal).astype(int)

        # Lag features with multiple periods
        for lag in [1, 7, 14, 30, 60, 90]:
            df[f'lag_{lag}'] = df.groupby('product')['quantity'].shift(lag)

        # Rolling statistics with multiple windows
        for window in [7, 14, 30, 60]:
            df[f'rolling_mean_{window}'] = df.groupby('product')['quantity'].rolling(window=window, min_periods=1).mean().reset_index(0, drop=True)
            df[f'rolling_std_{window}'] = df.groupby('product')['quantity'].rolling(window=window, min_periods=1).std().reset_index(0, drop=True)
            df[f'rolling_min_{window}'] = df.groupby('product')['quantity'].rolling(window=window, min_periods=1).min().reset_index(0, drop=True)
            df[f'rolling_max_{window}'] = df.groupby('product')['quantity'].rolling(window=window, min_periods=1).max().reset_index(0, drop=True)

        # Exponential moving averages
        for span in [7, 14, 30]:
            df[f'ema_{span}'] = df.groupby('product')['quantity'].ewm(span=span).mean().reset_index(0, drop=True)

        # Price elasticity features (if price data available)
        if 'price' in df.columns:
            df['price_change'] = df.groupby('product')['price'].pct_change()
            df['price_momentum'] = df.groupby('product')['price'].rolling(7).apply(lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] if len(x) > 1 else 0).reset_index(0, drop=True)

        # Add weather data if available
        if weather_data_path and os.path.exists(weather_data_path):
            weather_df = pd.read_csv(weather_data_path)
            weather_df['date'] = pd.to_datetime(weather_df['date'])
            df = df.merge(weather_df, on=['date', 'location'], how='left')

        # Add economic data if provided
        if economic_data:
            for key, value in economic_data.items():
                df[f'economic_{key}'] = value

        # Add market data if provided
        if market_data:
            for key, value in market_data.items():
                df[f'market_{key}'] = value

        # Fill NaN values using forward and backward fill
        df = df.bfill().ffill()

        # Remove any remaining NaN values
        df = df.dropna()

        return df

    def train_advanced_models(self, data_path: str, weather_data_path: str = None,
                            country: str = 'US') -> Dict[str, Any]:
        """Train advanced demand forecasting models with external factors"""
        # Fetch external data
        economic_data = self.fetch_economic_data(country)
        df = self.load_and_preprocess_data(data_path, weather_data_path, vars(economic_data), {})

        # Get unique products
        products = df['product'].unique()
        training_results = {}

        for product in products:
            logger.info(f"Training advanced model for product: {product}")
            product_df = df[df['product'] == product].copy()

            market_intel = self.fetch_market_intelligence(product)
            product_df = self.load_and_preprocess_data(data_path, weather_data_path, vars(economic_data), vars(market_intel))

            # Enhanced feature set
            base_features = [
                'day_of_week', 'month', 'quarter', 'week_of_year', 'day_of_month', 'day_of_year',
                'is_weekend', 'is_month_end', 'is_month_start', 'month_sin', 'month_cos',
                'day_of_week_sin', 'day_of_week_cos', 'lag_1', 'lag_7', 'lag_14', 'lag_30',
                'lag_60', 'lag_90', 'rolling_mean_7', 'rolling_std_7', 'rolling_mean_14',
                'rolling_std_14', 'rolling_mean_30', 'rolling_std_30', 'rolling_mean_60',
                'rolling_std_60', 'ema_7', 'ema_14', 'ema_30'
            ]

            # Add holiday features
            holiday_features = [f'is_holiday_{c}' for c in self.holiday_calendars.keys()]
            base_features.extend(holiday_features)

            # Add economic features
            economic_features = [f'economic_{key}' for key in vars(economic_data).keys()]
            base_features.extend(economic_features)

            # Add market features
            market_features = [f'market_{key}' for key in vars(market_intel).keys()]
            base_features.extend(market_features)

            # Add weather features if available
            weather_features = ['temperature', 'humidity', 'precipitation', 'wind_speed', 'weather_severity']
            for feature in weather_features:
                if feature in product_df.columns:
                    base_features.append(feature)

            # Add price features if available
            price_features = ['price_change', 'price_momentum']
            for feature in price_features:
                if feature in product_df.columns:
                    base_features.append(feature)

            X = product_df[base_features]
            y = product_df['quantity']

            # Use time series split for better validation
            tscv = TimeSeriesSplit(n_splits=5)

            # Train multiple models with hyperparameter tuning
            models = {
                'RandomForest': RandomForestRegressor(random_state=42),
                'GradientBoosting': GradientBoostingRegressor(random_state=42),
                'ExtraTrees': ExtraTreesRegressor(random_state=42),
                'LinearRegression': LinearRegression()
            }

            if XGBoost_AVAILABLE:
                models['XGBoost'] = XGBRegressor(random_state=42, objective='reg:squarederror')

            # Hyperparameter grids
            param_grids = {
                'RandomForest': {'n_estimators': [100, 200], 'max_depth': [10, 20, None]},
                'GradientBoosting': {'n_estimators': [100, 200], 'learning_rate': [0.1, 0.05]},
                'ExtraTrees': {'n_estimators': [100, 200], 'max_depth': [10, 20, None]},
                'LinearRegression': {}
            }

            if XGBoost_AVAILABLE:
                param_grids['XGBoost'] = {'n_estimators': [100, 200], 'max_depth': [3, 6, 9], 'learning_rate': [0.1, 0.05]}

            trained_models = {}
            model_scores = {}

            for model_name, model in models.items():
                # Perform grid search with time series cross-validation
                grid_search = GridSearchCV(
                    model, param_grids[model_name], cv=tscv, scoring='r2', n_jobs=-1
                )
                grid_search.fit(X, y)

                trained_models[model_name] = grid_search.best_estimator_
                model_scores[model_name] = grid_search.best_score_

            # Create weighted ensemble based on CV scores
            # Normalize scores to weights (add small epsilon to avoid division by zero)
            scores = np.array(list(model_scores.values()))
            weights = (scores - scores.min() + 1e-6) / (scores.max() - scores.min() + 1e-6)
            weights = weights / weights.sum()  # Normalize to sum to 1

            ensemble_weights = dict(zip(model_scores.keys(), weights))

            # Scale features for final models
            X_scaled = self.scaler.fit_transform(X)

            # Train final models on all data
            final_models = {}
            for model_name, model in trained_models.items():
                model.fit(X_scaled, y)
                final_models[model_name] = model

            # Save ensemble and scaler
            ensemble_data = {
                'models': final_models,
                'weights': ensemble_weights,
                'feature_names': base_features,
                'model_scores': model_scores
            }

            model_path = os.path.join(self.model_dir, f'enhanced_forecast_{product}.pkl')
            scaler_path = os.path.join(self.model_dir, f'scaler_{product}.pkl')

            joblib.dump(ensemble_data, model_path)
            joblib.dump(self.scaler, scaler_path)

            # Calculate average feature importance across models that support it
            feature_importance = {}
            for feature in base_features:
                importance_sum = 0
                count = 0
                for model_name, model in final_models.items():
                    if hasattr(model, 'feature_importances_'):
                        importance_sum += model.feature_importances_[base_features.index(feature)]
                        count += 1
                if count > 0:
                    feature_importance[feature] = importance_sum / count
                else:
                    feature_importance[feature] = 0

            # Store results
            training_results[product] = {
                'model_type': 'Ensemble',
                'best_score': np.mean(list(model_scores.values())),
                'feature_importance': feature_importance,
                'training_samples': len(X),
                'feature_count': len(base_features),
                'economic_factors': vars(economic_data),
                'market_factors': vars(market_intel),
                'ensemble_weights': ensemble_weights,
                'individual_scores': model_scores
            }

            logger.info(f"Ensemble model trained for {product}: Average R2: {np.mean(list(model_scores.values())):.4f}")
        return training_results

    def predict_demand_with_confidence(self, product: str, forecast_horizon: int = 30,
                                     country: str = 'US') -> List[Dict[str, Any]]:
        """Generate demand forecast with confidence intervals and external factors"""
        model_path = os.path.join(self.model_dir, f'enhanced_forecast_{product}.pkl')
        scaler_path = os.path.join(self.model_dir, f'scaler_{product}.pkl')

        if not os.path.exists(model_path):
            logger.warning(f"No trained model found for product: {product}, returning mock data")
            return self._generate_mock_forecast(product, forecast_horizon, country)

        ensemble_data = joblib.load(model_path)
        scaler = joblib.load(scaler_path)

        models = ensemble_data['models']
        weights = ensemble_data['weights']
        feature_names = ensemble_data['feature_names']

        # Fetch current external data
        economic_data = self.fetch_economic_data(country)
        market_intel = self.fetch_market_intelligence('general')

        # Generate future dates
        future_dates = [datetime.now() + timedelta(days=i) for i in range(1, forecast_horizon + 1)]

        # Create future features matching exactly the feature_names order
        future_data = np.zeros((len(future_dates), len(feature_names)))

        weather_defaults = {
            'temperature': 20.0,
            'humidity': 50.0,
            'precipitation': 0.0,
            'wind_speed': 10.0,
            'weather_severity': 0.3
        }

        for i, date in enumerate(future_dates):
            row = i

            # Set temporal and cyclical features
            if 'day_of_week' in feature_names:
                future_data[row, feature_names.index('day_of_week')] = date.weekday()
            if 'month' in feature_names:
                future_data[row, feature_names.index('month')] = date.month
            if 'quarter' in feature_names:
                future_data[row, feature_names.index('quarter')] = (date.month - 1) // 3 + 1
            if 'week_of_year' in feature_names:
                future_data[row, feature_names.index('week_of_year')] = date.isocalendar().week
            if 'day_of_month' in feature_names:
                future_data[row, feature_names.index('day_of_month')] = date.day
            if 'day_of_year' in feature_names:
                future_data[row, feature_names.index('day_of_year')] = date.timetuple().tm_yday
            if 'is_weekend' in feature_names:
                future_data[row, feature_names.index('is_weekend')] = 1 if date.weekday() in [5, 6] else 0
            if 'is_month_end' in feature_names:
                future_data[row, feature_names.index('is_month_end')] = 1 if date.day == (date + timedelta(days=1)).day else 0
            if 'is_month_start' in feature_names:
                future_data[row, feature_names.index('is_month_start')] = 1 if date.day == 1 else 0
            if 'month_sin' in feature_names:
                future_data[row, feature_names.index('month_sin')] = np.sin(2 * np.pi * date.month / 12)
            if 'month_cos' in feature_names:
                future_data[row, feature_names.index('month_cos')] = np.cos(2 * np.pi * date.month / 12)
            if 'day_of_week_sin' in feature_names:
                future_data[row, feature_names.index('day_of_week_sin')] = np.sin(2 * np.pi * date.weekday() / 7)
            if 'day_of_week_cos' in feature_names:
                future_data[row, feature_names.index('day_of_week_cos')] = np.cos(2 * np.pi * date.weekday() / 7)

            # Set lag, rolling, ema to 0 (as they are historical)
            lag_features = [f'lag_{lag}' for lag in [1, 7, 14, 30, 60, 90]]
            for feat in lag_features:
                if feat in feature_names:
                    future_data[row, feature_names.index(feat)] = 0
            rolling_features = [f'rolling_{stat}_{window}' for stat in ['mean', 'std', 'min', 'max'] for window in [7, 14, 30, 60]]
            for feat in rolling_features:
                if feat in feature_names:
                    future_data[row, feature_names.index(feat)] = 0
            ema_features = [f'ema_{span}' for span in [7, 14, 30]]
            for feat in ema_features:
                if feat in feature_names:
                    future_data[row, feature_names.index(feat)] = 0

            # Add holiday features
            for country_code, holiday_cal in self.holiday_calendars.items():
                col = f'is_holiday_{country_code}'
                if col in feature_names:
                    future_data[row, feature_names.index(col)] = 1 if date.date() in holiday_cal else 0

            # Add economic features
            for key, value in vars(economic_data).items():
                col = f'economic_{key}'
                if col in feature_names:
                    future_data[row, feature_names.index(col)] = value

            # Add market intelligence
            for key, value in vars(market_intel).items():
                col = f'market_{key}'
                if col in feature_names:
                    future_data[row, feature_names.index(col)] = value

            # Add weather defaults if features present
            for feat, default in weather_defaults.items():
                if feat in feature_names:
                    future_data[row, feature_names.index(feat)] = default

            # Add price features to 0 if present
            for feat in ['price_change', 'price_momentum']:
                if feat in feature_names:
                    future_data[row, feature_names.index(feat)] = 0

        # Scale features
        scaled_features = scaler.transform(future_data)

        # Make ensemble predictions
        predictions = []
        individual_predictions = {model_name: [] for model_name in models.keys()}

        for i in range(len(scaled_features)):
            ensemble_pred = 0
            for model_name, model in models.items():
                pred = model.predict(scaled_features[i:i+1])[0]
                individual_predictions[model_name].append(pred)
                ensemble_pred += weights[model_name] * pred
            predictions.append(ensemble_pred)

        predictions = np.array(predictions)

        # Calculate SHAP values if available (using the best performing model)
        shap_values = None
        if SHAP_AVAILABLE:
            try:
                best_model_name = max(weights, key=weights.get)
                best_model = models[best_model_name]
                if hasattr(best_model, 'predict'):
                    if hasattr(best_model, 'feature_importances_'):
                        explainer = shap.TreeExplainer(best_model)
                    else:
                        explainer = shap.LinearExplainer(best_model, scaled_features[:min(10, len(scaled_features))])
                    shap_values = explainer.shap_values(scaled_features)
            except Exception as e:
                logger.warning(f"Could not compute SHAP values: {e}")

        # Generate confidence intervals using ensemble variance
        model_preds = np.array([individual_predictions[model_name] for model_name in models.keys()])
        prediction_std = np.std(model_preds, axis=0).mean() * 0.3

        # Create forecast results
        forecast_results = []
        for i, (date, pred) in enumerate(zip(future_dates, predictions)):
            # Calculate confidence based on forecast horizon and model uncertainty
            base_confidence = max(0.6, 0.95 - (i * 0.02))  # Decreasing confidence over time

            # Adjust confidence based on external factors
            economic_volatility = np.std(list(vars(economic_data).values()))
            confidence_multiplier = max(0.7, 1 - economic_volatility * 0.5)

            final_confidence = base_confidence * confidence_multiplier

            # Calculate prediction intervals
            z_score = 1.96  # 95% confidence interval
            margin_of_error = z_score * prediction_std * np.sqrt(i + 1)

            result = {
                'date': date.strftime('%Y-%m-%d'),
                'predicted_demand': max(0, int(pred)),
                'confidence_score': float(final_confidence),
                'upper_bound': max(0, int(pred + margin_of_error)),
                'lower_bound': max(0, int(pred - margin_of_error)),
                'economic_factors': vars(economic_data),
                'market_intelligence': vars(market_intel),
                'risk_factors': self._calculate_risk_factors(economic_data, market_intel, i),
                'ensemble_weights': weights,
                'individual_predictions': {model_name: individual_predictions[model_name][i] for model_name in models.keys()}
            }

            if shap_values is not None:
                if isinstance(shap_values, list) and len(shap_values) > 0:
                    result['shap_values'] = shap_values[0][i].tolist() if hasattr(shap_values[0][i], 'tolist') else list(shap_values[0][i])
                else:
                    result['shap_values'] = list(shap_values[i]) if hasattr(shap_values[i], '__iter__') else [shap_values[i]]
                result['feature_names'] = feature_names

            forecast_results.append(result)

        return forecast_results

    def _generate_mock_forecast(self, product: str, forecast_horizon: int, country: str) -> List[Dict[str, Any]]:
        """Generate mock forecast data when no trained model exists"""
        future_dates = [datetime.now() + timedelta(days=i) for i in range(1, forecast_horizon + 1)]

        # Mock economic and market data
        economic_data = self.fetch_economic_data(country)
        market_intel = self.fetch_market_intelligence('general')

        forecast_results = []
        for i, date in enumerate(future_dates):
            # Generate mock predictions with some trend and seasonality
            base_demand = 100
            trend = i * 2  # Slight upward trend
            seasonality = np.sin(2 * np.pi * i / 7) * 20  # Weekly seasonality
            noise = np.random.normal(0, 10)
            predicted_demand = max(0, int(base_demand + trend + seasonality + noise))

            # Mock confidence decreasing over time
            confidence = max(0.7, 0.95 - (i * 0.02))

            result = {
                'date': date.strftime('%Y-%m-%d'),
                'predicted_demand': predicted_demand,
                'confidence_score': float(confidence),
                'upper_bound': int(predicted_demand * 1.2),
                'lower_bound': int(predicted_demand * 0.8),
                'economic_factors': vars(economic_data),
                'market_intelligence': vars(market_intel),
                'risk_factors': self._calculate_risk_factors(economic_data, market_intel, i),
                'ensemble_weights': {'RandomForest': 0.4, 'GradientBoosting': 0.35, 'XGBoost': 0.25},
                'individual_predictions': {
                    'RandomForest': predicted_demand + np.random.normal(0, 5),
                    'GradientBoosting': predicted_demand + np.random.normal(0, 3),
                    'XGBoost': predicted_demand + np.random.normal(0, 4)
                }
            }
            forecast_results.append(result)

        return forecast_results

    def _calculate_risk_factors(self, economic_data: ExternalFactors,
                              market_intel: MarketIntelligence, forecast_horizon: int) -> Dict[str, float]:
        """Calculate risk factors for the forecast"""
        risks = {
            'economic_risk': np.std(list(vars(economic_data).values())),
            'market_risk': abs(market_intel.market_demand_index - 1.0),
            'competitive_risk': abs(market_intel.competitor_price_index - 1.0),
            'volatility_risk': 1 - market_intel.economic_sentiment,
            'horizon_risk': min(1.0, forecast_horizon / 90)  # Risk increases with horizon
        }

        # Calculate overall risk score
        risks['overall_risk'] = np.mean(list(risks.values()))

        return risks

    def get_forecast_summary(self, product: str, forecast_horizon: int = 30,
                           country: str = 'US') -> Dict[str, Any]:
        """Get comprehensive forecast summary with risk analysis"""
        forecast = self.predict_demand_with_confidence(product, forecast_horizon, country)

        demands = [item['predicted_demand'] for item in forecast]
        confidences = [item['confidence_score'] for item in forecast]
        risks = [item['risk_factors']['overall_risk'] for item in forecast]

        # Calculate advanced metrics
        demand_trend = np.polyfit(range(len(demands)), demands, 1)[0]
        demand_volatility = np.std(demands) / np.mean(demands) if np.mean(demands) > 0 else 0

        return {
            'product': product,
            'forecast_horizon': forecast_horizon,
            'total_predicted_demand': sum(demands),
            'average_daily_demand': np.mean(demands),
            'demand_trend': demand_trend,
            'demand_volatility': demand_volatility,
            'average_confidence': np.mean(confidences),
            'average_risk': np.mean(risks),
            'max_demand': max(demands),
            'min_demand': min(demands),
            'forecast_details': forecast,
            'recommendations': self._generate_forecast_recommendations(forecast, demand_trend, demand_volatility)
        }

    def _generate_forecast_recommendations(self, forecast: List[Dict], trend: float,
                                        volatility: float) -> List[str]:
        """Generate actionable recommendations based on forecast"""
        recommendations = []

        if trend > 0.1:
            recommendations.append("Consider increasing inventory levels due to upward demand trend")
        elif trend < -0.1:
            recommendations.append("Monitor inventory closely due to declining demand trend")

        if volatility > 0.3:
            recommendations.append("High demand volatility detected - consider flexible supply arrangements")
        elif volatility < 0.1:
            recommendations.append("Stable demand pattern - good opportunity for long-term contracts")

        # Check for high-risk periods
        high_risk_periods = [item for item in forecast if item['risk_factors']['overall_risk'] > 0.7]
        if high_risk_periods:
            recommendations.append(f"Identified {len(high_risk_periods)} high-risk periods - consider contingency planning")

        return recommendations

    def retrain_model(self, product: str, new_data_path: str, weather_data_path: str = None,
                     country: str = 'US', force_retrain: bool = False) -> Dict[str, Any]:
        """Automated model retraining with performance monitoring"""
        model_path = os.path.join(self.model_dir, f'enhanced_forecast_{product}.pkl')

        # Check if retraining is needed
        if not force_retrain and os.path.exists(model_path):
            model_stats = os.stat(model_path)
            days_since_training = (datetime.now() - datetime.fromtimestamp(model_stats.st_mtime)).days

            if days_since_training < 30:
                return {
                    'status': 'no_retrain_needed',
                    'message': f'Model for {product} is only {days_since_training} days old',
                    'last_trained': datetime.fromtimestamp(model_stats.st_mtime).isoformat()
                }

        logger.info(f"Retraining model for product: {product}")

        training_results = self.train_advanced_models(new_data_path, weather_data_path, country)

        if product in training_results:
            validation_results = self.validate_model_performance(product, new_data_path, weather_data_path)

            return {
                'status': 'retrained',
                'product': product,
                'training_results': training_results[product],
                'validation_results': validation_results,
                'retrained_at': datetime.now().isoformat()
            }
        else:
            return {
                'status': 'error',
                'message': f'Failed to retrain model for {product}'
            }

    def validate_model_performance(self, product: str, validation_data_path: str,
                                 weather_data_path: str = None) -> Dict[str, Any]:
        """Validate model performance on new data"""
        model_path = os.path.join(self.model_dir, f'enhanced_forecast_{product}.pkl')
        scaler_path = os.path.join(self.model_dir, f'scaler_{product}.pkl')

        if not os.path.exists(model_path):
            return {'error': f'No trained model found for product: {product}'}

        # Load model and scaler
        ensemble_data = joblib.load(model_path)
        scaler = joblib.load(scaler_path)

        # Load validation data
        df = self.load_and_preprocess_data(validation_data_path, weather_data_path)

        if product not in df['product'].unique():
            return {'error': f'Product {product} not found in validation data'}

        product_df = df[df['product'] == product].copy()

        # Prepare features
        base_features = ensemble_data['feature_names']
        X = product_df[base_features]
        y_true = product_df['quantity']

        # Scale features
        X_scaled = scaler.transform(X)

        # Make predictions
        models = ensemble_data['models']
        weights = ensemble_data['weights']

        ensemble_predictions = []
        individual_predictions = {model_name: [] for model_name in models.keys()}

        for i in range(len(X_scaled)):
            ensemble_pred = 0
            for model_name, model in models.items():
                pred = model.predict(X_scaled[i:i+1])[0]
                individual_predictions[model_name].append(pred)
                ensemble_pred += weights[model_name] * pred
            ensemble_predictions.append(ensemble_pred)

        # Calculate metrics
        ensemble_mae = mean_absolute_error(y_true, ensemble_predictions)
        ensemble_rmse = np.sqrt(mean_squared_error(y_true, ensemble_predictions))
        ensemble_r2 = r2_score(y_true, ensemble_predictions)

        individual_metrics = {}
        for model_name, preds in individual_predictions.items():
            individual_metrics[model_name] = {
                'mae': mean_absolute_error(y_true, preds),
                'rmse': np.sqrt(mean_squared_error(y_true, preds)),
                'r2': r2_score(y_true, preds)
            }

        return {
            'ensemble_metrics': {
                'mae': ensemble_mae,
                'rmse': ensemble_rmse,
                'r2': ensemble_r2,
                'sample_size': len(y_true)
            },
            'individual_metrics': individual_metrics,
            'performance_comparison': {
                'best_individual_model': max(individual_metrics.keys(),
                                           key=lambda x: individual_metrics[x]['r2']),
                'ensemble_improvement': ensemble_r2 - max(m['r2'] for m in individual_metrics.values())
            }
        }

    def get_model_health_status(self, product: str) -> Dict[str, Any]:
        """Get comprehensive model health and performance status"""
        model_path = os.path.join(self.model_dir, f'enhanced_forecast_{product}.pkl')

        if not os.path.exists(model_path):
            return {
                'status': 'no_model',
                'product': product,
                'message': 'No trained model found'
            }

        model_stats = os.stat(model_path)
        last_trained = datetime.fromtimestamp(model_stats.st_mtime)
        days_since_training = (datetime.now() - last_trained).days

        # Load model data
        ensemble_data = joblib.load(model_path)

        health_status = {
            'product': product,
            'status': 'healthy',
            'last_trained': last_trained.isoformat(),
            'days_since_training': days_since_training,
            'model_info': {
                'type': 'Ensemble',
                'num_models': len(ensemble_data['models']),
                'feature_count': len(ensemble_data['feature_names']),
                'training_samples': ensemble_data.get('training_samples', 'unknown')
            },
            'performance_metrics': ensemble_data.get('model_scores', {}),
            'ensemble_weights': ensemble_data['weights']
        }

        # Determine health status
        if days_since_training > 90:
            health_status['status'] = 'needs_retraining'
            health_status['issues'] = ['Model is older than 90 days']
        elif days_since_training > 60:
            health_status['status'] = 'aging'
            health_status['warnings'] = ['Model is getting old, consider retraining soon']

        # Check for performance issues
        avg_r2 = np.mean(list(ensemble_data.get('model_scores', {}).values()))
        if avg_r2 < 0.5:
            health_status['status'] = 'poor_performance'

        return health_status

    def schedule_retraining(self, products: List[str], data_path: str,
                          weather_data_path: str = None, country: str = 'US') -> Dict[str, Any]:
        """Schedule automated retraining for multiple products"""
        results = {}

        for product in products:
            try:
                result = self.retrain_model(product, data_path, weather_data_path, country)
                results[product] = result
                logger.info(f"Retraining completed for {product}: {result['status']}")
            except Exception as e:
                results[product] = {
                    'status': 'error',
                    'error': str(e)
                }
                logger.error(f"Retraining failed for {product}: {str(e)}")

        return {
            'scheduled_at': datetime.now().isoformat(),
            'results': results,
            'summary': {
                'total_products': len(products),
                'successful_retrains': len([r for r in results.values() if r.get('status') == 'retrained']),
                'failed_retrains': len([r for r in results.values() if r.get('status') == 'error']),
                'skipped': len([r for r in results.values() if r.get('status') == 'no_retrain_needed'])
            }
        }
