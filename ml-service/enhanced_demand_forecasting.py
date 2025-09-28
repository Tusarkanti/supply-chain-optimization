import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV, cross_val_score
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
    competitor_price_index: float = 1.0
    market_demand_index: float = 1.0
    economic_sentiment: float = 0.5
    consumer_confidence: float = 0.5
    promotional_activity: float = 0.0
    seasonal_trend: float = 1.0

@dataclass
class ExternalFactors:
    gdp_growth: float = 2.5
    inflation_rate: float = 2.1
    unemployment_rate: float = 5.0
    fuel_price: float = 3.5
    exchange_rate: float = 1.0
    weather_severity: float = 0.0

class EnhancedDemandForecastingModule:
    def __init__(self):
        self.models = {}
        self.model_dir = 'models/enhanced_forecasting'
        self.scaler = RobustScaler()
        self.validation_metrics = {}
        self.feature_names = None
        os.makedirs(self.model_dir, exist_ok=True)

        # Initialize external data sources
        self.holiday_calendars = {
            'US': holidays.US(),
            'CA': holidays.CA(),
            'UK': holidays.UK(),
            'DE': holidays.DE(),
            'FR': holidays.FR()
        }

        # Load existing models and metrics if available
        self._load_models_and_metrics()

    def _load_models_and_metrics(self):
        """Load saved models and validation metrics"""
        for model_name in ['random_forest', 'gradient_boosting', 'extra_trees', 'linear_regression']:
            model_path = os.path.join(self.model_dir, f'{model_name}_model.pkl')
            metrics_path = os.path.join(self.model_dir, f'{model_name}_metrics.pkl')
            if os.path.exists(model_path):
                self.models[model_name] = joblib.load(model_path)
            if os.path.exists(metrics_path):
                self.validation_metrics[model_name] = joblib.load(metrics_path)
        if XGBoost_AVAILABLE and os.path.exists(os.path.join(self.model_dir, 'xgboost_model.pkl')):
            self.models['xgboost'] = joblib.load(os.path.join(self.model_dir, 'xgboost_model.pkl'))
            if os.path.exists(os.path.join(self.model_dir, 'xgboost_metrics.pkl')):
                self.validation_metrics['xgboost'] = joblib.load(os.path.join(self.model_dir, 'xgboost_metrics.pkl'))

        # If no models loaded, train with mock data to ensure consistency
        if not self.models:
            logger.info("No saved models found. Training with mock data for demonstration.")
            mock_data = self._generate_mock_training_data()
            self.train_advanced_models(mock_data)

        # Load feature names and scaler if available
        feature_path = os.path.join(self.model_dir, 'feature_names.pkl')
        scaler_path = os.path.join(self.model_dir, 'scaler.pkl')
        if os.path.exists(feature_path):
            self.feature_names = joblib.load(feature_path)
        else:
            # Try to get feature names from loaded models
            if self.models:
                model = next(iter(self.models.values()))
                if hasattr(model, 'feature_names_in_'):
                    self.feature_names = list(model.feature_names_in_)
                else:
                    # Default feature names if not saved (for backward compatibility)
                    self.feature_names = [
                        'day_of_week', 'month', 'quarter', 'week_of_year', 'is_weekend', 'is_month_end', 'is_month_start', 'is_holiday',
                        'lag_1', 'lag_7', 'lag_14', 'lag_30',
                        'rolling_mean_7', 'rolling_std_7', 'rolling_mean_14', 'rolling_std_14', 'rolling_mean_30', 'rolling_std_30',
                        'gdp_growth', 'inflation_rate', 'unemployment_rate', 'fuel_price', 'exchange_rate', 'weather_severity',
                        'competitor_price_index', 'market_demand_index', 'economic_sentiment', 'consumer_confidence', 'promotional_activity', 'seasonal_trend',
                        'temperature', 'humidity', 'precipitation'
                    ]
            else:
                # Default feature names if not saved (for backward compatibility)
                self.feature_names = [
                    'day_of_week', 'month', 'quarter', 'week_of_year', 'is_weekend', 'is_month_end', 'is_month_start', 'is_holiday',
                    'lag_1', 'lag_7', 'lag_14', 'lag_30',
                    'rolling_mean_7', 'rolling_std_7', 'rolling_mean_14', 'rolling_std_14', 'rolling_mean_30', 'rolling_std_30',
                    'gdp_growth', 'inflation_rate', 'unemployment_rate', 'fuel_price', 'exchange_rate', 'weather_severity',
                    'competitor_price_index', 'market_demand_index', 'economic_sentiment', 'consumer_confidence', 'promotional_activity', 'seasonal_trend',
                    'temperature', 'humidity', 'precipitation'
                ]
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
        else:
            # If scaler not saved, fit on dummy data for compatibility
            import numpy as np
            dummy_data = np.zeros((10, len(self.feature_names)))
            self.scaler.fit(dummy_data)

    def fetch_economic_data(self, country: str = 'US') -> ExternalFactors:
        """Fetch real-time economic indicators"""
        try:
            # Mock data for demonstration - in production, use APIs like FRED, World Bank
            return ExternalFactors(
                gdp_growth=np.random.normal(2.5, 0.5),
                inflation_rate=np.random.normal(2.1, 0.3),
                unemployment_rate=np.random.normal(5.0, 0.5),
                fuel_price=np.random.normal(3.5, 0.5),
                exchange_rate=np.random.normal(1.0, 0.05),
                weather_severity=np.random.uniform(0, 1)
            )
        except Exception as e:
            logger.warning(f"Failed to fetch economic data: {e}")
            return ExternalFactors()

    def load_and_preprocess_data(self, sales_data_path: str, weather_data_path: str = None, country: str = 'US') -> pd.DataFrame:
        """Load and preprocess sales data with external factors"""
        df = pd.read_csv(sales_data_path)
        df['date'] = pd.to_datetime(df['date'])

        # Basic temporal features
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['week_of_year'] = df['date'].dt.isocalendar().week
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
        df['is_month_start'] = df['date'].dt.is_month_start.astype(int)

        # Holiday features
        df['is_holiday'] = df['date'].apply(lambda x: 1 if x in self.holiday_calendars.get(country, holidays.US()) else 0)

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
            df = df.merge(weather_df, on='date', how='left')
            weather_columns = ['temperature', 'humidity', 'precipitation']
            for col in weather_columns:
                if col in df.columns:
                    df[col] = df[col].ffill().bfill().fillna(df[col].mean())

        # Add market intelligence and external factors (mock for now)
        economic_data = self.fetch_economic_data(country)
        df['gdp_growth'] = economic_data.gdp_growth
        df['inflation_rate'] = economic_data.inflation_rate
        df['unemployment_rate'] = economic_data.unemployment_rate
        df['fuel_price'] = economic_data.fuel_price
        df['exchange_rate'] = economic_data.exchange_rate
        df['weather_severity'] = economic_data.weather_severity

        market_data = MarketIntelligence()
        df['competitor_price_index'] = market_data.competitor_price_index
        df['market_demand_index'] = market_data.market_demand_index
        df['economic_sentiment'] = market_data.economic_sentiment
        df['consumer_confidence'] = market_data.consumer_confidence
        df['promotional_activity'] = market_data.promotional_activity
        df['seasonal_trend'] = market_data.seasonal_trend

        # Fill NaNs in lag and rolling features
        feature_cols = [col for col in df.columns if col.startswith('lag_') or col.startswith('rolling_')]
        for col in feature_cols:
            df[col] = df[col].ffill().fillna(0)

        return df

    def train_advanced_models(self, data: pd.DataFrame, target_col: str = 'quantity') -> Dict[str, Any]:
        """Train ensemble models with hyperparameter tuning and validation metrics"""
        X = data.drop(columns=[target_col, 'date', 'product', 'location'])
        y = data[target_col]
        self.feature_names = X.columns.tolist()

        # Define models and params
        models_config = {
            'random_forest': (RandomForestRegressor(random_state=42), {'n_estimators': [100, 200], 'max_depth': [10, 20]}),
            'gradient_boosting': (GradientBoostingRegressor(random_state=42), {'n_estimators': [100, 200], 'max_depth': [3, 5]}),
            'extra_trees': (ExtraTreesRegressor(random_state=42), {'n_estimators': [100, 200], 'max_depth': [10, 20]}),
            'linear_regression': (LinearRegression(), {}),
        }
        if XGBoost_AVAILABLE:
            models_config['xgboost'] = (XGBRegressor(random_state=42), {'n_estimators': [100, 200], 'max_depth': [3, 6]})

        self.validation_metrics = {}
        trained_models = {}
        cv_scores = {}

        tscv = TimeSeriesSplit(n_splits=5)

        for name, (model, params) in models_config.items():
            logger.info(f"Training {name}...")
            if params:
                grid_search = GridSearchCV(model, params, cv=tscv, scoring='r2', n_jobs=-1)
                grid_search.fit(X, y)
                best_model = grid_search.best_estimator_
                cv_scores[name] = grid_search.best_score_
            else:
                best_model = model
                scores = cross_val_score(best_model, X, y, cv=tscv, scoring='r2')
                cv_scores[name] = scores.mean()

            best_model.fit(X, y)
            trained_models[name] = best_model

            # Compute validation metrics on hold-out set
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
            y_pred = best_model.predict(X_val)
            metrics = {
                'mae': mean_absolute_error(y_val, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_val, y_pred)),
                'r2': r2_score(y_val, y_pred)
            }
            self.validation_metrics[name] = metrics

            # Save model and metrics
            joblib.dump(best_model, os.path.join(self.model_dir, f'{name}_model.pkl'))
            joblib.dump(metrics, os.path.join(self.model_dir, f'{name}_metrics.pkl'))

        # Fit scaler on full data
        self.scaler.fit(X)

        # Calculate ensemble weights based on CV scores
        total_score = sum(cv_scores.values())
        ensemble_weights = {name: score / total_score for name, score in cv_scores.items()}

        # Save ensemble weights and feature names
        joblib.dump(ensemble_weights, os.path.join(self.model_dir, 'ensemble_weights.pkl'))
        joblib.dump(self.feature_names, os.path.join(self.model_dir, 'feature_names.pkl'))
        joblib.dump(self.scaler, os.path.join(self.model_dir, 'scaler.pkl'))

        logger.info(f"Training completed. Ensemble R²: {np.mean(list(cv_scores.values())):.4f}")
        return {
            'models': trained_models,
            'weights': ensemble_weights,
            'cv_scores': cv_scores,
            'validation_metrics': self.validation_metrics
        }

    def predict_demand_with_confidence(self, product: str, forecast_horizon: int = 30, country: str = 'US') -> List[Dict]:
        """Generate ensemble predictions with confidence and individual model outputs"""
        if not self.models:
            logger.warning("No trained models found. Please train models first.")
            return []

        if not self.feature_names:
            logger.error("Feature names not available. Please retrain models.")
            return []

        # Load recent data for the product (mock for now - in production, query DB)
        # Assume we have last 60 days data
        recent_data = self._get_recent_product_data(product, days=60)
        if recent_data.empty:
            return []

        # Generate future dates
        last_date = recent_data['date'].max()
        future_dates = [last_date + timedelta(days=i+1) for i in range(forecast_horizon)]

        # Prepare future features
        future_features = []
        for date in future_dates:
            features = {
                'day_of_week': date.weekday(),
                'month': date.month,
                'quarter': (date.month - 1) // 3 + 1,
                'week_of_year': date.isocalendar()[1],
                'is_weekend': 1 if date.weekday() >= 5 else 0,
                'is_month_end': 1 if date.day == date.replace(day=28).day else 0,
                'is_month_start': 1 if date.day == 1 else 0,
                'is_holiday': 1 if date in self.holiday_calendars.get(country, holidays.US()) else 0,
                # Lag and rolling from recent data
                'lag_1': recent_data['quantity'].iloc[-1],
                'lag_7': recent_data['quantity'].iloc[-7] if len(recent_data) >= 7 else 0,
                'lag_14': recent_data['quantity'].iloc[-14] if len(recent_data) >= 14 else 0,
                'lag_30': recent_data['quantity'].iloc[-30] if len(recent_data) >= 30 else 0,
                'rolling_mean_7': recent_data['quantity'].tail(7).mean(),
                'rolling_std_7': recent_data['quantity'].tail(7).std(),
                'rolling_mean_14': recent_data['quantity'].tail(14).mean(),
                'rolling_std_14': recent_data['quantity'].tail(14).std(),
                'rolling_mean_30': recent_data['quantity'].tail(30).mean(),
                'rolling_std_30': recent_data['quantity'].tail(30).std(),
                # External factors
                'gdp_growth': self.fetch_economic_data(country).gdp_growth,
                'inflation_rate': self.fetch_economic_data(country).inflation_rate,
                'unemployment_rate': self.fetch_economic_data(country).unemployment_rate,
                'fuel_price': self.fetch_economic_data(country).fuel_price,
                'exchange_rate': self.fetch_economic_data(country).exchange_rate,
                'weather_severity': self.fetch_economic_data(country).weather_severity,
                # Market intelligence
                'competitor_price_index': MarketIntelligence().competitor_price_index,
                'market_demand_index': MarketIntelligence().market_demand_index,
                'economic_sentiment': MarketIntelligence().economic_sentiment,
                'consumer_confidence': MarketIntelligence().consumer_confidence,
                'promotional_activity': MarketIntelligence().promotional_activity,
                'seasonal_trend': MarketIntelligence().seasonal_trend,
                # Weather (mock)
                'temperature': 20.0,
                'humidity': 50.0,
                'precipitation': 0.0
            }
            future_features.append(features)

        # Ensure feature order matches training
        feature_df = pd.DataFrame(future_features)
        feature_df = feature_df.reindex(columns=self.feature_names, fill_value=0)  # feature_names from training
        scaled_features = self.scaler.transform(feature_df)

        # Ensemble prediction
        ensemble_weights = joblib.load(os.path.join(self.model_dir, 'ensemble_weights.pkl'))
        predictions = []
        individual_predictions_list = []

        for i in range(len(scaled_features)):
            individual_preds = {}
            ensemble_pred = 0.0
            for model_name, model in self.models.items():
                pred = model.predict(scaled_features[i:i+1])[0]
                pred_float = float(pred)
                weight = ensemble_weights.get(model_name, 0.0)
                individual_preds[model_name] = pred_float
                ensemble_pred += pred_float * weight
            individual_predictions_list.append(individual_preds)
            predictions.append(ensemble_pred)

        # Confidence based on ensemble variance and external factors
        variances = [np.var([ind['random_forest'], ind['gradient_boosting']]) for ind in individual_predictions_list]  # Simplified
        max_var = np.max(variances) if np.max(variances) > 0 else 1
        confidence_scores = [float(1 - var / max_var) for var in variances]

        # Risk factors (simplified)
        risk_factors = [{'overall_risk': np.random.uniform(0.2, 0.8)} for _ in range(forecast_horizon)]

        forecast_results = []
        for i, (date, pred, conf, ind_preds, risk) in enumerate(zip(future_dates, predictions, confidence_scores, individual_predictions_list, risk_factors)):
            pred_float = float(pred)
            conf_float = float(conf)
            upper_bound = max(0, int(pred_float * (1 + 0.2 * (1 - conf_float))))
            lower_bound = max(0, int(pred_float * (1 - 0.2 * (1 - conf_float))))
            forecast_results.append({
                'date': date.strftime('%Y-%m-%d'),
                'predicted_demand': max(0, int(pred_float)),
                'confidence_score': conf_float,
                'upper_bound': upper_bound,
                'lower_bound': lower_bound,
                'individual_predictions': ind_preds,
                'risk_factors': risk
            })

        # SHAP explanation if available
        if SHAP_AVAILABLE and self.models:
            # Simplified SHAP for first prediction
            explainer = shap.Explainer(list(self.models.values())[0])
            shap_values = explainer(scaled_features[:1])
            logger.info(f"SHAP values computed for feature importance")

        return forecast_results

    def _get_recent_product_data(self, product: str, days: int = 60) -> pd.DataFrame:
        """Get recent data for a product (mock - replace with DB query)"""
        # Mock data
        dates = pd.date_range(end=datetime.now(), periods=days)
        data = pd.DataFrame({
            'date': dates,
            'product': product,
            'quantity': np.random.poisson(100, days) + np.sin(dates.dayofyear / 365 * 2 * np.pi) * 20,
            'location': 'default_location'  # Add location for consistency
        })
        return data

    def _generate_mock_training_data(self) -> pd.DataFrame:
        """Generate mock training data for initial model training"""
        # Generate mock sales data
        dates = pd.date_range(start='2023-01-01', periods=365, freq='D')
        products = ['Product1', 'Product2']
        locations = ['WarehouseA', 'WarehouseB']
        
        data = []
        for date in dates:
            for product in products:
                for location in locations:
                    quantity = np.random.poisson(100) + np.sin(date.dayofyear / 365 * 2 * np.pi) * 20 + np.random.normal(0, 10)
                    data.append({
                        'date': date,
                        'product': product,
                        'location': location,
                        'quantity': max(0, quantity)
                    })
        
        df = pd.DataFrame(data)
        
        # Use the preprocessor to add features (no file paths needed for mock)
        # Manually add basic features for training
        df = df.sort_values(['product', 'date'])
        
        # Basic temporal features
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['week_of_year'] = df['date'].dt.isocalendar().week
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
        df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
        df['is_holiday'] = 0  # Simplified for mock

        # Lag features
        for lag in [1, 7, 14, 30]:
            df[f'lag_{lag}'] = df.groupby('product')['quantity'].shift(lag)

        # Rolling statistics
        for window in [7, 14, 30]:
            df[f'rolling_mean_{window}'] = df.groupby('product')['quantity'].rolling(window=window, min_periods=1).mean().reset_index(0, drop=True)
            df[f'rolling_std_{window}'] = df.groupby('product')['quantity'].rolling(window=window, min_periods=1).std().reset_index(0, drop=True)

        # Mock external factors
        economic_data = self.fetch_economic_data()
        df['gdp_growth'] = economic_data.gdp_growth
        df['inflation_rate'] = economic_data.inflation_rate
        df['unemployment_rate'] = economic_data.unemployment_rate
        df['fuel_price'] = economic_data.fuel_price
        df['exchange_rate'] = economic_data.exchange_rate
        df['weather_severity'] = economic_data.weather_severity

        market_data = MarketIntelligence()
        df['competitor_price_index'] = market_data.competitor_price_index
        df['market_demand_index'] = market_data.market_demand_index
        df['economic_sentiment'] = market_data.economic_sentiment
        df['consumer_confidence'] = market_data.consumer_confidence
        df['promotional_activity'] = market_data.promotional_activity
        df['seasonal_trend'] = market_data.seasonal_trend

        # Mock weather
        df['temperature'] = 20.0 + np.sin(df['date'].dt.dayofyear / 365 * 2 * np.pi) * 10
        df['humidity'] = 50.0 + np.random.normal(0, 10, len(df))
        df['precipitation'] = np.random.uniform(0, 5, len(df))

        # Fill NaNs
        feature_cols = [col for col in df.columns if col.startswith('lag_') or col.startswith('rolling_')]
        for col in feature_cols:
            df[col] = df[col].ffill().fillna(0)

        return df

    def retrain_models_automated(self, product: str, data_path: str, drift_threshold: float = 0.05) -> Dict[str, Any]:
        """Automated retraining with drift detection"""
        # Load new data
        new_data = pd.read_csv(data_path)
        new_data['date'] = pd.to_datetime(new_data['date'])

        # Detect drift (simplified: compare recent R² with saved)
        recent_data = new_data[new_data['product'] == product].tail(30)
        if len(recent_data) < 10:
            return {'status': 'insufficient_data', 'retrained': False}

        # Mock drift calculation - compare prediction error on recent data
        X_recent = recent_data.drop(columns=['quantity', 'date', 'product', 'location'])
        feature_names = X_recent.columns.tolist()
        y_recent = recent_data['quantity']
        X_recent_scaled = self.scaler.transform(X_recent.reindex(columns=feature_names, fill_value=0))
        predictions = np.mean([model.predict(X_recent_scaled) for model in self.models.values()], axis=0)
        current_drift = np.mean(np.abs(y_recent - predictions)) / np.mean(y_recent) if np.mean(y_recent) > 0 else 0

        if current_drift < drift_threshold:
            return {'status': 'no_drift', 'drift': current_drift, 'retrained': False}

        # Retrain
        full_data = self.load_and_preprocess_data(data_path)
        product_data = full_data[full_data['product'] == product]
        training_result = self.train_advanced_models(product_data)

        return {
            'status': 'retrained',
            'drift': current_drift,
            'new_cv_score': np.mean(list(training_result['cv_scores'].values())),
            'retrained': True,
            'metrics': training_result['validation_metrics']
        }

    def get_forecast_summary(self, product: str, forecast_horizon: int = 30, country: str = 'US') -> Dict[str, Any]:
        """Get comprehensive forecast summary with risk analysis and validation metrics"""
        forecast = self.predict_demand_with_confidence(product, forecast_horizon, country)

        demands = [item['predicted_demand'] for item in forecast]
        confidences = [item['confidence_score'] for item in forecast]
        risks = [item['risk_factors']['overall_risk'] for item in forecast]

        # Calculate advanced metrics
        demand_trend = float(np.polyfit(range(len(demands)), demands, 1)[0])
        avg_demand = float(np.mean(demands)) if np.mean(demands) > 0 else 0.0
        demand_volatility = float(np.std(demands) / avg_demand) if avg_demand > 0 else 0.0
        avg_conf = float(np.mean(confidences))
        avg_risk = float(np.mean(risks))

        summary = {
            'product': product,
            'forecast_horizon': forecast_horizon,
            'total_predicted_demand': int(sum(demands)),
            'average_daily_demand': avg_demand,
            'demand_trend': demand_trend,
            'demand_volatility': demand_volatility,
            'average_confidence': avg_conf,
            'average_risk': avg_risk,
            'max_demand': max(demands),
            'min_demand': min(demands),
            'forecast_details': forecast,
            'recommendations': self._generate_forecast_recommendations(forecast, demand_trend, demand_volatility),
            'validation_metrics': self.validation_metrics  # Include per-model metrics
        }

        # Load ensemble weights for model performance visualization
        ensemble_weights_path = os.path.join(self.model_dir, 'ensemble_weights.pkl')
        if os.path.exists(ensemble_weights_path):
            ensemble_weights = joblib.load(ensemble_weights_path)
            summary['ensemble_weights'] = {name: float(weight * 100) for name, weight in ensemble_weights.items()}  # As percentages
        else:
            summary['ensemble_weights'] = {}

        return summary

    def _generate_forecast_recommendations(self, forecast: List[Dict], trend: float, volatility: float) -> List[str]:
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
