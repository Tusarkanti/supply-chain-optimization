import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
from datetime import datetime, timedelta

class EnhancedDetectionModule:
    def __init__(self):
        self.model = None
        self.model_path = 'models/enhanced_detection_model.pkl'
        self.anomaly_threshold = 0.7

    def load_data(self, data_path):
        """Load and preprocess sales data"""
        df = pd.read_csv(data_path)
        df['date'] = pd.to_datetime(df['date'])

        # Feature engineering
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

        # Calculate rolling statistics
        df['rolling_mean_7'] = df.groupby('product')['quantity'].rolling(window=7, min_periods=1).mean().reset_index(0, drop=True)
        df['rolling_std_7'] = df.groupby('product')['quantity'].rolling(window=7, min_periods=1).std().reset_index(0, drop=True)

        # Create lag features
        for lag in [1, 7, 14, 30]:
            df[f'lag_{lag}'] = df.groupby('product')['quantity'].shift(lag)

        # Fill NaN values
        df = df.fillna(method='bfill').fillna(method='ffill')

        return df

    def train_model(self, data_path):
        """Train the enhanced detection model"""
        df = self.load_data(data_path)

        # Prepare features and target
        feature_columns = ['day_of_week', 'month', 'quarter', 'is_weekend',
                          'rolling_mean_7', 'rolling_std_7', 'lag_1', 'lag_7', 'lag_14', 'lag_30']

        X = df[feature_columns]
        y = df['quantity']

        # Create binary target for anomaly detection (above normal range)
        df['mean_quantity'] = df.groupby('product')['quantity'].transform('mean')
        df['std_quantity'] = df.groupby('product')['quantity'].transform('std')
        df['is_anomaly'] = ((df['quantity'] - df['mean_quantity']) / df['std_quantity']).abs() > 2

        X_train, X_test, y_train, y_test = train_test_split(X, df['is_anomaly'], test_size=0.2, random_state=42)

        # Train Random Forest Classifier
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)

        # Evaluate model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy: {accuracy}")
        print(classification_report(y_test, y_pred))

        # Save model
        os.makedirs('models', exist_ok=True)
        joblib.dump(self.model, self.model_path)

        return self.model

    def predict_anomalies(self, new_data):
        """Predict anomalies in new data"""
        if self.model is None:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
            else:
                raise ValueError("Model not trained or loaded")

        # Preprocess new data
        df = self.load_data(new_data)

        feature_columns = ['day_of_week', 'month', 'quarter', 'is_weekend',
                          'rolling_mean_7', 'rolling_std_7', 'lag_1', 'lag_7', 'lag_14', 'lag_30']

        X = df[feature_columns]
        predictions = self.model.predict_proba(X)[:, 1]  # Probability of being anomaly

        df['anomaly_score'] = predictions
        df['is_anomaly'] = predictions > self.anomaly_threshold

        return df

    def get_anomaly_report(self, data_path):
        """Generate comprehensive anomaly report"""
        df = self.predict_anomalies(data_path)

        anomalies = df[df['is_anomaly']]

        report = {
            'total_records': len(df),
            'total_anomalies': len(anomalies),
            'anomaly_rate': len(anomalies) / len(df) if len(df) > 0 else 0,
            'top_anomalies': anomalies.nlargest(10, 'anomaly_score')[['product', 'date', 'quantity', 'anomaly_score']].to_dict('records'),
            'anomalies_by_product': anomalies.groupby('product').size().to_dict(),
            'anomalies_by_location': anomalies.groupby('location').size().to_dict()
        }

        return report
