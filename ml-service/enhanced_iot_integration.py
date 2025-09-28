import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import json
import asyncio
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os
from collections import defaultdict
import threading

logger = logging.getLogger(__name__)

@dataclass
class SensorData:
    sensor_id: str
    device_id: str
    timestamp: datetime
    location: Dict[str, float]
    sensor_type: str
    value: float
    unit: str
    quality_score: float
    battery_level: float
    signal_strength: float

@dataclass
class PredictiveMaintenanceAlert:
    device_id: str
    component: str
    failure_probability: float
    expected_failure_date: datetime
    recommended_action: str
    urgency_level: str  # 'low', 'medium', 'high', 'critical'
    maintenance_cost_estimate: float
    downtime_estimate_hours: float

@dataclass
class SupplyChainEvent:
    event_id: str
    event_type: str  # 'delay', 'quality_issue', 'temperature_exceedance', 'maintenance_needed'
    location: Dict[str, float]
    timestamp: datetime
    severity: str
    description: str
    affected_assets: List[str]
    estimated_impact: Dict[str, Any]
    recommended_actions: List[str]

@dataclass
class RealTimeDashboard:
    total_sensors: int
    active_sensors: int
    alerts_count: int
    critical_alerts: int
    system_health_score: float
    last_updated: datetime
    sensor_health: Dict[str, float]
    recent_events: List[SupplyChainEvent]

class EnhancedIoTIntegrationModule:
    def __init__(self):
        self.sensor_data_buffer = defaultdict(list)
        self.maintenance_models = {}
        self.anomaly_detectors = {}
        self.alert_thresholds = {
            'temperature': {'min': -10, 'max': 35, 'critical_min': -20, 'critical_max': 40},
            'humidity': {'min': 20, 'max': 80, 'critical_min': 10, 'critical_max': 90},
            'vibration': {'min': 0, 'max': 10, 'critical_min': 0, 'critical_max': 20},
            'pressure': {'min': 0.8, 'max': 1.2, 'critical_min': 0.5, 'critical_max': 1.5}
        }
        self.model_dir = 'models/enhanced_iot'
        os.makedirs(self.model_dir, exist_ok=True)

    def process_sensor_data_stream(self, sensor_data: Dict[str, Any]) -> SensorData:
        """Process incoming sensor data and validate quality"""
        try:
            # Parse and validate sensor data
            sensor = SensorData(
                sensor_id=sensor_data['sensor_id'],
                device_id=sensor_data['device_id'],
                timestamp=datetime.fromisoformat(sensor_data['timestamp']),
                location=sensor_data.get('location', {'latitude': 0, 'longitude': 0}),
                sensor_type=sensor_data['sensor_type'],
                value=float(sensor_data['value']),
                unit=sensor_data.get('unit', 'unknown'),
                quality_score=self._calculate_data_quality_score(sensor_data),
                battery_level=sensor_data.get('battery_level', 100.0),
                signal_strength=sensor_data.get('signal_strength', 100.0)
            )

            # Store in buffer for analysis
            self.sensor_data_buffer[sensor.device_id].append(sensor)

            # Keep only recent data (last 24 hours)
            cutoff_time = datetime.now() - timedelta(hours=24)
            self.sensor_data_buffer[sensor.device_id] = [
                s for s in self.sensor_data_buffer[sensor.device_id]
                if s.timestamp > cutoff_time
            ]

            return sensor

        except Exception as e:
            logger.error(f"Error processing sensor data: {e}")
            raise ValueError(f"Invalid sensor data format: {e}")

    def _calculate_data_quality_score(self, sensor_data: Dict[str, Any]) -> float:
        """Calculate quality score for sensor data"""
        score = 1.0

        # Check for missing required fields
        required_fields = ['sensor_id', 'device_id', 'timestamp', 'sensor_type', 'value']
        missing_fields = [field for field in required_fields if field not in sensor_data]
        score -= len(missing_fields) * 0.2

        # Check data freshness
        if 'timestamp' in sensor_data:
            try:
                data_age = (datetime.now() - datetime.fromisoformat(sensor_data['timestamp'])).seconds
                if data_age > 3600:  # Older than 1 hour
                    score -= 0.3
                elif data_age > 1800:  # Older than 30 minutes
                    score -= 0.1
            except:
                score -= 0.2

        # Check battery level
        if 'battery_level' in sensor_data:
            battery = sensor_data['battery_level']
            if battery < 20:
                score -= 0.3
            elif battery < 50:
                score -= 0.1

        # Check signal strength
        if 'signal_strength' in sensor_data:
            signal = sensor_data['signal_strength']
            if signal < 30:
                score -= 0.3
            elif signal < 70:
                score -= 0.1

        return max(0.0, min(1.0, score))

    def detect_anomalies(self, device_id: str, sensor_type: str) -> Dict[str, Any]:
        """Detect anomalies in sensor data using machine learning"""
        sensor_data = self.sensor_data_buffer[device_id]

        if len(sensor_data) < 50:  # Need minimum data for reliable detection
            return {
                'anomaly_detected': False,
                'confidence': 0.0,
                'reason': 'Insufficient data for anomaly detection'
            }

        # Filter data for specific sensor type
        relevant_data = [s for s in sensor_data if s.sensor_type == sensor_type]

        if len(relevant_data) < 30:
            return {
                'anomaly_detected': False,
                'confidence': 0.0,
                'reason': 'Insufficient sensor-specific data'
            }

        # Prepare data for anomaly detection
        values = np.array([s.value for s in relevant_data]).reshape(-1, 1)
        timestamps = np.array([(s.timestamp - relevant_data[0].timestamp).total_seconds() for s in relevant_data]).reshape(-1, 1)

        # Create feature matrix
        X = np.hstack([values, timestamps])

        # Train or load anomaly detector
        model_key = f"{device_id}_{sensor_type}"
        if model_key not in self.anomaly_detectors:
            self.anomaly_detectors[model_key] = self._train_anomaly_detector(X)

        detector = self.anomaly_detectors[model_key]

        # Detect anomalies on recent data
        recent_data = X[-10:]  # Last 10 readings
        anomaly_scores = detector.decision_function(recent_data)
        predictions = detector.predict(recent_data)

        # Check if any recent readings are anomalous
        anomaly_detected = any(predictions == -1)
        max_anomaly_score = min(anomaly_scores) if len(anomaly_scores) > 0 else 0

        return {
            'anomaly_detected': anomaly_detected,
            'confidence': abs(max_anomaly_score),
            'anomaly_score': max_anomaly_score,
            'recent_values': [s.value for s in relevant_data[-10:]],
            'threshold': detector.threshold_ if hasattr(detector, 'threshold_') else -0.5
        }

    def _train_anomaly_detector(self, data: np.ndarray) -> IsolationForest:
        """Train an anomaly detection model"""
        # Remove outliers for training
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)

        # Train Isolation Forest
        detector = IsolationForest(
            contamination=0.1,  # Expected proportion of outliers
            random_state=42,
            n_estimators=100
        )
        detector.fit(data_scaled)

        return detector

    def predict_maintenance_needs(self, device_id: str) -> List[PredictiveMaintenanceAlert]:
        """Predict maintenance needs based on sensor data patterns"""
        alerts = []

        # Get sensor data for the device
        device_sensors = [s for s in self.sensor_data_buffer[device_id]]

        if len(device_sensors) < 100:
            return alerts

        # Group sensors by type
        sensor_types = set(s.sensor_type for s in device_sensors)

        for sensor_type in sensor_types:
            # Get maintenance prediction for this sensor type
            prediction = self._predict_component_maintenance(device_id, sensor_type, device_sensors)

            if prediction and prediction['failure_probability'] > 0.3:  # 30% threshold
                alerts.append(PredictiveMaintenanceAlert(
                    device_id=device_id,
                    component=f"{sensor_type}_sensor",
                    failure_probability=prediction['failure_probability'],
                    expected_failure_date=prediction['expected_failure_date'],
                    recommended_action=prediction['recommended_action'],
                    urgency_level=prediction['urgency_level'],
                    maintenance_cost_estimate=prediction['maintenance_cost'],
                    downtime_estimate_hours=prediction['downtime_estimate']
                ))

        return alerts

    def _predict_component_maintenance(self, device_id: str, sensor_type: str,
                                     sensor_data: List[SensorData]) -> Dict[str, Any]:
        """Predict maintenance needs for a specific component"""
        # Filter data for this sensor type
        relevant_data = [s for s in sensor_data if s.sensor_type == sensor_type]

        if len(relevant_data) < 50:
            return None

        # Extract features for prediction
        values = np.array([s.value for s in relevant_data])
        timestamps = np.array([(s.timestamp - relevant_data[0].timestamp).total_seconds() for s in relevant_data])

        # Calculate degradation indicators
        degradation_rate = self._calculate_degradation_rate(values, timestamps)
        current_value_trend = self._calculate_trend(values[-20:])  # Last 20 readings
        variability_increase = self._calculate_variability_trend(values)

        # Calculate failure probability
        base_probability = 0.1  # Base 10% probability

        # Increase probability based on degradation indicators
        if degradation_rate > 0.01:  # Significant degradation
            base_probability += 0.3
        if abs(current_value_trend) > 0.1:  # Rapid change
            base_probability += 0.2
        if variability_increase > 0.05:  # Increasing variability
            base_probability += 0.2

        # Cap at 95%
        failure_probability = min(0.95, base_probability)

        if failure_probability > 0.3:
            # Estimate failure date
            days_until_failure = max(1, int(30 / failure_probability))  # More likely = sooner
            expected_failure_date = datetime.now() + timedelta(days=days_until_failure)

            # Determine urgency level
            if failure_probability > 0.8:
                urgency = 'critical'
            elif failure_probability > 0.6:
                urgency = 'high'
            elif failure_probability > 0.4:
                urgency = 'medium'
            else:
                urgency = 'low'

            # Estimate costs
            maintenance_cost = 500 if urgency == 'low' else 1000 if urgency == 'medium' else 2000
            downtime_estimate = 2 if urgency == 'low' else 4 if urgency == 'medium' else 8

            # Recommend action
            if urgency == 'critical':
                action = "Immediate maintenance required - schedule within 24 hours"
            elif urgency == 'high':
                action = "Schedule maintenance within 1 week"
            elif urgency == 'medium':
                action = "Plan maintenance within 2 weeks"
            else:
                action = "Monitor closely and schedule maintenance within 1 month"

            return {
                'failure_probability': failure_probability,
                'expected_failure_date': expected_failure_date,
                'recommended_action': action,
                'urgency_level': urgency,
                'maintenance_cost': maintenance_cost,
                'downtime_estimate': downtime_estimate,
                'degradation_rate': degradation_rate,
                'trend': current_value_trend
            }

        return None

    def _calculate_degradation_rate(self, values: np.ndarray, timestamps: np.ndarray) -> float:
        """Calculate rate of degradation over time"""
        if len(values) < 10:
            return 0.0

        # Linear regression to find trend
        from scipy import stats
        slope, _, _, _, _ = stats.linregress(timestamps, values)

        return abs(slope)

    def _calculate_trend(self, values: np.ndarray) -> float:
        """Calculate trend in recent values"""
        if len(values) < 5:
            return 0.0

        # Simple linear trend
        x = np.arange(len(values))
        slope, _, _, _, _ = stats.linregress(x, values)

        return slope

    def _calculate_variability_trend(self, values: np.ndarray) -> float:
        """Calculate if variability is increasing"""
        if len(values) < 20:
            return 0.0

        # Split into two halves
        mid = len(values) // 2
        first_half = values[:mid]
        second_half = values[mid:]

        first_var = np.var(first_half)
        second_var = np.var(second_half)

        if first_var == 0:
            return 0.0

        return (second_var - first_var) / first_var

    def generate_supply_chain_events(self, device_id: str) -> List[SupplyChainEvent]:
        """Generate supply chain events based on sensor data analysis"""
        events = []

        # Check for temperature exceedances
        temp_sensors = [s for s in self.sensor_data_buffer[device_id] if s.sensor_type == 'temperature']
        if temp_sensors:
            recent_temps = [s.value for s in temp_sensors[-10:]]  # Last 10 readings
            max_temp = max(recent_temps) if recent_temps else 0
            min_temp = min(recent_temps) if recent_temps else 0

            thresholds = self.alert_thresholds['temperature']

            if max_temp > thresholds['critical_max']:
                events.append(SupplyChainEvent(
                    event_id=f"temp_exceedance_{device_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    event_type='temperature_exceedance',
                    location=temp_sensors[0].location,
                    timestamp=datetime.now(),
                    severity='critical',
                    description=f"Temperature exceeded critical threshold: {max_temp}°C",
                    affected_assets=[device_id],
                    estimated_impact={'quality_risk': 'high', 'spoilage_risk': 'critical'},
                    recommended_actions=['Immediate inspection required', 'Check cooling systems', 'Isolate affected inventory']
                ))

            elif max_temp > thresholds['max']:
                events.append(SupplyChainEvent(
                    event_id=f"temp_warning_{device_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    event_type='temperature_exceedance',
                    location=temp_sensors[0].location,
                    timestamp=datetime.now(),
                    severity='high',
                    description=f"Temperature above normal range: {max_temp}°C",
                    affected_assets=[device_id],
                    estimated_impact={'quality_risk': 'medium', 'spoilage_risk': 'low'},
                    recommended_actions=['Monitor temperature closely', 'Check ventilation systems']
                ))

        # Check for humidity issues
        humidity_sensors = [s for s in self.sensor_data_buffer[device_id] if s.sensor_type == 'humidity']
        if humidity_sensors:
            recent_humidity = [s.value for s in humidity_sensors[-10:]]
            max_humidity = max(recent_humidity) if recent_humidity else 0

            thresholds = self.alert_thresholds['humidity']

            if max_humidity > thresholds['critical_max']:
                events.append(SupplyChainEvent(
                    event_id=f"humidity_exceedance_{device_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    event_type='humidity_exceedance',
                    location=humidity_sensors[0].location,
                    timestamp=datetime.now(),
                    severity='critical',
                    description=f"Humidity exceeded critical threshold: {max_humidity}%",
                    affected_assets=[device_id],
                    estimated_impact={'mold_risk': 'critical', 'quality_risk': 'high'},
                    recommended_actions=['Immediate dehumidification required', 'Check sealing', 'Inspect for water damage']
                ))

        # Check for vibration anomalies
        vibration_sensors = [s for s in self.sensor_data_buffer[device_id] if s.sensor_type == 'vibration']
        if vibration_sensors:
            anomaly_result = self.detect_anomalies(device_id, 'vibration')

            if anomaly_result['anomaly_detected']:
                events.append(SupplyChainEvent(
                    event_id=f"vibration_anomaly_{device_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    event_type='equipment_anomaly',
                    location=vibration_sensors[0].location,
                    timestamp=datetime.now(),
                    severity='medium',
                    description=f"Unusual vibration pattern detected: {anomaly_result['anomaly_score']:.3f}",
                    affected_assets=[device_id],
                    estimated_impact={'maintenance_needed': 'medium', 'downtime_risk': 'low'},
                    recommended_actions=['Schedule vibration analysis', 'Check for loose components', 'Monitor equipment performance']
                ))

        return events

    def generate_real_time_dashboard(self) -> RealTimeDashboard:
        """Generate real-time dashboard with system status and alerts"""
        total_sensors = sum(len(sensors) for sensors in self.sensor_data_buffer.values())
        active_sensors = sum(
            len([s for s in sensors if (datetime.now() - s.timestamp).seconds < 3600])
            for sensors in self.sensor_data_buffer.values()
        )

        # Calculate system health score
        recent_data_points = sum(len(sensors) for sensors in self.sensor_data_buffer.values())
        data_quality_scores = [
            s.quality_score for sensors in self.sensor_data_buffer.values()
            for s in sensors[-10:]  # Last 10 readings per device
        ]

        avg_data_quality = np.mean(data_quality_scores) if data_quality_scores else 1.0
        data_freshness = active_sensors / max(total_sensors, 1)

        system_health_score = (avg_data_quality * 0.6) + (data_freshness * 0.4)

        # Get recent events
        recent_events = []
        for device_sensors in self.sensor_data_buffer.values():
            for device_id in [s.device_id for s in device_sensors]:
                events = self.generate_supply_chain_events(device_id)
                recent_events.extend(events[-5:])  # Last 5 events per device

        # Sort by timestamp and take most recent
        recent_events.sort(key=lambda x: x.timestamp, reverse=True)
        recent_events = recent_events[:20]  # Top 20 most recent

        # Calculate sensor health
        sensor_health = {}
        for device_id, sensors in self.sensor_data_buffer.items():
            if sensors:
                recent_sensor = sensors[-1]
                health_score = (
                    recent_sensor.quality_score * 0.4 +
                    (recent_sensor.battery_level / 100) * 0.3 +
                    (recent_sensor.signal_strength / 100) * 0.3
                )
                sensor_health[device_id] = health_score

        return RealTimeDashboard(
            total_sensors=total_sensors,
            active_sensors=active_sensors,
            alerts_count=len(recent_events),
            critical_alerts=len([e for e in recent_events if e.severity == 'critical']),
            system_health_score=system_health_score,
            last_updated=datetime.now(),
            sensor_health=sensor_health,
            recent_events=recent_events
        )

    def get_device_health_report(self, device_id: str) -> Dict[str, Any]:
        """Generate comprehensive health report for a specific device"""
        if device_id not in self.sensor_data_buffer:
            return {'error': 'Device not found'}

        sensors = self.sensor_data_buffer[device_id]

        # Basic device info
        device_info = {
            'device_id': device_id,
            'total_sensors': len(set(s.sensor_type for s in sensors)),
            'data_points': len(sensors),
            'last_update': max(s.timestamp for s in sensors) if sensors else None,
            'sensor_types': list(set(s.sensor_type for s in sensors))
        }

        # Sensor health by type
        sensor_health_by_type = {}
        for sensor_type in device_info['sensor_types']:
            type_sensors = [s for s in sensors if s.sensor_type == sensor_type]
            if type_sensors:
                avg_quality = np.mean([s.quality_score for s in type_sensors])
                avg_battery = np.mean([s.battery_level for s in type_sensors])
                avg_signal = np.mean([s.signal_strength for s in type_sensors])

                sensor_health_by_type[sensor_type] = {
                    'count': len(type_sensors),
                    'avg_quality': avg_quality,
                    'avg_battery': avg_battery,
                    'avg_signal': avg_signal,
                    'last_reading': type_sensors[-1].value,
                    'unit': type_sensors[-1].unit
                }

        # Anomaly detection results
        anomaly_results = {}
        for sensor_type in device_info['sensor_types']:
            anomaly_results[sensor_type] = self.detect_anomalies(device_id, sensor_type)

        # Maintenance predictions
        maintenance_alerts = self.predict_maintenance_needs(device_id)

        # Recent events
        recent_events = self.generate_supply_chain_events(device_id)

        return {
            'device_info': device_info,
            'sensor_health_by_type': sensor_health_by_type,
            'anomaly_results': anomaly_results,
            'maintenance_alerts': [
                {
                    'component': alert.component,
                    'failure_probability': alert.failure_probability,
                    'expected_failure_date': alert.expected_failure_date.isoformat(),
                    'recommended_action': alert.recommended_action,
                    'urgency_level': alert.urgency_level,
                    'maintenance_cost_estimate': alert.maintenance_cost_estimate,
                    'downtime_estimate_hours': alert.downtime_estimate_hours
                } for alert in maintenance_alerts
            ],
            'recent_events': [
                {
                    'event_type': event.event_type,
                    'severity': event.severity,
                    'description': event.description,
                    'timestamp': event.timestamp.isoformat(),
                    'estimated_impact': event.estimated_impact
                } for event in recent_events
            ]
        }

    def simulate_sensor_data_stream(self, device_configs: List[Dict[str, Any]]) -> None:
        """Simulate sensor data stream for testing and demonstration"""
        def generate_sensor_data():
            while True:
                for config in device_configs:
                    # Generate realistic sensor data based on configuration
                    sensor_data = {
                        'sensor_id': config['sensor_id'],
                        'device_id': config['device_id'],
                        'timestamp': datetime.now().isoformat(),
                        'sensor_type': config['sensor_type'],
                        'value': self._generate_realistic_value(config),
                        'unit': config.get('unit', 'unknown'),
                        'location': config.get('location', {'latitude': 0, 'longitude': 0}),
                        'battery_level': max(0, 100 - np.random.exponential(0.1)),
                        'signal_strength': min(100, 80 + np.random.normal(0, 10))
                    }

                    try:
                        self.process_sensor_data_stream(sensor_data)
                    except Exception as e:
                        logger.warning(f"Error processing simulated data: {e}")

                # Wait before next batch
                import time
                time.sleep(1)  # Generate data every second

        # Start simulation in background thread
        thread = threading.Thread(target=generate_sensor_data, daemon=True)
        thread.start()

    def _generate_realistic_value(self, config: Dict[str, Any]) -> float:
        """Generate realistic sensor values based on configuration"""
        sensor_type = config['sensor_type']
        base_value = config.get('base_value', 20)
        variation = config.get('variation', 5)

        # Add realistic patterns
        hour = datetime.now().hour

        if sensor_type == 'temperature':
            # Temperature varies with time of day
            time_factor = 5 * np.sin(2 * np.pi * hour / 24)
            return base_value + time_factor + np.random.normal(0, variation)

        elif sensor_type == 'humidity':
            # Humidity often correlates with temperature
            temp_factor = (hour - 12) * 2  # Higher in middle of day
            return max(0, min(100, base_value + temp_factor + np.random.normal(0, variation)))

        elif sensor_type == 'vibration':
            # Vibration might be higher during working hours
            work_hour_factor = 2 if 8 <= hour <= 18 else 0.5
            return max(0, work_hour_factor + np.random.exponential(variation))

        else:
            return base_value + np.random.normal(0, variation)
