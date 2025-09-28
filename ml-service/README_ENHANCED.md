# Enhanced Supply Chain Optimization System

## Overview

This enhanced supply chain optimization system provides a comprehensive, production-ready solution for managing complex supply chain operations with advanced AI/ML capabilities, real-time monitoring, and intelligent decision support.

## 🚀 Key Features

### 1. Enhanced Demand Forecasting
- **Multi-model ensemble** with Random Forest, Gradient Boosting, and Extra Trees
- **External factor integration** including economic indicators, market intelligence, and weather data
- **Confidence intervals** and risk assessment for forecast reliability
- **Holiday and seasonal pattern recognition** across multiple countries
- **Real-time forecast updates** with streaming data integration

### 2. Advanced Inventory Management
- **Multi-echelon optimization** with sophisticated safety stock calculations
- **Supplier risk assessment** with financial and operational metrics
- **Sustainability tracking** including carbon footprint and waste reduction
- **Monte Carlo simulation** for inventory scenario planning
- **Dynamic reorder point optimization** based on demand uncertainty

### 3. Intelligent Logistics Optimization
- **Real-time traffic integration** with live traffic data from multiple providers
- **Carbon footprint optimization** with route efficiency calculations
- **Regulatory compliance checking** for international shipping
- **Multi-constraint optimization** including time windows, capacity, and costs
- **Risk assessment** for route planning with traffic and compliance factors

### 4. IoT Integration and Predictive Maintenance
- **Real-time sensor data processing** with quality validation
- **Anomaly detection** using machine learning algorithms
- **Predictive maintenance alerts** with failure probability calculations
- **Supply chain event generation** from sensor data patterns
- **Real-time dashboard** with system health monitoring

## 📁 Project Structure

```
ml-service/
├── app.py                          # Main Flask application
├── config.py                       # Configuration management
├── database_schema.py              # Database models and schema
├── advanced_logging.py             # Enhanced logging system
├── enhanced_demand_forecasting.py  # Advanced demand forecasting
├── enhanced_inventory_management.py # Smart inventory optimization
├── enhanced_logistics_optimization.py # Intelligent route optimization
├── enhanced_iot_integration.py     # IoT and predictive maintenance
├── requirements.txt                # Core dependencies
├── enhanced_requirements.txt       # Additional dependencies for enhanced features
├── models/                         # Trained ML models
│   └── enhanced_forecasting/       # Demand forecasting models
│   └── enhanced_inventory/         # Inventory optimization models
│   └── enhanced_logistics/         # Route optimization models
│   └── enhanced_iot/               # IoT and anomaly detection models
└── logs/                           # Application logs
```

## 🛠 Installation and Setup

### Prerequisites
- Python 3.8+
- PostgreSQL database
- Redis for caching and message queuing
- Docker (optional, for containerized deployment)

### 1. Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install core dependencies
pip install -r requirements.txt

# Install enhanced dependencies (optional, for full functionality)
pip install -r enhanced_requirements.txt
```

### 2. Database Setup
```bash
# Set up PostgreSQL database
createdb supply_chain_db

# Run database migrations
python -c "from database_schema import init_db; init_db()"
```

### 3. Configuration
Update `config.py` with your settings:
```python
# Database configuration
SQLALCHEMY_DATABASE_URI = 'postgresql://user:password@localhost/supply_chain_db'

# API Keys for external services
WEATHER_API_KEY = 'your_weather_api_key'
TRAFFIC_API_KEY = 'your_traffic_api_key'
FINANCIAL_DATA_API_KEY = 'your_financial_api_key'

# Model configuration
MODEL_UPDATE_INTERVAL = 3600  # Update models every hour
PREDICTION_HORIZON = 30      # Days for forecasting
```

### 4. Start the Application
```bash
# Development mode
python app.py

# Production mode with Gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## 📊 API Endpoints

### Demand Forecasting
```http
POST /api/enhanced-forecast/train
Content-Type: application/json

{
  "data_path": "path/to/sales_data.csv",
  "weather_data_path": "path/to/weather_data.csv",
  "country": "US",
  "products": ["product1", "product2"]
}

GET /api/enhanced-forecast/predict/{product_id}
Parameters:
  - horizon: 30 (days)
  - country: "US"

GET /api/enhanced-forecast/summary/{product_id}
Parameters:
  - horizon: 30 (days)
  - country: "US"
```

### Inventory Management
```http
POST /api/enhanced-inventory/optimize
Content-Type: application/json

{
  "inventory_data": [...],
  "demand_data": [...],
  "service_level_target": 0.95
}

GET /api/enhanced-inventory/report/{analysis_id}

POST /api/enhanced-inventory/simulate
Content-Type: application/json

{
  "current_stock": 1000,
  "daily_demand": [50, 45, 60, ...],
  "lead_time": 7,
  "reorder_point": 300,
  "order_quantity": 500,
  "scenarios": 1000
}
```

### Logistics Optimization
```http
POST /api/enhanced-logistics/optimize
Content-Type: application/json

{
  "warehouse_locations": [...],
  "customer_locations": [...],
  "vehicle_types": ["truck_medium", "truck_large"],
  "constraints": {
    "max_vehicles": 10,
    "vehicle_capacity": 1000
  }
}

GET /api/enhanced-logistics/report/{optimization_id}

POST /api/enhanced-logistics/simulate-traffic
Content-Type: application/json

{
  "route_id": "route_123",
  "scenarios": 100
}
```

### IoT Integration
```http
POST /api/enhanced-iot/process-sensor-data
Content-Type: application/json

{
  "sensor_id": "sensor_001",
  "device_id": "device_001",
  "timestamp": "2024-01-01T10:00:00",
  "sensor_type": "temperature",
  "value": 23.5,
  "unit": "celsius",
  "location": {"latitude": 40.7128, "longitude": -74.0060}
}

GET /api/enhanced-iot/dashboard

GET /api/enhanced-iot/device-health/{device_id}

GET /api/enhanced-iot/anomalies/{device_id}
Parameters:
  - sensor_type: "temperature"
  - hours_back: 24
```

## 🔧 Advanced Configuration

### Model Training Configuration
```python
# In config.py
DEMAND_FORECAST_CONFIG = {
    'models': ['RandomForest', 'GradientBoosting', 'ExtraTrees'],
    'hyperparameter_tuning': True,
    'cross_validation_folds': 5,
    'feature_engineering': {
        'lag_features': [1, 7, 14, 30, 60, 90],
        'rolling_windows': [7, 14, 30, 60],
        'holiday_calendars': ['US', 'CA', 'UK', 'DE', 'FR']
    }
}

INVENTORY_CONFIG = {
    'service_level_target': 0.95,
    'lead_time_multiplier': 1.5,
    'safety_stock_method': 'advanced',  # 'basic' or 'advanced'
    'supplier_risk_weight': 0.3
}

LOGISTICS_CONFIG = {
    'traffic_providers': ['google', 'here', 'tomtom'],
    'optimization_algorithm': 'ortools',  # 'ortools' or 'custom'
    'carbon_tracking': True,
    'regulatory_compliance': True
}
```

### Monitoring and Alerting
```python
# Alert thresholds
ALERT_THRESHOLDS = {
    'demand_forecast_accuracy': 0.8,  # R² score threshold
    'inventory_service_level': 0.95,  # Service level threshold
    'logistics_delay': 30,            # Minutes delay threshold
    'iot_sensor_quality': 0.7,        # Sensor data quality threshold
    'system_health_score': 0.8        # Overall system health threshold
}

# Notification channels
NOTIFICATION_CONFIG = {
    'email': ['admin@company.com'],
    'slack': ['#supply-chain-alerts'],
    'sms': ['+1234567890'],
    'webhook': ['https://api.company.com/alerts']
}
```

## 📈 Performance Monitoring

### Key Metrics
- **Demand Forecast Accuracy**: R² score, MAE, RMSE
- **Inventory Optimization**: Service level achievement, stockout rate, carrying costs
- **Logistics Efficiency**: Route optimization ratio, fuel efficiency, delivery time
- **System Health**: Sensor uptime, data quality, prediction confidence

### Dashboard Metrics
```python
# Real-time metrics
REAL_TIME_METRICS = {
    'active_sensors': 'Number of active IoT sensors',
    'forecast_confidence': 'Average confidence score of predictions',
    'inventory_turnover': 'Inventory turnover rate',
    'route_efficiency': 'Average route optimization efficiency',
    'carbon_footprint': 'Total CO2 emissions tracking',
    'supplier_risk_score': 'Average supplier risk assessment'
}
```

## 🔒 Security Considerations

### Data Protection
- All sensitive data encrypted at rest and in transit
- API authentication using JWT tokens
- Role-based access control (RBAC)
- Audit logging for all operations

### Compliance
- GDPR compliance for personal data
- Industry-specific regulations (FDA, ISO, etc.)
- Data retention policies
- Regular security audits

## 🚀 Deployment Options

### Docker Deployment
```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

### Kubernetes Deployment
```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: supply-chain-optimization
spec:
  replicas: 3
  selector:
    matchLabels:
      app: supply-chain
  template:
    metadata:
      labels:
        app: supply-chain
    spec:
      containers:
      - name: ml-service
        image: supply-chain-ml:latest
        ports:
        - containerPort: 5000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: database-url
```

### Cloud Deployment
- **AWS**: ECS, EKS, or Lambda
- **Azure**: Container Instances or Kubernetes Service
- **GCP**: Cloud Run or Kubernetes Engine

## 🤝 Integration Examples

### ERP System Integration
```python
# SAP integration example
from sap_connector import SAPClient

def sync_inventory_with_sap():
    sap_client = SAPClient(api_key=SAP_API_KEY)

    # Get inventory recommendations
    recommendations = inventory_module.optimize_inventory(inventory_data)

    # Sync with SAP
    for rec in recommendations:
        sap_client.update_inventory(
            product_id=rec.product_id,
            warehouse_id=rec.warehouse_id,
            reorder_point=rec.reorder_point,
            safety_stock=rec.safety_stock
        )
```

### External API Integration
```python
# Weather API integration
import requests

def fetch_weather_data(location):
    api_key = config.WEATHER_API_KEY
    url = f"https://api.weatherapi.com/v1/forecast.json?key={api_key}&q={location}"

    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return {
            'temperature': data['current']['temp_c'],
            'humidity': data['current']['humidity'],
            'precipitation': data['current']['precip_mm'],
            'wind_speed': data['current']['wind_kph']
        }
    return None
```

## 📚 Testing and Validation

### Unit Tests
```bash
# Run unit tests
pytest tests/unit/

# Run integration tests
pytest tests/integration/

# Run performance tests
pytest tests/performance/
```

### Model Validation
```python
# Validate demand forecasting models
from sklearn.metrics import mean_absolute_error, r2_score

def validate_forecasting_model(model, X_test, y_test):
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    return {
        'mae': mae,
        'r2_score': r2,
        'accuracy': 'acceptable' if r2 > 0.8 else 'needs_improvement'
    }
```

## 🔄 Continuous Improvement

### Model Retraining
- Automated model retraining based on performance degradation
- A/B testing for new algorithms
- Ensemble model updates
- Hyperparameter optimization

### System Monitoring
- Real-time performance monitoring
- Anomaly detection in system behavior
- Predictive maintenance for infrastructure
- Automated scaling based on load

## 📞 Support and Maintenance

### Monitoring
- System health dashboards
- Alert management
- Performance analytics
- Usage reporting

### Maintenance
- Regular model updates
- Database optimization
- Security patches
- Feature enhancements

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📞 Contact

For questions, support, or contributions, please contact:
- Email: support@supplychain-ai.com
- Documentation: https://docs.supplychain-ai.com
- Community: https://community.supplychain-ai.com

---

**Built with ❤️ for the future of supply chain optimization**
