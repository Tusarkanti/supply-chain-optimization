# Intelligent Supply Chain Optimization System

A comprehensive, AI-powered supply chain optimization platform that integrates advanced machine learning models for demand forecasting, inventory management, logistics optimization, and anomaly detection.

## 🚀 Features

### 🔮 Demand Forecasting
- **Advanced ML Models**: Random Forest and Gradient Boosting algorithms
- **Feature Engineering**: Time-based features, lag variables, rolling statistics
- **Weather Integration**: Incorporate external weather data for better predictions
- **Confidence Intervals**: Probabilistic forecasting with uncertainty estimates

### 📦 Inventory Management
- **Smart Reordering**: EOQ-based optimization with safety stock calculations
- **Service Level Optimization**: Configurable service levels (95%, 99%, etc.)
- **Real-time Analysis**: Dynamic inventory level monitoring and recommendations
- **Cost Optimization**: Balance holding costs vs. stockout costs

### 🚚 Logistics Optimization
- **Route Optimization**: Google OR-Tools powered vehicle routing
- **Multi-constraint Solving**: Capacity, time windows, and distance optimization
- **Cost Analysis**: Comprehensive cost modeling and savings calculations
- **Scenario Simulation**: What-if analysis for different logistics scenarios

### 🔍 Anomaly Detection
- **Statistical Methods**: Z-score, IQR, and isolation forest algorithms
- **Real-time Monitoring**: Continuous anomaly detection in sales data
- **Alert System**: Automated alerting for unusual patterns
- **Root Cause Analysis**: Detailed reporting and analysis tools

## 🏗️ Architecture

### Backend (Python/Flask)
- **Framework**: Flask with SQLAlchemy ORM
- **ML Libraries**: scikit-learn, pandas, numpy
- **Optimization**: Google OR-Tools
- **Authentication**: JWT-based authentication
- **Database**: SQLite (production-ready for PostgreSQL/MySQL)

### Frontend (React)
- **Framework**: React 18 with modern hooks
- **Styling**: Custom CSS with responsive design
- **State Management**: React Context and useState
- **HTTP Client**: Fetch API with JWT authentication

### Key Components

#### Backend Modules
1. **Demand Forecasting Module** (`demand_forecasting.py`)
   - Time series analysis and prediction
   - Model training and evaluation
   - Forecast generation with confidence intervals

2. **Inventory Management Module** (`inventory_management.py`)
   - Safety stock calculations
   - Reorder point optimization
   - Economic order quantity (EOQ) computation

3. **Logistics Optimization Module** (`logistics_optimization.py`)
   - Vehicle routing problem solving
   - Multi-objective optimization
   - Route cost and distance calculation

4. **Anomaly Detection Module** (`enhanced_detection.py`)
   - Statistical anomaly detection
   - Machine learning-based outlier detection
   - Alert generation and reporting

#### Frontend Components
1. **Dashboard**: System overview with key metrics
2. **ForecastForm**: Demand forecasting interface
3. **InventoryTable**: Inventory management interface
4. **Login**: Authentication component

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- Node.js 16+
- pip (Python package manager)
- npm (Node package manager)

### Backend Setup

1. **Navigate to the ml-service directory:**
   ```bash
   cd ml-service
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Flask application:**
   ```bash
   python app.py
   ```

   The backend will start on `http://localhost:5000`

### Frontend Setup

1. **Navigate to the frontend directory:**
   ```bash
   cd frontend
   ```

2. **Install Node.js dependencies:**
   ```bash
   npm install
   ```

3. **Start the React development server:**
   ```bash
   npm start
   ```

   The frontend will start on `http://localhost:3000`

## 🔐 Authentication

The system uses JWT-based authentication:

- **Username**: `admin`
- **Password**: `password`

## 📊 API Endpoints

### Authentication
- `POST /api/login` - User authentication

### Health Check
- `GET /api/health` - System health status

### Demand Forecasting
- `POST /api/forecast/train` - Train forecasting models
- `POST /api/forecast/predict` - Generate demand forecasts

### Inventory Management
- `POST /api/inventory/analyze` - Analyze inventory and get recommendations
- `GET /api/inventory/recommendations` - Get current recommendations

### Logistics Optimization
- `POST /api/logistics/optimize` - Optimize delivery routes

### Anomaly Detection
- `POST /api/anomaly/detect` - Detect anomalies in data

### Dashboard
- `GET /api/dashboard/data` - Get comprehensive dashboard data

## 📁 Project Structure

```
supply-chain-optimization/
├── ml-service/                 # Backend Python application
│   ├── app.py                 # Main Flask application
│   ├── config.py              # Configuration management
│   ├── database_schema.py     # Database models and schema
│   ├── requirements.txt       # Python dependencies
│   ├── demand_forecasting.py  # Demand forecasting module
│   ├── inventory_management.py # Inventory management module
│   ├── logistics_optimization.py # Logistics optimization module
│   ├── enhanced_detection.py  # Anomaly detection module
│   └── advanced_logging.py    # Logging and monitoring
├── frontend/                  # React frontend application
│   ├── public/
│   │   └── index.html         # HTML template
│   ├── src/
│   │   ├── App.js             # Main React component
│   │   ├── index.js           # React entry point
│   │   ├── App.css            # Main application styles
│   │   └── components/        # React components
│   │       ├── Login.js       # Authentication component
│   │       ├── Dashboard.js   # Main dashboard
│   │       ├── ForecastForm.js # Forecasting interface
│   │       ├── InventoryTable.js # Inventory management
│   │       └── *.css          # Component styles
│   └── package.json           # Node.js dependencies
└── README.md                  # Project documentation
```

## 🎯 Usage Examples

### Training Demand Forecasting Models

```python
from demand_forecasting import DemandForecastingModule

forecaster = DemandForecastingModule()
results = forecaster.train_models('data/sales.csv', 'data/weather.csv')
print(f"Models trained: {results}")
```

### Generating Demand Forecasts

```python
forecast = forecaster.predict_demand('Product_A', forecast_horizon=30)
summary = forecaster.get_forecast_summary('Product_A', 30)
```

### Inventory Analysis

```python
from inventory_management import InventoryManagementModule

inventory_manager = InventoryManagementModule()
analysis = inventory_manager.analyze_inventory_levels(inventory_df, sales_df)
report = inventory_manager.generate_inventory_report(analysis)
```

### Route Optimization

```python
from logistics_optimization import LogisticsOptimizationModule

optimizer = LogisticsOptimizationModule()
results = optimizer.optimize_routes(warehouses, customers)
report = optimizer.generate_route_report(results)
```

## 🔧 Configuration

The system is configured through the `config.py` file:

```python
# Database configuration
SQLALCHEMY_DATABASE_URI = 'sqlite:///supply_chain.db'

# JWT configuration
JWT_SECRET_KEY = 'your-secret-key-here'

# Model configuration
MODEL_UPDATE_INTERVAL = 24  # hours
FORECAST_HORIZON = 30      # days

# Logging configuration
LOG_LEVEL = 'INFO'
LOG_FILE = 'logs/supply_chain.log'
```

## 📈 Performance Metrics

The system provides comprehensive performance tracking:

- **Forecast Accuracy**: MAE, RMSE, R² scores
- **Inventory Optimization**: Service level achievement, cost savings
- **Logistics Efficiency**: Distance reduction, cost savings
- **Anomaly Detection**: Precision, recall, F1-score

## 🔒 Security Features

- JWT-based authentication
- Password hashing
- CORS configuration
- Input validation and sanitization
- SQL injection protection via SQLAlchemy ORM

## 🚀 Deployment

### Production Deployment

1. **Backend Deployment:**
   - Use Gunicorn for production WSGI server
   - Configure production database (PostgreSQL/MySQL)
   - Set up environment variables
   - Configure logging and monitoring

2. **Frontend Deployment:**
   - Build production bundle: `npm run build`
   - Deploy to web server or CDN
   - Configure API proxy settings

3. **Docker Deployment:**
   - Use provided Dockerfiles for containerization
   - Configure Docker Compose for multi-service deployment

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Create an issue in the GitHub repository
- Check the documentation
- Review the API documentation

## 🔮 Future Enhancements

- [ ] Real-time data streaming
- [ ] Advanced ML models (Deep Learning)
- [ ] Mobile application
- [ ] Multi-tenant architecture
- [ ] Advanced analytics dashboard
- [ ] Integration with ERP systems
- [ ] Blockchain-based supply chain tracking

---

**Built with ❤️ for intelligent supply chain optimization**
