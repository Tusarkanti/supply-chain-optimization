from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_jwt_extended import JWTManager, jwt_required, create_access_token, get_jwt_identity
import pandas as pd
import os
from datetime import datetime, timedelta
import json
import logging

# Import custom modules
from config import get_config, Config
from database_schema import db, Product, Warehouse, Inventory, Sale, Customer, Order, OrderItem, Shipment, InventoryMovement, DemandForecast, RouteOptimization, RouteStop
from advanced_logging import logger
from demand_forecasting import DemandForecastingModule
from inventory_management import InventoryManagementModule
from logistics_optimization import LogisticsOptimizationModule, RouteStop as RouteStopDataClass
from enhanced_detection import EnhancedDetectionModule

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(get_config())

# Initialize extensions
CORS(app)
db.init_app(app)
JWTManager(app)

# Initialize modules
demand_forecaster = DemandForecastingModule()
inventory_manager = InventoryManagementModule()
logistics_optimizer = LogisticsOptimizationModule()
anomaly_detector = EnhancedDetectionModule()

# Create database tables
with app.app_context():
    db.create_all()

@app.route('/')
def index():
    """Serve the main dashboard"""
    return render_template('index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'modules': {
            'demand_forecasting': 'active',
            'inventory_management': 'active',
            'logistics_optimization': 'active',
            'anomaly_detection': 'active'
        }
    })

@app.route('/api/login', methods=['POST'])
def login():
    """User authentication"""
    data = request.get_json()

    # Debug: Print what we received
    print("Received login data:", data)

    if not data or not data.get('username') or not data.get('password'):
        return jsonify({'error': 'Missing credentials'}), 400

    # Simple authentication (in production, use proper auth)
    if data['username'] == 'admin' and data['password'] == 'password':
        access_token = create_access_token(identity=data['username'], expires_delta=timedelta(hours=24))
        return jsonify({'access_token': access_token}), 200

    return jsonify({'error': 'Invalid credentials'}), 401

# Demand Forecasting Endpoints
@app.route('/api/forecast/train', methods=['POST'])
@jwt_required()
def train_forecast_models():
    """Train demand forecasting models"""
    try:
        data = request.get_json()
        sales_data_path = data.get('sales_data_path', 'data/sales.csv')
        weather_data_path = data.get('weather_data_path')

        logger.log_prediction('demand_forecast', {'data_path': sales_data_path})

        results = demand_forecaster.train_models(sales_data_path, weather_data_path)

        return jsonify({
            'success': True,
            'message': 'Models trained successfully',
            'results': results
        })

    except Exception as e:
        logger.log_error(e, 'forecast_training')
        return jsonify({'error': str(e)}), 500

@app.route('/api/forecast/predict', methods=['POST'])
@jwt_required()
def predict_demand():
    """Generate demand forecast"""
    try:
        data = request.get_json()
        product = data.get('product')
        forecast_horizon = data.get('forecast_horizon', 30)

        if not product:
            return jsonify({'error': 'Product is required'}), 400

        forecast = demand_forecaster.predict_demand(product, forecast_horizon)
        summary = demand_forecaster.get_forecast_summary(product, forecast_horizon)

        # Save forecast to database
        for item in forecast:
            db_forecast = DemandForecast(
                product_id=product,
                warehouse_id='default',  # Should come from request
                forecast_date=datetime.strptime(item['date'], '%Y-%m-%d').date(),
                predicted_demand=item['predicted_demand'],
                confidence_score=item['confidence_score'],
                model_used='RandomForest'  # Should come from model metadata
            )
            db.session.add(db_forecast)

        db.session.commit()

        return jsonify({
            'success': True,
            'forecast': forecast,
            'summary': summary
        })

    except Exception as e:
        logger.log_error(e, 'demand_prediction')
        return jsonify({'error': str(e)}), 500

# Inventory Management Endpoints
@app.route('/api/inventory/analyze', methods=['POST'])
@jwt_required()
def analyze_inventory():
    """Analyze inventory levels and generate recommendations"""
    try:
        data = request.get_json()
        inventory_data_path = data.get('inventory_data_path', 'data/inventory.csv')
        sales_data_path = data.get('sales_data_path', 'data/sales.csv')

        # Load data
        inventory_df = pd.read_csv(inventory_data_path)
        sales_df = pd.read_csv(sales_data_path)

        results = inventory_manager.analyze_inventory_levels(inventory_df, sales_df)
        report = inventory_manager.generate_inventory_report(results)

        return jsonify({
            'success': True,
            'analysis': results,
            'report': report
        })

    except Exception as e:
        logger.log_error(e, 'inventory_analysis')
        return jsonify({'error': str(e)}), 500

@app.route('/api/inventory/recommendations', methods=['GET'])
@jwt_required()
def get_inventory_recommendations():
    """Get current inventory recommendations"""
    try:
        # This would typically query the database for current recommendations
        # For now, return a placeholder
        return jsonify({
            'success': True,
            'recommendations': []
        })

    except Exception as e:
        logger.log_error(e, 'inventory_recommendations')
        return jsonify({'error': str(e)}), 500

# Logistics Optimization Endpoints
@app.route('/api/logistics/optimize', methods=['POST'])
@jwt_required()
def optimize_routes():
    """Optimize delivery routes"""
    try:
        data = request.get_json()

        # Convert data to RouteStop objects
        warehouse_locations = [
            RouteStopDataClass(
                location_id=wh['id'],
                latitude=wh['latitude'],
                longitude=wh['longitude'],
                demand=0
            ) for wh in data.get('warehouses', [])
        ]

        customer_locations = [
            RouteStopDataClass(
                location_id=cust['id'],
                latitude=cust['latitude'],
                longitude=cust['longitude'],
                demand=cust.get('demand', 0)
            ) for cust in data.get('customers', [])
        ]

        optimization_results = logistics_optimizer.optimize_routes(warehouse_locations, customer_locations)
        report = logistics_optimizer.generate_route_report(optimization_results)

        # Save optimization results to database
        if optimization_results['success']:
            for route in optimization_results['routes']:
                db_optimization = RouteOptimization(
                    optimization_date=datetime.now().date(),
                    total_distance=route.total_distance,
                    total_cost=route.total_cost,
                    vehicle_count=1,
                    optimization_status='completed'
                )
                db.session.add(db_optimization)

                for i, stop in enumerate(route.stops):
                    db_stop = RouteStop(
                        optimization_id=db_optimization.id,
                        warehouse_id=stop.location_id if stop.demand == 0 else 'customer',
                        customer_id=stop.location_id if stop.demand > 0 else None,
                        stop_sequence=i,
                        load_quantity=stop.demand
                    )
                    db.session.add(db_stop)

            db.session.commit()

        return jsonify({
            'success': optimization_results['success'],
            'results': optimization_results,
            'report': report
        })

    except Exception as e:
        logger.log_error(e, 'route_optimization')
        return jsonify({'error': str(e)}), 500

# Anomaly Detection Endpoints
@app.route('/api/anomaly/detect', methods=['POST'])
@jwt_required()
def detect_anomalies():
    """Detect anomalies in sales data"""
    try:
        data = request.get_json()
        data_path = data.get('data_path', 'data/sales.csv')

        anomalies = anomaly_detector.predict_anomalies(data_path)
        report = anomaly_detector.get_anomaly_report(data_path)

        return jsonify({
            'success': True,
            'anomalies': anomalies.to_dict('records'),
            'report': report
        })

    except Exception as e:
        logger.log_error(e, 'anomaly_detection')
        return jsonify({'error': str(e)}), 500

# Dashboard Data Endpoint
@app.route('/api/dashboard/data', methods=['GET'])
@jwt_required()
def get_dashboard_data():
    """Get comprehensive dashboard data"""
    try:
        # This would aggregate data from all modules
        dashboard_data = {
            'forecast_summary': {},
            'inventory_status': {},
            'logistics_metrics': {},
            'anomaly_alerts': []
        }

        return jsonify({
            'success': True,
            'data': dashboard_data
        })

    except Exception as e:
        logger.log_error(e, 'dashboard_data')
        return jsonify({'error': str(e)}), 500

# Error Handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Resource not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.log_error(error, 'internal_error')
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    logger.main_logger.info("Starting Supply Chain Optimization System")
    app.run(debug=app.config['DEBUG'], host='0.0.0.0', port=5000)
