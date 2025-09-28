# ml-service/app.py (with added debug print for config URI)
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_jwt_extended import JWTManager, jwt_required, create_access_token, get_jwt_identity, verify_jwt_in_request
from flask_socketio import SocketIO, emit, ConnectionRefusedError
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import os
from datetime import datetime, timedelta
import logging
import re

# Import custom modules
from config import get_config
from database_schema import db, User, Product, Warehouse, Inventory, Sale, Customer, Order, OrderItem, Shipment, InventoryMovement, DemandForecast, RouteOptimization, RouteStop
from advanced_logging import logger

# Import ML modules with fallback
enhanced_available = False
try:
    from enhanced_demand_forecasting import EnhancedDemandForecastingModule
    from enhanced_inventory_management import EnhancedInventoryManagementModule
    from enhanced_logistics_optimization import EnhancedLogisticsOptimizationModule
    from enhanced_iot_integration import EnhancedIoTIntegrationModule
    enhanced_available = True
    ML_MODULES_AVAILABLE = True
    # Import RouteStop from basic module as fallback
    try:
        from logistics_optimization import RouteStop as RouteStopDataClass
    except ImportError:
        RouteStopDataClass = object
except ImportError as e:
    print(f"Enhanced ML modules not available: {e}")
    enhanced_available = False
    try:
        from demand_forecasting import EnhancedDemandForecastingModule
        from inventory_management import InventoryManagementModule as EnhancedInventoryManagementModule
        from logistics_optimization import LogisticsOptimizationModule as EnhancedLogisticsOptimizationModule, RouteStop as RouteStopDataClass
        from enhanced_detection import EnhancedDetectionModule as EnhancedIoTIntegrationModule
        print("Falling back to basic ML modules")
        ML_MODULES_AVAILABLE = True
    except ImportError as e2:
        print(f"Basic ML modules not available: {e2}")
        ML_MODULES_AVAILABLE = False

        class EnhancedDemandForecastingModule:
            def __init__(self):
                self.models = {}

            def train_models(self, *args, **kwargs): return {"status": "ML not available"}
            def predict_demand_with_confidence(self, *args, **kwargs): return {"status": "ML not available"}
            def retrain_models_automated(self, *args, **kwargs): return {"retrained": False, "status": "ML not available"}
            def load_and_preprocess_data(self, *args, **kwargs):
                try:
                    import pandas as pd
                    return pd.DataFrame()
                except ImportError:
                    return None
            def train_advanced_models(self, data, *args, **kwargs): return {"status": "ML not available", "models_trained": False}
            def get_forecast_summary(self, *args, **kwargs): return {"status": "ML not available"}

        class EnhancedInventoryManagementModule:
            def analyze_inventory_levels(self, *args, **kwargs): return {'recommendations': []}
            def generate_inventory_report(self, *args, **kwargs): return {"status": "ML not available"}

        class EnhancedLogisticsOptimizationModule:
            def optimize_routes(self, *args, **kwargs): return {"success": False, "message": "ML not available"}
            def generate_route_report(self, *args, **kwargs): return {"status": "ML not available"}

        class EnhancedIoTIntegrationModule:
            def predict_anomalies(self, *args, **kwargs): return []
            def get_anomaly_report(self, *args, **kwargs): return {"status": "ML not available"}

        RouteStopDataClass = object

# Initialize Flask app
app = Flask(__name__, instance_relative_config=True)
app.config.from_object(get_config())
print("App config URI:", app.config['SQLALCHEMY_DATABASE_URI'])
CORS(app)
db.init_app(app)
jwt = JWTManager(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
limiter = Limiter(app=app, key_func=get_remote_address, default_limits=["1000 per day", "200 per hour"])

# Initialize ML modules
demand_forecaster = EnhancedDemandForecastingModule()
inventory_manager = EnhancedInventoryManagementModule()
logistics_optimizer = EnhancedLogisticsOptimizationModule()
anomaly_detector = EnhancedIoTIntegrationModule()

# Train models if not already trained
if ML_MODULES_AVAILABLE and not demand_forecaster.models:
    try:
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(script_dir)
        sales_data_path = os.path.join(root_dir, 'data', 'sales.csv')
        weather_data_path = os.path.join(root_dir, 'data', 'weather.csv') if os.path.exists(os.path.join(root_dir, 'data', 'weather.csv')) else None
        if os.path.exists(sales_data_path):
            processed_data = demand_forecaster.load_and_preprocess_data(sales_data_path, weather_data_path)
            demand_forecaster.train_advanced_models(processed_data)
            logger.log_info("Models trained successfully on startup", 'model_training')
        else:
            logger.log_warning("Sales data file not found, skipping startup training", 'model_training')
    except Exception as e:
        logger.log_error(f"Startup training error: {str(e)}", 'model_training')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger.main_logger.info("Initializing Supply Chain Optimization System")

# -------------------- DATABASE SETUP --------------------
def seed_demo_data():
    """Seed basic demo users; full data via init_db.py"""
    try:
        if User.query.count() > 0:
            logger.log_info("Demo data already exists, skipping seeding", 'database')
            print("Demo data already exists; run python ml-service/init_db.py to reset and seed full data")
            return

        admin_user = User.query.filter_by(email='admin@supplychain.com').first()
        demo_user = User.query.filter_by(email='demo@supplychain.com').first()

        if admin_user and demo_user:
            return

        if not admin_user:
            admin = User(email='admin@supplychain.com', name='Admin User', email_verified=True, two_factor_enabled=False)
            admin.set_password('admin')
            db.session.add(admin)

        if not demo_user:
            demo = User(email='demo@supplychain.com', name='Demo User', email_verified=True, two_factor_enabled=True)
            demo.set_password('demo123')
            demo.generate_two_factor_secret()
            db.session.add(demo)

        db.session.commit()
        logger.log_info("Basic demo users seeded successfully", 'database')
        print("Basic demo users seeded; run python ml-service/init_db.py for full data")
    except Exception as e:
        logger.log_error(f"Error seeding demo data: {str(e)}", 'database')
        db.session.rollback()

with app.app_context():
    db.create_all()
    seed_demo_data()

# -------------------- UTILITY FUNCTIONS --------------------
def validate_password_strength(password):
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    if not re.search(r'[A-Z]', password):
        return False, "Password must contain at least one uppercase letter"
    if not re.search(r'[a-z]', password):
        return False, "Password must contain at least one lowercase letter"
    if not re.search(r'\d', password):
        return False, "Password must contain at least one number"
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        return False, "Password must contain at least one special character"
    return True, "Password is strong"

# -------------------- API ROUTES --------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'modules': {
            'demand_forecasting': 'active' if ML_MODULES_AVAILABLE else 'unavailable',
            'inventory_management': 'active' if ML_MODULES_AVAILABLE else 'unavailable',
            'logistics_optimization': 'active' if ML_MODULES_AVAILABLE else 'unavailable',
            'anomaly_detection': 'active' if ML_MODULES_AVAILABLE else 'unavailable'
        }
    })

@app.route('/api/login', methods=['POST'])
@limiter.limit("5 per minute")
def login():
    try:
        print("Login attempt started")
        data = request.get_json()
        print(f"Received data: {data}")
        if not data or not data.get('email') or not data.get('password'):
            print("Missing email or password")
            return jsonify({'error': 'Email and password are required'}), 400

        email = data['email'].strip().lower()
        password = data['password']
        print(f"Looking for user: {email}")
        user = User.query.filter_by(email=email).first()
        print(f"User found: {user is not None}")

        if not user:
            print("User not found")
            logger.log_error(f"Login attempt for non-existent user: {email}", 'authentication')
            return jsonify({'error': 'Invalid email or password'}), 401

        # Account locking removed - no lock check

        print("Checking password")
        print(f"Input password: {password}")
        print(f"Stored hash: {user.password_hash}")
        if not user.check_password(password):
            print("Password incorrect")
            logger.log_error(f"Failed login attempt for user: {email}", 'authentication')
            return jsonify({'error': 'Invalid email or password'}), 401

        print("Password correct, resetting attempts")
        user.reset_failed_attempts()
        db.session.commit()
        print("Creating access token")
        access_token = create_access_token(
            identity=user.email,
            expires_delta=timedelta(hours=24),
            additional_claims={
                'user_id': user.id,
                'name': user.name,
                'two_factor_enabled': user.two_factor_enabled
            }
        )
        print("Token created successfully")
        logger.log_info(f"Successful login for user: {email}", 'authentication')
        user_dict = user.to_dict()
        print(f"User dict: {user_dict}")
        response = jsonify({
            'access_token': access_token,
            'user': user_dict,
            'requires_2fa': user.two_factor_enabled
        })
        print("Returning success response")
        return response, 200

    except Exception as e:
        print(f"Exception in login: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        logger.log_error(f"Login error: {str(e)}", 'authentication')
        return jsonify({'error': 'Internal server error'}), 500

# ... (rest of app.py remains the same)

@app.route('/api/register', methods=['POST'])
@limiter.limit("3 per hour")
def register():
    try:
        data = request.get_json()
        if not data or not data.get('email') or not data.get('password') or not data.get('name'):
            return jsonify({'error': 'Name, email, and password are required'}), 400

        email = data['email'].strip().lower()
        name = data['name'].strip()
        password = data['password']

        if not re.match(r'^[^\s@]+@[^\s@]+\.[^\s@]+$', email):
            return jsonify({'error': 'Please enter a valid email address'}), 400

        if len(name) < 2:
            return jsonify({'error': 'Name must be at least 2 characters long'}), 400

        is_valid, password_message = validate_password_strength(password)
        if not is_valid:
            return jsonify({'error': password_message}), 400

        if User.query.filter_by(email=email).first():
            return jsonify({'error': 'User already exists with this email address'}), 409

        new_user = User(email=email, name=name, email_verified=False)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()
        logger.log_info(f"New user registered: {email}", 'authentication')
        return jsonify({
            'success': True,
            'message': 'Account created successfully. Please check your email to verify your account.',
            'user': {
                'id': new_user.id,
                'email': new_user.email,
                'name': new_user.name,
                'email_verified': new_user.email_verified
            }
        }), 201

    except Exception as e:
        db.session.rollback()
        logger.log_error(f"Registration error: {str(e)}", 'authentication')
        return jsonify({'error': 'Internal server error during registration'}), 500

@app.route('/api/forecast/train', methods=['POST'])
@jwt_required()
def train_forecast_models():
    try:
        data = request.get_json()
        sales_data_path = data.get('sales_data_path', '../data/sales.csv')
        weather_data_path = data.get('weather_data_path')
        logger.log_prediction('demand_forecast', {'data_path': sales_data_path})
        results = demand_forecaster.train_models(sales_data_path, weather_data_path)
        return jsonify({
            'success': True,
            'message': 'Models trained successfully',
            'results': results
        })
    except Exception as e:
        logger.log_error(f"Forecast training error: {str(e)}", 'forecast_training')
        return jsonify({'error': str(e)}), 500

@app.route('/api/forecast/predict', methods=['POST'])
@jwt_required()
def predict_demand():
    try:
        data = request.get_json()
        product = data.get('product')
        forecast_horizon = data.get('forecast_horizon', 30)
        country = data.get('country', 'US')
        if not product:
            return jsonify({'error': 'Product is required'}), 400
        forecast = demand_forecaster.predict_demand_with_confidence(product, forecast_horizon, country)
        summary = demand_forecaster.get_forecast_summary(product, forecast_horizon)
        if not forecast:
            forecast = [
                {'date': (datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d'), 'predicted_demand': 100, 'confidence_score': 0.9}
                for i in range(forecast_horizon)
            ]
            summary = {'status': 'Mock data', 'avg_demand': 100}
        for item in forecast:
            existing = DemandForecast.query.filter_by(
                product_id=product,
                warehouse_id='default',
                forecast_date=datetime.strptime(item['date'], '%Y-%m-%d').date()
            ).first()
            if not existing:
                db_forecast = DemandForecast(
                    product_id=product,
                    warehouse_id='default',
                    forecast_date=datetime.strptime(item['date'], '%Y-%m-%d').date(),
                    predicted_demand=item['predicted_demand'],
                    confidence_score=item['confidence_score'],
                    model_used='Ensemble'
                )
                db.session.add(db_forecast)
        db.session.commit()
        return jsonify({
            'success': True,
            'forecast': forecast,
            'summary': summary
        })
    except Exception as e:
        logger.log_error(f"Demand prediction error: {str(e)}", 'demand_prediction')
        return jsonify({'error': str(e)}), 500

@app.route('/api/inventory/analyze', methods=['POST'])
@jwt_required()
def analyze_inventory():
    try:
        data = request.get_json()
        inventory_data_path = data.get('inventory_data_path', '../data/inventory.csv')
        sales_data_path = data.get('sales_data_path', '../data/sales.csv')
        try:
            import pandas as pd
            inventory_df = pd.read_csv(inventory_data_path)
            sales_df = pd.read_csv(sales_data_path) if os.path.exists(sales_data_path) else None
        except ImportError:
            return jsonify({'error': 'Data analysis libraries not available'}), 503
        results = inventory_manager.analyze_inventory_levels(inventory_df, sales_df)
        report = inventory_manager.generate_inventory_report(results)
        if not results:
            results = {'recommendations': []}
            report = {'status': 'Mock data', 'summary': 'No analysis available'}
        return jsonify({
            'success': True,
            'analysis': results,
            'report': report
        })
    except Exception as e:
        logger.log_error(f"Inventory analysis error: {str(e)}", 'inventory_analysis')
        return jsonify({'error': str(e)}), 500

@app.route('/api/retrain-demand-model/<int:product_id>', methods=['POST'])
@jwt_required()
def retrain_demand_model(product_id):
    try:
        current_user = get_jwt_identity()
        data_path = '../data/sales.csv'  # Default data path
        drift_threshold = request.json.get('drift_threshold', 0.05) if request.is_json else 0.05
        
        logger.log_info(f"Retraining demand model for product {product_id} requested by {current_user}", 'model_retraining')
        
        retrain_result = demand_forecaster.retrain_models_automated(product_id, data_path, drift_threshold)
        
        if retrain_result['retrained']:
            logger.log_info(f"Model retrained successfully for product {product_id}", 'model_retraining')
        else:
            logger.log_info(f"No retraining needed for product {product_id}: {retrain_result['status']}", 'model_retraining')
        
        return jsonify({
            'success': True,
            'result': retrain_result
        })
    except Exception as e:
        logger.log_error(f"Model retraining error for product {product_id}: {str(e)}", 'model_retraining')
        return jsonify({'error': str(e)}), 500

@app.route('/api/inventory-recommendations', methods=['GET'])
@jwt_required()
def get_inventory_recommendations():
    try:
        inventory_csv = '../data/inventory.csv'
        sales_csv = '../data/sales.csv'
        recommendations_data = []
        if ML_MODULES_AVAILABLE:
            try:
                import pandas as pd
                inventory_df = pd.read_csv(inventory_csv)
                sales_df = pd.read_csv(sales_csv) if os.path.exists(sales_csv) else None
                recommendations_data = inventory_manager.analyze_inventory_levels(inventory_df, sales_df).get('recommendations', [])
            except ImportError:
                pass
        if not recommendations_data:
            recommendations_data = [
                {
                    'product_id': 'PROD001',
                    'warehouse_id': 'WH001',
                    'current_stock': 150,
                    'recommended_order_quantity': 50,
                    'safety_stock': 30,
                    'confidence_score': 0.85,
                    'reasoning': 'Stock level below optimal threshold',
                    'lead_time_days': 5
                },
                {
                    'product_id': 'PROD002',
                    'warehouse_id': 'WH001',
                    'current_stock': 300,
                    'recommended_order_quantity': 0,
                    'safety_stock': 25,
                    'confidence_score': 0.92,
                    'reasoning': 'Stock is sufficient',
                    'lead_time_days': 3
                }
            ]
        recommendations = []
        for item in recommendations_data:
            recommendations.append({
                'product_id': item['product_id'] if isinstance(item, dict) else item.product_id,
                'warehouse_id': item['warehouse_id'] if isinstance(item, dict) else item.warehouse_id,
                'current_stock': item['current_stock'] if isinstance(item, dict) else item.current_stock,
                'reserved_stock': item.get('reserved_stock', 0) if isinstance(item, dict) else 0,
                'available_stock': item.get('available_stock', item['current_stock']) if isinstance(item, dict) else item.current_stock,
                'recommended_order_quantity': item.get('recommended_order_quantity', 0) if isinstance(item, dict) else item.recommended_order_quantity,
                'safety_stock': item.get('safety_stock', 30) if isinstance(item, dict) else item.safety_stock,
                'confidence_score': item.get('confidence_score', 0.9) if isinstance(item, dict) else item.confidence_score,
                'reasoning': item.get('reasoning', 'Stock is sufficient') if isinstance(item, dict) else item.reasoning,
                'lead_time_days': item.get('lead_time_days', 5) if isinstance(item, dict) else item.lead_time_days
            })
        return jsonify({'success': True, 'recommendations': recommendations})
    except Exception as e:
        logger.log_error(f"Inventory recommendations error: {str(e)}", 'inventory_recommendations')
        return jsonify({'error': str(e)}), 500

@app.route('/api/inventory/multi-echelon-optimize', methods=['POST'])
@jwt_required()
def optimize_multi_echelon_inventory():
    """API endpoint for multi-echelon inventory optimization using PuLP"""
    try:
        data = request.get_json()
        warehouses = data.get('warehouses', [])
        products = data.get('products', [])
        demand_forecasts = data.get('demand_forecasts', {})
        supplier_data = data.get('supplier_data', {})

        if not warehouses or not products:
            return jsonify({'error': 'Warehouses and products data are required'}), 400

        optimization_result = inventory_manager.optimize_multi_echelon_inventory_pulp(
            warehouses, products, demand_forecasts, supplier_data
        )

        return jsonify({
            'success': True,
            'optimization_result': optimization_result
        })
    except Exception as e:
        logger.log_error(f"Multi-echelon optimization error: {str(e)}", 'multi_echelon_optimization')
        return jsonify({'error': str(e)}), 500

@app.route('/api/inventory/send-reorder-alert', methods=['POST'])
@jwt_required()
def send_reorder_alert():
    """API endpoint for sending reorder alerts via email"""
    try:
        data = request.get_json()
        recommendation_data = data.get('recommendation')
        recipient_emails = data.get('recipient_emails', [])

        if not recommendation_data or not recipient_emails:
            return jsonify({'error': 'Recommendation data and recipient emails are required'}), 400

        # Convert dict to InventoryRecommendation object
        if enhanced_available:
            from enhanced_inventory_management import InventoryRecommendation
            recommendation = InventoryRecommendation(
                product_id=recommendation_data['product_id'],
                warehouse_id=recommendation_data['warehouse_id'],
                current_stock=recommendation_data['current_stock'],
                recommended_order_quantity=recommendation_data['recommended_order_quantity'],
                reorder_point=recommendation_data['reorder_point'],
                safety_stock=recommendation_data['safety_stock'],
                lead_time_days=recommendation_data['lead_time_days'],
                confidence_score=recommendation_data['confidence_score'],
                reasoning=recommendation_data['reasoning'],
                supplier_recommendations=recommendation_data['supplier_recommendations'],
                sustainability_impact=recommendation_data['sustainability_impact']
            )
        else:
            # Fallback for basic modules
            class InventoryRecommendation:
                def __init__(self, **kwargs):
                    for k, v in kwargs.items():
                        setattr(self, k, v)
            recommendation = InventoryRecommendation(**recommendation_data)

        success = inventory_manager.send_reorder_alert(recommendation, recipient_emails)

        if success:
            return jsonify({
                'success': True,
                'message': 'Reorder alert sent successfully'
            })
        else:
            return jsonify({'error': 'Failed to send reorder alert'}), 500

    except Exception as e:
        logger.log_error(f"Send reorder alert error: {str(e)}", 'send_reorder_alert')
        return jsonify({'error': str(e)}), 500

@app.route('/api/logistics/optimize', methods=['POST'])
@jwt_required()
def optimize_routes():
    try:
        data = request.get_json()
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
        logger.log_error(f"Route optimization error: {str(e)}", 'route_optimization')
        return jsonify({'error': str(e)}), 500

@app.route('/api/anomaly/detect', methods=['POST'])
@jwt_required()
def detect_anomalies():
    try:
        data = request.get_json()
        data_path = data.get('data_path', '../data/sales.csv')
        anomalies = anomaly_detector.predict_anomalies(data_path)
        report = anomaly_detector.get_anomaly_report(data_path)
        if not anomalies:
            anomalies = []
            report = {'status': 'Mock data', 'summary': 'No anomalies detected'}
        return jsonify({
            'success': True,
            'anomalies': anomalies if isinstance(anomalies, list) else anomalies.to_dict('records'),
            'report': report
        })
    except Exception as e:
        logger.log_error(f"Anomaly detection error: {str(e)}", 'anomaly_detection')
        return jsonify({'error': str(e)}), 500

@app.route('/api/dashboard/data', methods=['GET'])
@jwt_required()
def get_dashboard_data():
    try:
        logger.log_info(f"Dashboard data requested by user: {get_jwt_identity()}", 'dashboard_data')
        dashboard_data = {
            'demand_forecasting': {
                'total_products': 25,
                'active_models': 23,
                'avg_accuracy': 94.2
            },
            'inventory_status': {
                'total_products': 1247,
                'low_stock_alerts': 12,
                'optimal_stock': 89
            },
            'logistics_performance': {
                'routes_optimized': 156,
                'total_distance': 2847,
                'cost_savings': 23.5
            },
            'anomaly_detection': {
                'data_points_analyzed': 45231,
                'anomalies_detected': 8,
                'detection_rate': 99.1
            },
            'recent_activity': [
                {'time': '2 hours ago', 'description': 'Demand forecast updated for Product A', 'type': 'forecast'},
                {'time': '4 hours ago', 'description': 'Inventory optimization completed', 'type': 'inventory'},
                {'time': '6 hours ago', 'description': 'Route optimization saved 15% distance', 'type': 'logistics'},
                {'time': '8 hours ago', 'description': 'Anomaly detected in sales data', 'type': 'anomaly'}
            ],
            'system_health': {
                'api_status': 'healthy',
                'database': 'connected',
                'ml_models': 'active' if ML_MODULES_AVAILABLE else 'unavailable',
                'last_update': datetime.now().strftime('%H:%M:%S')
            }
        }
        return jsonify(dashboard_data)
    except Exception as e:
        logger.log_error(f"Dashboard data error: {str(e)}", 'dashboard_data')
        return jsonify({'error': str(e)}), 500

@app.route('/api/dashboard-metrics', methods=['GET'])
@jwt_required()
def get_dashboard_metrics():
    try:
        metrics = {
            'totalInventory': 1250000,
            'activeOrders': 45,
            'delayedShipments': 3,
            'totalRevenue': 890000,
            'demandForecast': 95,
            'supplierPerformance': 87,
            'systemHealth': 95,
            'lastUpdate': datetime.now().strftime('%H:%M:%S')
        }
        logger.log_info(f"Dashboard metrics fetched for user: {get_jwt_identity()}", 'dashboard_metrics')
        return jsonify(metrics)
    except Exception as e:
        logger.log_error(f"Dashboard metrics error: {str(e)}", 'dashboard_metrics')
        return jsonify({'error': str(e)}), 500

@app.route('/api/inventory', methods=['GET'])
@jwt_required()
def get_inventory():
    try:
        inventory_items = Inventory.query.all()
        data = []
        for item in inventory_items:
            data.append({
                'id': item.id,
                'product_id': item.product_id,
                'warehouse_id': item.warehouse_id,
                'current_stock': item.current_stock,
                'reserved_stock': getattr(item, 'reserved_stock', 0),
                'min_stock': getattr(item, 'min_stock', 0),
                'max_stock': getattr(item, 'max_stock', 0),
                'last_updated': item.last_updated.isoformat() if item.last_updated else None
            })
        logger.log_info(f"Inventory data fetched for user: {get_jwt_identity()}", 'inventory_fetch')
        return jsonify({'success': True, 'inventory': data})
    except Exception as e:
        logger.log_error(f"Inventory fetch error: {str(e)}", 'inventory_fetch')
        return jsonify({'error': str(e)}), 500

@app.route('/api/inventory-analysis', methods=['GET'])
@jwt_required()
def get_inventory_analysis():
    try:
        return jsonify({
            'success': True,
            'recommendations': [
                {
                    'product_id': 'PROD001',
                    'product_name': 'Product A',
                    'warehouse_id': 'WH001',
                    'current_stock': 150,
                    'recommended_order_quantity': 50,
                    'reorder_point': 200,
                    'safety_stock': 30,
                    'confidence': 0.85,
                    'reason': 'Stock level below reorder point. Recommended to prevent stockout.',
                    'lead_time_days': 5
                },
                {
                    'product_id': 'PROD002',
                    'product_name': 'Product B',
                    'warehouse_id': 'WH001',
                    'current_stock': 300,
                    'recommended_order_quantity': 0,
                    'reorder_point': 250,
                    'safety_stock': 25,
                    'confidence': 0.92,
                    'reason': 'Stock is sufficient. No immediate action required.',
                    'lead_time_days': 3
                },
                {
                    'product_id': 'PROD003',
                    'product_name': 'Product C',
                    'warehouse_id': 'WH001',
                    'current_stock': 80,
                    'recommended_order_quantity': 120,
                    'reorder_point': 100,
                    'safety_stock': 20,
                    'confidence': 0.78,
                    'reason': 'Low stock detected. Order to cover predicted demand.',
                    'lead_time_days': 7
                }
            ],
            'analysis': {
                'total_products': 1247,
                'low_stock_items': 12,
                'optimal_stock_percentage': 89,
                'turnover_rate': 4.2
            }
        })
    except Exception as e:
        logger.log_error(f"Inventory analysis error: {str(e)}", 'inventory_analysis')
        return jsonify({'error': str(e)}), 500

@app.route('/api/products', methods=['GET'])
@jwt_required()
def get_products():
    try:
        return jsonify({
            'success': True,
            'products': [
                {'id': 1, 'name': 'Product A', 'category': 'Electronics', 'stock': 150},
                {'id': 2, 'name': 'Product B', 'category': 'Clothing', 'stock': 75},
                {'id': 3, 'name': 'Product C', 'category': 'Home Goods', 'stock': 200}
            ]
        })
    except Exception as e:
        logger.log_error(f"Products error: {str(e)}", 'products')
        return jsonify({'error': str(e)}), 500

@app.route('/api/demand-forecast/<int:product_id>', methods=['GET'])
@jwt_required()
def get_demand_forecast(product_id):
    try:
        current_user = get_jwt_identity()
        forecast_horizon = request.args.get('horizon', 30, type=int)
        country = request.args.get('country', 'US')
        cache_key = f"forecast_{product_id}_{current_user}_{forecast_horizon}_{country}"
        cache_ttl = timedelta(hours=1)  # Cache for 1 hour

        # Check in-memory cache first
        if 'forecast_cache' in globals() and cache_key in forecast_cache:
            cached_time, cached_data = forecast_cache[cache_key]
            if datetime.now() - cached_time < cache_ttl:
                logger.log_info(f"Cache hit for forecast {product_id}", 'demand_forecast')
                return jsonify(cached_data)

        logger.log_info(f"Generating demand forecast for product {product_id}, horizon {forecast_horizon}, country {country}", 'demand_forecast')
        
        summary = demand_forecaster.get_forecast_summary(product_id, forecast_horizon, country)
        forecast_details = summary['forecast_details']
        
        if not forecast_details or isinstance(forecast_details, dict):
            # Fallback to mock data
            forecast_details = [
                {
                    'date': (datetime.now() + timedelta(days=i+1)).strftime('%Y-%m-%d'), 
                    'predicted_demand': round(100 + (i * 5) + (i % 7 * 10)),  # Simulate trend and seasonality
                    'confidence_score': round(0.85 + (0.1 * (i % 5)), 2),  # Varying confidence
                    'lower_bound': round(80 + (i * 4), 0),
                    'upper_bound': round(120 + (i * 6), 0),
                    'individual_predictions': {
                        'RandomForest': round(95 + (i * 3)),
                        'GradientBoosting': round(105 + (i * 4)),
                        'ARIMA': round(90 + (i * 2.5))
                    }
                }
                for i in range(forecast_horizon)
            ]
            # Mock summary for fallback
            summary = {
                'total_predicted_demand': sum([item['predicted_demand'] for item in forecast_details]),
                'average_daily_demand': round(sum([item['predicted_demand'] for item in forecast_details]) / len(forecast_details), 2),
                'demand_volatility': 0.15,
                'ensemble_weights': {
                    'RandomForest': 0.4,
                    'GradientBoosting': 0.35,
                    'ARIMA': 0.25
                },
                'validation_metrics': {}
            }
        
        # Prepare response structure matching frontend expectations
        response_data = {
            'success': True,
            'total_predicted_demand': summary['total_predicted_demand'],
            'average_daily_demand': summary['average_daily_demand'],
            'demand_volatility': summary['demand_volatility'],
            'ensemble_weights': summary.get('ensemble_weights', {}),
            'forecast_details': forecast_details,
            'validation_metrics': summary.get('validation_metrics', {}),
            'recommendations': summary.get('recommendations', [])
        }
        
        # Cache the response
        forecast_cache[cache_key] = (datetime.now(), response_data)
        
        # Store in database (only predicted_demand and confidence for simplicity), avoiding duplicates
        for item in forecast_details:
            existing = DemandForecast.query.filter_by(
                product_id=product_id,
                warehouse_id='default',
                forecast_date=datetime.strptime(item['date'], '%Y-%m-%d').date()
            ).first()
            if not existing:
                db_forecast = DemandForecast(
                    product_id=product_id,
                    warehouse_id='default',
                    forecast_date=datetime.strptime(item['date'], '%Y-%m-%d').date(),
                    predicted_demand=item['predicted_demand'],
                    confidence_score=item['confidence_score'],
                    model_used='Ensemble'
                )
                db.session.add(db_forecast)
        db.session.commit()
        
        logger.log_info(f"Demand forecast generated and cached for product {product_id}", 'demand_forecast')
        return jsonify(response_data)
    except Exception as e:
        logger.log_error(f"Demand forecast GET error: {str(e)}", 'demand_forecast')
        return jsonify({'error': str(e)}), 500

@app.route('/api/logistics-routes', methods=['GET'])
@jwt_required()
def get_logistics_routes():
    try:
        return jsonify({
            'success': True,
            'routes': [
                {
                    'id': 1,
                    'name': 'Route 1',
                    'origin': 'New York',
                    'destination': 'Boston',
                    'distance': 125.5,
                    'current_cost': 450.0,
                    'current_time': '4.2 hours',
                    'status': 'active'
                },
                {
                    'id': 2,
                    'name': 'Route 2',
                    'origin': 'Los Angeles',
                    'destination': 'San Francisco',
                    'distance': 89.2,
                    'current_cost': 320.0,
                    'current_time': '3.1 hours',
                    'status': 'completed'
                },
                {
                    'id': 3,
                    'name': 'Route 3',
                    'origin': 'Chicago',
                    'destination': 'Detroit',
                    'distance': 156.8,
                    'current_cost': 580.0,
                    'current_time': '5.8 hours',
                    'status': 'planned'
                }
            ]
        })
    except Exception as e:
        logger.log_error(f"Logistics routes error: {str(e)}", 'logistics_routes')
        return jsonify({'error': str(e)}), 500


@app.route('/api/eta-estimates', methods=['GET'])
@jwt_required()
def get_eta_estimates():
    try:
        # Base routes data
        routes = [
            {'id': 1, 'current_time': '2.25', 'status': 'active'},  # 2h 15m
            {'id': 2, 'current_time': '4.5', 'status': 'active'},   # 4h 30m
            {'id': 3, 'current_time': '1.75', 'status': 'active'},  # 1h 45m
            {'id': 4, 'current_time': '3.166', 'status': 'active'}  # 3h 10m
        ]
        
        eta_data = []
        import random
        for route in routes:
            hours_float = float(route['current_time'])
            hours = int(hours_float)
            minutes = int((hours_float - hours) * 60)
            eta = f"{hours}h {minutes}m"
            
            # Determine status based on time and random factor
            base_status = 'on-time' if hours_float < 2 else 'risky' if hours_float < 4 else 'delayed'
            status = base_status if random.random() > 0.2 else random.choice(['on-time', 'risky', 'delayed'])
            
            progress = random.randint(40, 95)
            
            eta_data.append({
                'id': route['id'],
                'eta': eta,
                'status': status,
                'progress': progress
            })
        
        logger.log_info(f"ETA estimates fetched for user: {get_jwt_identity()}", 'eta_estimates')
        return jsonify({
            'success': True,
            'etaData': eta_data
        })
    except Exception as e:
        logger.log_error(f"ETA estimates error: {str(e)}", 'eta_estimates')
        return jsonify({'error': str(e)}), 500

@app.route('/api/optimize-route/<int:route_id>', methods=['POST'])
@jwt_required()
def optimize_route(route_id):
    try:
        data = request.get_json()
        optimization_criteria = data.get('optimization_criteria', 'cost_and_time')

        # Mock route data based on route_id (in a real app, this would come from database)
        routes_data = {
            1: {
                'id': 1,
                'name': 'Route 1',
                'origin': 'New York',
                'destination': 'Boston',
                'distance': 125.5,
                'current_cost': 450.0,
                'current_time': 4.2
            },
            2: {
                'id': 2,
                'name': 'Route 2',
                'origin': 'Los Angeles',
                'destination': 'San Francisco',
                'distance': 89.2,
                'current_cost': 320.0,
                'current_time': 3.1
            },
            3: {
                'id': 3,
                'name': 'Route 3',
                'origin': 'Chicago',
                'destination': 'Detroit',
                'distance': 156.8,
                'current_cost': 580.0,
                'current_time': 5.8
            }
        }

        if route_id not in routes_data:
            return jsonify({'error': 'Route not found'}), 404

        current_route = routes_data[route_id]

        # Simulate optimization based on criteria
        if optimization_criteria == 'cost_and_time':
            # Apply optimization improvements
            distance_improvement = 0.15  # 15% reduction
            cost_improvement = 0.20      # 20% reduction
            time_improvement = 0.18      # 18% reduction
            co2_improvement = 0.25       # 25% reduction

            new_distance = current_route['distance'] * (1 - distance_improvement)
            new_cost = current_route['current_cost'] * (1 - cost_improvement)
            new_time = current_route['current_time'] * (1 - time_improvement)
            co2_reduction = current_route['distance'] * 0.15 * co2_improvement  # Mock CO2 calculation

            cost_savings = current_route['current_cost'] - new_cost
            time_savings = current_route['current_time'] - new_time

            optimization_result = {
                'cost_savings': round(cost_savings, 2),
                'cost_savings_percentage': round(cost_improvement * 100, 1),
                'time_savings': round(time_savings, 2),
                'time_savings_percentage': round(time_improvement * 100, 1),
                'co2_reduction': round(co2_reduction, 2),
                'co2_reduction_percentage': round(co2_improvement * 100, 1),
                'new_distance': round(new_distance, 2),
                'new_cost': round(new_cost, 2),
                'new_time': round(new_time, 2),
                'optimization_method': 'Enhanced Route Optimization with Traffic Analysis'
            }

        elif optimization_criteria == 'cost_only':
            cost_improvement = 0.25
            new_cost = current_route['current_cost'] * (1 - cost_improvement)
            cost_savings = current_route['current_cost'] - new_cost

            optimization_result = {
                'cost_savings': round(cost_savings, 2),
                'cost_savings_percentage': round(cost_improvement * 100, 1),
                'time_savings': 0,
                'time_savings_percentage': 0,
                'co2_reduction': round(current_route['distance'] * 0.15 * 0.1, 2),
                'co2_reduction_percentage': 10,
                'new_distance': current_route['distance'],
                'new_cost': round(new_cost, 2),
                'new_time': current_route['current_time'],
                'optimization_method': 'Cost-Focused Route Optimization'
            }

        elif optimization_criteria == 'time_only':
            time_improvement = 0.22
            new_time = current_route['current_time'] * (1 - time_improvement)
            time_savings = current_route['current_time'] - new_time

            optimization_result = {
                'cost_savings': round(current_route['current_cost'] * 0.05, 2),
                'cost_savings_percentage': 5,
                'time_savings': round(time_savings, 2),
                'time_savings_percentage': round(time_improvement * 100, 1),
                'co2_reduction': round(current_route['distance'] * 0.15 * 0.15, 2),
                'co2_reduction_percentage': 15,
                'new_distance': current_route['distance'],
                'new_cost': round(current_route['current_cost'] * 0.95, 2),
                'new_time': round(new_time, 2),
                'optimization_method': 'Time-Focused Route Optimization'
            }

        else:
            return jsonify({'error': 'Invalid optimization criteria'}), 400

        logger.log_info(f"Route {route_id} optimized successfully with criteria: {optimization_criteria}", 'route_optimization')

        return jsonify({
            'success': True,
            'optimization_result': optimization_result
        })

    except Exception as e:
        logger.log_error(f"Route optimization error for route {route_id}: {str(e)}", 'route_optimization')
        return jsonify({'error': 'Failed to optimize route'}), 500

@app.route('/api/implement-optimization', methods=['POST'])
@jwt_required()
def implement_optimization():
    try:
        data = request.get_json()
        # In a real implementation, this would update the database with the optimized route
        logger.log_info("Optimization implementation requested", 'optimization_implementation')
        return jsonify({
            'success': True,
            'message': 'Optimization implemented successfully'
        })
    except Exception as e:
        logger.log_error(f"Optimization implementation error: {str(e)}", 'optimization_implementation')
        return jsonify({'error': 'Failed to implement optimization'}), 500

@app.route('/api/verify-2fa', methods=['POST'])
@jwt_required()
def verify_2fa():
    try:
        data = request.get_json()
        code = data.get('code')
        if not code:
            return jsonify({'error': '2FA code is required'}), 400
        user = User.query.filter_by(email=get_jwt_identity()).first()
        if not user:
            return jsonify({'error': 'User not found'}), 404
        # Mock 2FA verification (replace with TOTP in production)
        if code == '123456':
            access_token = create_access_token(
                identity=user.email,
                expires_delta=timedelta(hours=24),
                additional_claims={
                    'user_id': user.id,
                    'name': user.name,
                    'two_factor_enabled': user.two_factor_enabled,
                    '2fa_verified': True
                }
            )
            return jsonify({
                'success': True,
                'token': access_token,
                'message': '2FA verification successful'
            })
        return jsonify({'error': 'Invalid 2FA code'}), 401
    except Exception as e:
        logger.log_error(f"2FA verification error: {str(e)}", '2fa_verification')
        return jsonify({'error': str(e)}), 500

# -------------------- SOCKET.IO EVENTS --------------------
@socketio.on('connect')
def handle_connect(auth):
    try:
        if auth and 'token' in auth:
            token = auth['token']
            # Create a mutable headers dictionary
            headers = {'Authorization': f'Bearer {token}'}
            # Use Flask's test request context to verify JWT
            with app.test_request_context(headers=headers):
                verify_jwt_in_request(locations=['headers'])
                identity = get_jwt_identity()
                sid = getattr(request, 'sid', 'unknown')
                logger.log_info(f"WebSocket client connected: {sid} for user: {identity}", 'websocket')
                emit('message', {'type': 'info', 'payload': f'Connected to WebSocket server as {identity}'})
        else:
            logger.log_error("WebSocket connection refused: No token provided", 'websocket')
            raise ConnectionRefusedError('Authentication token required')
    except Exception as e:
        logger.log_error(f"WebSocket connection error: {str(e)}", 'websocket')
        raise ConnectionRefusedError('Authentication failed')

@socketio.on('disconnect')
def handle_disconnect():
    logger.log_info(f"WebSocket client disconnected: {request.sid}", 'websocket')

@socketio.on('dashboard_update')
def handle_dashboard_update(data):
    try:
        with app.test_request_context(headers=request.headers):
            verify_jwt_in_request(locations=['headers'])
            identity = get_jwt_identity()
            logger.log_info(f"Received dashboard update request from {identity}: {data}", 'websocket')
            metrics = {
                'totalInventory': 1250000,
                'activeOrders': 45,
                'delayedShipments': 3,
                'totalRevenue': 890000,
                'demandForecast': 95,
                'supplierPerformance': 87,
                'systemHealth': 95,
                'lastUpdate': datetime.now().strftime('%H:%M:%S')
            }
            emit('dashboard_metrics', metrics)
    except Exception as e:
        logger.log_error(f"Dashboard update error: {str(e)}", 'websocket')
        emit('error', {'error': str(e)})

@socketio.on('inventory_update')
def handle_inventory_update(data):
    try:
        with app.test_request_context(headers=request.headers):
            verify_jwt_in_request(locations=['headers'])
            logger.log_info(f"Inventory update: {data}", 'websocket')
            emit('inventory_update', {'total_inventory': data.get('total_inventory', 1250000)})
    except Exception as e:
        logger.log_error(f"Inventory update error: {str(e)}", 'websocket')
        emit('error', {'error': str(e)})

@socketio.on('order_update')
def handle_order_update(data):
    try:
        with app.test_request_context(headers=request.headers):
            verify_jwt_in_request(locations=['headers'])
            logger.log_info(f"Order update: {data}", 'websocket')
            emit('order_update', {'active_orders': data.get('active_orders', 45)})
    except Exception as e:
        logger.log_error(f"Order update error: {str(e)}", 'websocket')
        emit('error', {'error': str(e)})

@socketio.on('alert')
def handle_alert(data):
    try:
        with app.test_request_context(headers=request.headers):
            verify_jwt_in_request(locations=['headers'])
            logger.log_info(f"Alert: {data}", 'websocket')
            emit('alert', {
                'timestamp': datetime.now().strftime('%H:%M:%S'),
                'message': data.get('message', 'Alert received'),
                'priority': data.get('priority', 'info')
            })
    except Exception as e:
        logger.log_error(f"Alert error: {str(e)}", 'websocket')
        emit('error', {'error': str(e)})

# -------------------- ERROR HANDLERS --------------------
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Resource not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.log_error(f"Internal error: {str(error)}", 'internal_error')
    return jsonify({'error': 'Internal server error'}), 500

# -------------------- RUN SERVER --------------------
# Global cache for forecasts (simple in-memory dict)
forecast_cache = {}

# Register futuristic endpoints
try:
    from futuristic_endpoints import register_futuristic_endpoints
    app = register_futuristic_endpoints(app)
    print("Futuristic inventory management endpoints registered successfully")
except ImportError as e:
    print(f"Futuristic endpoints not available: {e}")

if __name__ == '__main__':
    logger.main_logger.info("Starting Supply Chain Optimization System with WebSocket")
    socketio.run(app, debug=app.config['DEBUG'], host='0.0.0.0', port=5000)
