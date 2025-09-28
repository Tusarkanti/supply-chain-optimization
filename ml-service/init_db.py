```python
# ml-service/init_db.py
import os
import sys
from datetime import datetime, timedelta
from werkzeug.security import generate_password_hash

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app, db
from database_schema import (
    User, Product, Warehouse, Inventory, Customer,
    Order, OrderItem, Shipment, DemandForecast,
    RouteOptimization, RouteStop
)
from config import get_config

# Update config for init (use development)
app.config.from_object(get_config('development'))
print("Init config URI:", app.config['SQLALCHEMY_DATABASE_URI'])


def create_database():
    """For SQLite, no database creation needed (MySQL/Postgres would require it)."""
    print("Using SQLite - no database creation needed.")
    return True


def seed_demo_data():
    """Seed the database with sample data for all models."""
    with app.app_context():
        # Clear existing data
        db.session.query(RouteStop).delete()
        db.session.query(RouteOptimization).delete()
        db.session.query(DemandForecast).delete()
        db.session.query(Shipment).delete()
        db.session.query(OrderItem).delete()
        db.session.query(Order).delete()
        db.session.query(Inventory).delete()
        db.session.query(Customer).delete()
        db.session.query(Warehouse).delete()
        db.session.query(Product).delete()
        db.session.query(User).delete()
        db.session.commit()

        # ---------------- USERS ----------------
        users = [
            User(email='admin@supplychain.com', name='Admin User',
                 password_hash=generate_password_hash('admin'),
                 two_factor_enabled=False, email_verified=True),
            User(email='user1@supplychain.com', name='John Doe',
                 password_hash=generate_password_hash('password123'),
                 two_factor_enabled=False, email_verified=True),
            User(email='demo@supplychain.com', name='Demo User',
                 password_hash=generate_password_hash('demo123'),
                 two_factor_enabled=False, email_verified=True)
        ]
        db.session.add_all(users)
        db.session.commit()

        # ---------------- PRODUCTS ----------------
        products = [
            Product(name='Laptop', sku='LAP001', category='ELECTRONICS',
                    description='High-end gaming laptop',
                    unit_cost=999.99, selling_price=1299.99,
                    weight=2.5, dimensions='35x25x2'),
            Product(name='T-Shirt', sku='TSH001', category='CLOTHING',
                    description='Cotton t-shirt',
                    unit_cost=5.00, selling_price=15.00,
                    weight=0.2, dimensions='30x20x1'),
            Product(name='Smartphone', sku='SPH001', category='ELECTRONICS',
                    description='Latest smartphone',
                    unit_cost=699.99, selling_price=899.99,
                    weight=0.18, dimensions='15x7x0.8'),
            Product(name='Book', sku='BOK001', category='BOOKS',
                    description='Supply Chain Management book',
                    unit_cost=20.00, selling_price=25.00,
                    weight=0.5, dimensions='20x15x3'),
            Product(name='Coffee Maker', sku='CMK001', category='HOME',
                    description='Automatic coffee maker',
                    unit_cost=49.99, selling_price=79.99,
                    weight=3.0, dimensions='25x20x30')
        ]
        db.session.add_all(products)
        db.session.commit()

        # ---------------- CUSTOMERS ----------------
        customers = [
            Customer(name='Acme Corp', email='orders@acme.com',
                     phone='123-456-7890',
                     address='123 Main St, City, State 12345'),
            Customer(name='Beta Inc', email='sales@beta.com',
                     phone='098-765-4321',
                     address='456 Oak Ave, Town, State 67890'),
            Customer(name='Gamma Ltd', email='purchasing@gamma.com',
                     phone='555-123-4567',
                     address='789 Pine Rd, Village, State 11223')
        ]
        db.session.add_all(customers)
        db.session.commit()

        # ---------------- WAREHOUSES ----------------
        warehouses = [
            Warehouse(name='Main Warehouse', code='WH001',
                      address='100 Industrial Blvd',
                      city='Metro City', state='CA', country='USA',
                      postal_code='90210', latitude=34.0522, longitude=-118.2437,
                      capacity=10000),
            Warehouse(name='East Distribution Center', code='WH002',
                      address='200 Logistics Park',
                      city='East Town', state='NY', country='USA',
                      postal_code='10001', latitude=40.7128, longitude=-74.0060,
                      capacity=5000)
        ]
        db.session.add_all(warehouses)
        db.session.commit()

        # ---------------- INVENTORY ----------------
        inventory_entries = [
            Inventory(product_id=1, warehouse_id=1, current_stock=50,
                      reorder_point=20, max_stock=200),
            Inventory(product_id=2, warehouse_id=1, current_stock=300,
                      reorder_point=100, max_stock=1000),
            Inventory(product_id=3, warehouse_id=2, current_stock=100,
                      reorder_point=50, max_stock=500),
            Inventory(product_id=4, warehouse_id=1, current_stock=75,
                      reorder_point=30, max_stock=300),
            Inventory(product_id=5, warehouse_id=2, current_stock=25,
                      reorder_point=10, max_stock=100)
        ]
        db.session.add_all(inventory_entries)
        db.session.commit()

        # ---------------- ORDERS & ITEMS ----------------
        order1 = Order(customer_id=1, total_amount=2599.98, status='PENDING',
                       shipping_address='123 Main St, City, CA 12345',
                       shipping_city='City', shipping_state='CA', shipping_postal_code='12345')
        db.session.add(order1)
        db.session.commit()

        db.session.add_all([
            OrderItem(order_id=order1.id, product_id=1, quantity=2,
                      unit_price=1299.99, total_price=2599.98),
            OrderItem(order_id=order1.id, product_id=2, quantity=5,
                      unit_price=15.00, total_price=75.00)
        ])

        order2 = Order(customer_id=2, total_amount=899.99, status='SHIPPED',
                       shipping_address='456 Oak Ave, Town, NY 67890',
                       shipping_city='Town', shipping_state='NY', shipping_postal_code='67890')
        db.session.add(order2)
        db.session.commit()

        db.session.add(
            OrderItem(order_id=order2.id, product_id=3, quantity=1,
                      unit_price=899.99, total_price=899.99)
        )

        order3 = Order(customer_id=3, total_amount=125.00, status='DELIVERED',
                       shipping_address='789 Pine Rd, Village, CA 11223',
                       shipping_city='Village', shipping_state='CA', shipping_postal_code='11223')
        db.session.add(order3)
        db.session.commit()

        db.session.add(
            OrderItem(order_id=order3.id, product_id=4, quantity=5,
                      unit_price=25.00, total_price=125.00)
        )
        db.session.commit()

        # ---------------- SHIPMENTS ----------------
        db.session.add_all([
            Shipment(order_id=order2.id, warehouse_id=2, tracking_number='TRK123456',
                     status='IN_TRANSIT', shipment_date=datetime.now(), delivery_date=None),
            Shipment(order_id=order3.id, warehouse_id=1, tracking_number='TRK789012',
                     status='DELIVERED', shipment_date=datetime.now() - timedelta(days=2),
                     delivery_date=datetime.now() - timedelta(days=1))
        ])
        db.session.commit()

        # ---------------- DEMAND FORECASTS ----------------
        forecasts = []
        for i in range(10):
            forecasts.append(
                DemandForecast(product_id=(i % 5) + 1,
                               warehouse_id=(i % 2) + 1,
                               forecast_date=datetime.now().date() + timedelta(days=i+1),
                               predicted_demand=50 + (i * 10),
                               confidence_score=0.85,
                               model_used='XGBoost')
            )
        db.session.add_all(forecasts)
        db.session.commit()

        # ---------------- ROUTE OPTIMIZATION ----------------
        route1 = RouteOptimization(
            optimization_date=datetime.now().date(),
            total_distance=150.5, total_cost=250.00,
            vehicle_count=1, optimization_status='OPTIMIZED'
        )
        db.session.add(route1)
        db.session.commit()

        db.session.add_all([
            RouteStop(optimization_id=route1.id, warehouse_id=1, customer_id=1,
                      stop_sequence=1,
                      arrival_time=datetime.now() + timedelta(hours=1),
                      departure_time=datetime.now() + timedelta(hours=1.5),
                      load_quantity=10),
            RouteStop(optimization_id=route1.id, warehouse_id=1, customer_id=2,
                      stop_sequence=2,
                      arrival_time=datetime.now() + timedelta(hours=2),
                      departure_time=datetime.now() + timedelta(hours=2.5),
                      load_quantity=15)
        ])

        route2 = RouteOptimization(
            optimization_date=datetime.now().date(),
            total_distance=450.0, total_cost=750.00,
            vehicle_count=1, optimization_status='IN_PROGRESS'
        )
        db.session.add(route2)
        db.session.commit()

        # ---------------- SUMMARY ----------------
        print("✅ Demo data seeded successfully!")
        print(f"Users: {User.query.count()}")
        print(f"Products: {Product.query.count()}")
        print(f"Customers: {Customer.query.count()}")
        print(f"Warehouses: {Warehouse.query.count()}")
        print(f"Inventory: {Inventory.query.count()}")
        print(f"Orders: {Order.query.count()}")
        print(f"Shipments: {Shipment.query.count()}")
        print(f"Demand Forecasts: {DemandForecast.query.count()}")
        print(f"Route Optimizations: {RouteOptimization.query.count()}")


if __name__ == '__main__':
    if create_database():
        with app.app_context():
            db.create_all()
            seed_demo_data()
        print("🎯 Database initialization complete. Run 'python ml-service/app.py' to start the server.")
    else:
        print("❌ Failed to create database. Check DB server and credentials.")
```
