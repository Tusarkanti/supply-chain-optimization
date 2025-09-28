from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from enum import Enum
import secrets

db = SQLAlchemy()

class OrderStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"

class ProductCategory(Enum):
    ELECTRONICS = "electronics"
    CLOTHING = "clothing"
    FOOD = "food"
    BOOKS = "books"
    HOME = "home"
    SPORTS = "sports"

class Product(db.Model):
    __tablename__ = 'products'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    sku = db.Column(db.String(50), unique=True, nullable=False)
    category = db.Column(db.Enum(ProductCategory), nullable=False)
    description = db.Column(db.Text)
    unit_cost = db.Column(db.Float, nullable=False)
    selling_price = db.Column(db.Float, nullable=False)
    weight = db.Column(db.Float)  # in kg
    dimensions = db.Column(db.String(50))  # LxWxH in cm
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    inventory_levels = db.relationship('Inventory', backref='product', lazy=True)
    sales = db.relationship('Sale', backref='product', lazy=True)
    order_items = db.relationship('OrderItem', backref='product', lazy=True)

class Warehouse(db.Model):
    __tablename__ = 'warehouses'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    code = db.Column(db.String(20), unique=True, nullable=False)
    address = db.Column(db.Text, nullable=False)
    city = db.Column(db.String(50), nullable=False)
    state = db.Column(db.String(50), nullable=False)
    country = db.Column(db.String(50), nullable=False)
    postal_code = db.Column(db.String(20), nullable=False)
    latitude = db.Column(db.Float)
    longitude = db.Column(db.Float)
    capacity = db.Column(db.Integer)  # max units
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationships
    inventory_levels = db.relationship('Inventory', backref='warehouse', lazy=True)
    shipments = db.relationship('Shipment', backref='warehouse', lazy=True)

class Inventory(db.Model):
    __tablename__ = 'inventory'

    id = db.Column(db.Integer, primary_key=True)
    product_id = db.Column(db.Integer, db.ForeignKey('products.id'), nullable=False)
    warehouse_id = db.Column(db.Integer, db.ForeignKey('warehouses.id'), nullable=False)
    supplier_id = db.Column(db.Integer, db.ForeignKey('suppliers.id'), nullable=True)
    current_stock = db.Column(db.Integer, nullable=False, default=0)
    reserved_stock = db.Column(db.Integer, nullable=False, default=0)
    available_stock = db.Column(db.Integer, nullable=False, default=0)
    reorder_point = db.Column(db.Integer, nullable=False, default=0)
    max_stock = db.Column(db.Integer, nullable=False, default=0)
    last_updated = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationships
    movements = db.relationship('InventoryMovement', backref='inventory', lazy=True)

    __table_args__ = (db.UniqueConstraint('product_id', 'warehouse_id', name='unique_product_warehouse'),)

class Sale(db.Model):
    __tablename__ = 'sales'

    id = db.Column(db.Integer, primary_key=True)
    product_id = db.Column(db.Integer, db.ForeignKey('products.id'), nullable=False)
    quantity = db.Column(db.Integer, nullable=False)
    unit_price = db.Column(db.Float, nullable=False)
    total_amount = db.Column(db.Float, nullable=False)
    sale_date = db.Column(db.DateTime, nullable=False)
    location = db.Column(db.String(100), nullable=False)
    customer_id = db.Column(db.String(50))
    promotion_applied = db.Column(db.Boolean, default=False)
    weather_condition = db.Column(db.String(50))  # sunny, rainy, cloudy, etc.
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Customer(db.Model):
    __tablename__ = 'customers'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    phone = db.Column(db.String(20))
    address = db.Column(db.Text)
    city = db.Column(db.String(50))
    state = db.Column(db.String(50))
    postal_code = db.Column(db.String(20))
    latitude = db.Column(db.Float)
    longitude = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationships
    orders = db.relationship('Order', backref='customer', lazy=True)

class Order(db.Model):
    __tablename__ = 'orders'

    id = db.Column(db.Integer, primary_key=True)
    customer_id = db.Column(db.Integer, db.ForeignKey('customers.id'), nullable=False)
    order_date = db.Column(db.DateTime, default=datetime.utcnow)
    status = db.Column(db.Enum(OrderStatus), default=OrderStatus.PENDING)
    total_amount = db.Column(db.Float, nullable=False)
    shipping_address = db.Column(db.Text, nullable=False)
    shipping_city = db.Column(db.String(50), nullable=False)
    shipping_state = db.Column(db.String(50), nullable=False)
    shipping_postal_code = db.Column(db.String(20), nullable=False)
    shipping_latitude = db.Column(db.Float)
    shipping_longitude = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationships
    order_items = db.relationship('OrderItem', backref='order', lazy=True)
    shipments = db.relationship('Shipment', backref='order', lazy=True)

class OrderItem(db.Model):
    __tablename__ = 'order_items'

    id = db.Column(db.Integer, primary_key=True)
    order_id = db.Column(db.Integer, db.ForeignKey('orders.id'), nullable=False)
    product_id = db.Column(db.Integer, db.ForeignKey('products.id'), nullable=False)
    quantity = db.Column(db.Integer, nullable=False)
    unit_price = db.Column(db.Float, nullable=False)
    total_price = db.Column(db.Float, nullable=False)

class Shipment(db.Model):
    __tablename__ = 'shipments'

    id = db.Column(db.Integer, primary_key=True)
    order_id = db.Column(db.Integer, db.ForeignKey('orders.id'), nullable=False)
    warehouse_id = db.Column(db.Integer, db.ForeignKey('warehouses.id'), nullable=False)
    tracking_number = db.Column(db.String(100), unique=True)
    shipment_date = db.Column(db.DateTime)
    delivery_date = db.Column(db.DateTime)
    status = db.Column(db.String(50), default='preparing')
    shipping_cost = db.Column(db.Float, default=0.0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class InventoryMovement(db.Model):
    __tablename__ = 'inventory_movements'

    id = db.Column(db.Integer, primary_key=True)
    inventory_id = db.Column(db.Integer, db.ForeignKey('inventory.id'), nullable=False)
    movement_type = db.Column(db.String(50), nullable=False)  # in, out, transfer, adjustment
    quantity = db.Column(db.Integer, nullable=False)
    previous_stock = db.Column(db.Integer, nullable=False)
    new_stock = db.Column(db.Integer, nullable=False)
    reference_id = db.Column(db.String(100))  # order_id, shipment_id, etc.
    notes = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class DemandForecast(db.Model):
    __tablename__ = 'demand_forecasts'

    id = db.Column(db.Integer, primary_key=True)
    product_id = db.Column(db.Integer, db.ForeignKey('products.id'), nullable=False)
    warehouse_id = db.Column(db.Integer, db.ForeignKey('warehouses.id'), nullable=False)
    forecast_date = db.Column(db.Date, nullable=False)
    predicted_demand = db.Column(db.Integer, nullable=False)
    confidence_score = db.Column(db.Float)
    model_used = db.Column(db.String(50))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    __table_args__ = (db.UniqueConstraint('product_id', 'warehouse_id', 'forecast_date', name='unique_forecast'),)

class RouteOptimization(db.Model):
    __tablename__ = 'route_optimizations'

    id = db.Column(db.Integer, primary_key=True)
    optimization_date = db.Column(db.Date, nullable=False)
    total_distance = db.Column(db.Float, nullable=False)
    total_cost = db.Column(db.Float, nullable=False)
    vehicle_count = db.Column(db.Integer, nullable=False)
    optimization_status = db.Column(db.String(50), default='completed')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationships
    route_stops = db.relationship('RouteStop', backref='optimization', lazy=True)

class RouteStop(db.Model):
    __tablename__ = 'route_stops'

    id = db.Column(db.Integer, primary_key=True)
    optimization_id = db.Column(db.Integer, db.ForeignKey('route_optimizations.id'), nullable=False)
    warehouse_id = db.Column(db.Integer, db.ForeignKey('warehouses.id'), nullable=False)
    customer_id = db.Column(db.Integer, db.ForeignKey('customers.id'), nullable=False)
    stop_sequence = db.Column(db.Integer, nullable=False)
    arrival_time = db.Column(db.DateTime)
    departure_time = db.Column(db.DateTime)
    distance_from_previous = db.Column(db.Float)
    load_quantity = db.Column(db.Integer, default=0)

class Supplier(db.Model):
    __tablename__ = 'suppliers'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    code = db.Column(db.String(20), unique=True, nullable=False)
    contact_email = db.Column(db.String(100))
    contact_phone = db.Column(db.String(20))
    address = db.Column(db.Text)
    city = db.Column(db.String(50))
    state = db.Column(db.String(50))
    country = db.Column(db.String(50))
    postal_code = db.Column(db.String(20))
    latitude = db.Column(db.Float)
    longitude = db.Column(db.Float)
    # KPIs
    on_time_delivery_rate = db.Column(db.Float, default=0.0)  # Percentage (0-1)
    lead_time_variance = db.Column(db.Float, default=0.0)  # Days
    order_accuracy_rate = db.Column(db.Float, default=0.0)  # Percentage (0-1)
    quality_rating = db.Column(db.Float, default=0.0)  # 1-5 scale
    scorecard_score = db.Column(db.Float, default=0.0)  # Overall score (0-100)
    total_orders = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    inventory_levels = db.relationship('Inventory', backref='supplier', lazy=True)

class Alert(db.Model):
    __tablename__ = 'alerts'

    id = db.Column(db.Integer, primary_key=True)
    alert_type = db.Column(db.String(50), nullable=False)  # reorder, stockout, supplier_risk
    product_id = db.Column(db.Integer, db.ForeignKey('products.id'), nullable=False)
    warehouse_id = db.Column(db.Integer, db.ForeignKey('warehouses.id'), nullable=False)
    supplier_id = db.Column(db.Integer, db.ForeignKey('suppliers.id'), nullable=True)
    message = db.Column(db.Text, nullable=False)
    severity = db.Column(db.String(20), default='medium')  # low, medium, high, critical
    status = db.Column(db.String(20), default='active')  # active, acknowledged, resolved
    sent_at = db.Column(db.DateTime, nullable=True)
    acknowledged_at = db.Column(db.DateTime, nullable=True)
    resolved_at = db.Column(db.DateTime, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class User(db.Model):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    name = db.Column(db.String(100), nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    two_factor_secret = db.Column(db.String(32), nullable=True)
    two_factor_enabled = db.Column(db.Boolean, default=False)
    email_verified = db.Column(db.Boolean, default=False)
    account_locked = db.Column(db.Boolean, default=False)
    failed_login_attempts = db.Column(db.Integer, default=0)
    lockout_until = db.Column(db.DateTime, nullable=True)
    last_login = db.Column(db.DateTime, nullable=True)
    password_changed_at = db.Column(db.DateTime, default=datetime.utcnow)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def set_password(self, password):
        """Set password hash using passlib"""
        from passlib.hash import bcrypt
        self.password_hash = bcrypt.hash(password)

    def check_password(self, password):
        """Check password against hash"""
        if not self.password_hash:
            return False
        from passlib.hash import bcrypt
        try:
            return bcrypt.verify(password, self.password_hash)
        except ValueError:
            return False

    def generate_two_factor_secret(self):
        """Generate a random secret for 2FA"""
        self.two_factor_secret = secrets.token_hex(16)
        return self.two_factor_secret

    def is_account_locked(self):
        """Check if account is currently locked"""
        if not self.account_locked:
            return False
        if self.lockout_until and datetime.utcnow() > self.lockout_until:
            self.account_locked = False
            self.failed_login_attempts = 0
            return False
        return True

    def increment_failed_attempts(self):
        """Increment failed login attempts and lock account if necessary"""
        self.failed_login_attempts += 1
        if self.failed_login_attempts >= 5:  # Lock after 5 failed attempts
            self.account_locked = True
            from datetime import timedelta
            self.lockout_until = datetime.utcnow() + timedelta(minutes=15)
        return self.is_account_locked()

    def reset_failed_attempts(self):
        """Reset failed login attempts on successful login"""
        self.failed_login_attempts = 0
        self.account_locked = False
        self.lockout_until = None
        self.last_login = datetime.utcnow()

    def to_dict(self):
        """Convert user to dictionary for API responses"""
        return {
            'id': self.id,
            'email': self.email,
            'name': self.name,
            'two_factor_enabled': self.two_factor_enabled,
            'email_verified': self.email_verified,
            'last_login': self.last_login.isoformat() if self.last_login else None,
            'created_at': self.created_at.isoformat()
        }
