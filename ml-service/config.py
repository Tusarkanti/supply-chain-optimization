import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Base configuration"""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    DEBUG = False
    TESTING = False

    # Database
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///supply_chain.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {}

    # JWT
    JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY') or 'jwt-secret-key-change-in-production'
    JWT_ACCESS_TOKEN_EXPIRES = 3600  # 1 hour

    # ML Model Settings
    MODEL_UPDATE_FREQUENCY = int(os.environ.get('MODEL_UPDATE_FREQUENCY', 24))  # hours
    ANOMALY_DETECTION_THRESHOLD = float(os.environ.get('ANOMALY_DETECTION_THRESHOLD', 0.7))
    FORECAST_HORIZON = int(os.environ.get('FORECAST_HORIZON', 30))  # days

    # Optimization Settings
    MAX_VEHICLES = int(os.environ.get('MAX_VEHICLES', 10))
    MAX_DISTANCE_PER_VEHICLE = float(os.environ.get('MAX_DISTANCE_PER_VEHICLE', 500.0))  # km
    AVERAGE_VEHICLE_SPEED = float(os.environ.get('AVERAGE_VEHICLE_SPEED', 60.0))  # km/h

    # API Settings
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

    # Logging
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_DIR = os.environ.get('LOG_DIR', 'logs')

    # File Uploads
    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', 'uploads')
    MAX_FILE_SIZE = int(os.environ.get('MAX_FILE_SIZE', 16 * 1024 * 1024))  # 16MB

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = os.environ.get('DEV_DATABASE_URL') or 'sqlite:///../instance/supply_chain_dev.db'

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = os.environ.get('TEST_DATABASE_URL') or 'postgresql://user:password@localhost/supply_chain_test'
    WTF_CSRF_ENABLED = False

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False

    # Use environment variables for production
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL')
    SECRET_KEY = os.environ.get('SECRET_KEY')
    JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY')

    def __init__(self):
        super().__init__()
        if not self.SQLALCHEMY_DATABASE_URI or not self.SECRET_KEY or not self.JWT_SECRET_KEY:
            raise ValueError("Production requires DATABASE_URL, SECRET_KEY, and JWT_SECRET_KEY environment variables")

# Configuration mapping
config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}

def get_config(config_name=None):
    """Get configuration object"""
    if config_name is None:
        config_name = os.environ.get('FLASK_ENV', 'development')

    return config.get(config_name, config['default'])()
