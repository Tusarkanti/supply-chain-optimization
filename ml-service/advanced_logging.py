import logging
import logging.handlers
import json
import os
from datetime import datetime
from typing import Dict, Any
import sys

class AdvancedLogger:
    def __init__(self, log_dir: str = 'logs'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # Create formatters
        self.detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )

        self.json_formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}'
        )

        # Setup loggers
        self.setup_main_logger()
        self.setup_performance_logger()
        self.setup_security_logger()
        self.setup_ml_logger()

    def setup_main_logger(self):
        """Setup main application logger"""
        self.main_logger = logging.getLogger('supply_chain_main')
        self.main_logger.setLevel(logging.INFO)

        # File handler with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            os.path.join(self.log_dir, 'main.log'),
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setFormatter(self.detailed_formatter)
        self.main_logger.addHandler(file_handler)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(self.detailed_formatter)
        self.main_logger.addHandler(console_handler)

    def setup_performance_logger(self):
        """Setup performance monitoring logger"""
        self.perf_logger = logging.getLogger('supply_chain_performance')
        self.perf_logger.setLevel(logging.INFO)

        # JSON formatted file handler for performance metrics
        perf_handler = logging.FileHandler(os.path.join(self.log_dir, 'performance.json'))
        perf_handler.setFormatter(self.json_formatter)
        self.perf_logger.addHandler(perf_handler)

    def setup_security_logger(self):
        """Setup security events logger"""
        self.security_logger = logging.getLogger('supply_chain_security')
        self.security_logger.setLevel(logging.WARNING)

        # Separate file for security events
        security_handler = logging.FileHandler(os.path.join(self.log_dir, 'security.log'))
        security_handler.setFormatter(self.detailed_formatter)
        self.security_logger.addHandler(security_handler)

    def setup_ml_logger(self):
        """Setup ML-specific logger"""
        self.ml_logger = logging.getLogger('supply_chain_ml')
        self.ml_logger.setLevel(logging.INFO)

        # ML events handler
        ml_handler = logging.FileHandler(os.path.join(self.log_dir, 'ml_events.log'))
        ml_handler.setFormatter(self.detailed_formatter)
        self.ml_logger.addHandler(ml_handler)

    def log_prediction(self, model_name: str, prediction_data: Dict[str, Any]):
        """Log ML prediction events"""
        log_entry = {
            'event_type': 'prediction',
            'model': model_name,
            'timestamp': datetime.now().isoformat(),
            'input_data': prediction_data
        }
        self.ml_logger.info(f"Prediction made: {json.dumps(log_entry)}")

    def log_performance_metric(self, metric_name: str, value: float, metadata: Dict[str, Any] = None):
        """Log performance metrics"""
        log_entry = {
            'metric': metric_name,
            'value': value,
            'metadata': metadata or {}
        }
        self.perf_logger.info(json.dumps(log_entry))

    def log_security_event(self, event_type: str, details: Dict[str, Any]):
        """Log security events"""
        log_entry = {
            'event_type': event_type,
            'timestamp': datetime.now().isoformat(),
            'details': details
        }
        self.security_logger.warning(f"Security event: {json.dumps(log_entry)}")

    def log_error(self, error: Exception, context: str = None):
        """Log errors with context"""
        error_details = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context,
            'traceback': str(error.__traceback__) if error.__traceback__ else None
        }
        self.main_logger.error(f"Error occurred: {json.dumps(error_details)}")

    def get_recent_logs(self, logger_name: str, lines: int = 100):
        """Get recent log entries from a specific logger"""
        log_file = os.path.join(self.log_dir, f'{logger_name}.log')
        if not os.path.exists(log_file):
            return []

        with open(log_file, 'r') as f:
            return f.readlines()[-lines:]

    def cleanup_old_logs(self, days_to_keep: int = 30):
        """Clean up log files older than specified days"""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)

        for filename in os.listdir(self.log_dir):
            if filename.endswith('.log'):
                file_path = os.path.join(self.log_dir, filename)
                file_modified = datetime.fromtimestamp(os.path.getmtime(file_path))

                if file_modified < cutoff_date:
                    os.remove(file_path)
                    self.main_logger.info(f"Removed old log file: {filename}")

# Global logger instance
logger = AdvancedLogger()
