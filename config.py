import os

class Config:
    """
    Base configuration class for the Flask application.
    """
    SECRET_KEY = os.environ.get('SECRET_KEY', 'your_secret_key_here')
    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', 'uploads')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max upload size

class DevelopmentConfig(Config):
    """
    Development configuration with debugging enabled.
    """
    DEBUG = True
    FLASK_ENV = 'development'

class ProductionConfig(Config):
    """
    Production configuration with debugging disabled.
    """
    DEBUG = False
    FLASK_ENV = 'production'

# Use the appropriate configuration based on the environment
def get_config(environment='development'):
    if environment == 'production':
        return ProductionConfig
    return DevelopmentConfig

