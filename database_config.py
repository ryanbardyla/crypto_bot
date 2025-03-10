import os

# Database configuration
DB_CONFIG = {
    'host': os.environ.get('POSTGRES_HOST', 'localhost'),
    'port': os.environ.get('POSTGRES_PORT', '5432'),
    'database': os.environ.get('POSTGRES_DB', 'trading_db'),
    'user': os.environ.get('POSTGRES_USER', 'bot_user'),
    'password': os.environ.get('POSTGRES_PASSWORD', 'secure_password')
}

# Connection string for SQLAlchemy
DB_URI = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"

# For backward compatibility
SQLITE_DB_URI = "sqlite:///sentiment_database.db"

def get_db_uri():
    """
    Returns the database URI based on environment settings
    
    Returns:
    str: SQLAlchemy connection string
    """
    use_sqlite = os.environ.get('USE_SQLITE', 'False').lower() == 'true'
    if use_sqlite:
        return SQLITE_DB_URI
    return DB_URI