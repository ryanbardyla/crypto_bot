# db_migration.py
import os
import sqlite3
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DBMigration")

def migrate_database():
    """
    Update the SQLite database schema to add the mentioned_cryptos column
    """
    db_file = "sentiment_database.db"
    
    if not os.path.exists(db_file):
        logger.error(f"Database file {db_file} not found")
        return False
    
    logger.info(f"Migrating database {db_file}")
    
    try:
        # Connect to the database
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        
        # Check if the column already exists
        cursor.execute("PRAGMA table_info(sentiment_youtube)")
        columns = [col[1] for col in cursor.fetchall()]
        
        if "mentioned_cryptos" not in columns:
            logger.info("Adding column 'mentioned_cryptos' to sentiment_youtube table")
            
            # Add the new column
            cursor.execute("ALTER TABLE sentiment_youtube ADD COLUMN mentioned_cryptos TEXT")
            
            # Commit the changes
            conn.commit()
            logger.info("Migration successful!")
        else:
            logger.info("Column 'mentioned_cryptos' already exists")
        
        # Close the connection
        conn.close()
        return True
    
    except Exception as e:
        logger.error(f"Error migrating database: {str(e)}")
        return False

if __name__ == "__main__":
    migrate_database()