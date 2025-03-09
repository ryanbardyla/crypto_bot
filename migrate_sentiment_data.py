# migrate_sentiment_data.py

import os
import json
import logging
import argparse
from tqdm import tqdm
from database_manager import DatabaseManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("migration.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DataMigration")

def migrate_sentiment_data(json_dir="sentiment_data", db_path="sqlite:///sentiment_database.db", batch_size=100):
    """
    Migrate sentiment data from JSON files to SQLite database.
    
    Args:
        json_dir (str): Directory containing JSON files
        db_path (str): SQLite database path
        batch_size (int): Number of records to process in a batch
    
    Returns:
        tuple: (success_count, failed_count)
    """
    db_manager = DatabaseManager(db_path)
    
    # Check if directory exists
    if not os.path.exists(json_dir):
        logger.error(f"Directory {json_dir} does not exist")
        return 0, 0
    
    # Get list of JSON files
    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    total_files = len(json_files)
    
    if total_files == 0:
        logger.info("No JSON files found for migration")
        return 0, 0
    
    logger.info(f"Found {total_files} JSON files to migrate")
    
    success_count = 0
    failed_count = 0
    
    # Use tqdm for progress bar
    for file_name in tqdm(json_files, desc="Migrating files"):
        file_path = os.path.join(json_dir, file_name)
        
        try:
            # Load JSON data
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Determine record type and ID
            record_id = file_name.replace('.json', '')
            
            # Figure out if it's a YouTube or Twitter record
            if 'video_id' in data or record_id.startswith('youtube-'):
                record_type = 'youtube'
                if 'video_id' in data:
                    record_id = data['video_id']
                elif record_id.startswith('youtube-'):
                    record_id = record_id.replace('youtube-', '')
            else:
                record_type = 'twitter'
                if 'tweet_id' in data:
                    record_id = data['tweet_id']
            
            # Save to database
            if db_manager.save_sentiment_data(data, record_type, record_id):
                success_count += 1
            else:
                failed_count += 1
                
        except Exception as e:
            logger.error(f"Error migrating {file_name}: {str(e)}")
            failed_count += 1
        
        # Commit in batches to improve performance
        if (success_count + failed_count) % batch_size == 0:
            logger.info(f"Progress: {success_count} successful, {failed_count} failed out of {total_files}")
    
    # Final report
    logger.info(f"Migration completed: {success_count} successful, {failed_count} failed out of {total_files}")
    
    return success_count, failed_count

def verify_migration(json_dir="sentiment_data", db_path="sqlite:///sentiment_database.db"):
    """
    Verify that all JSON files have been migrated to the database.
    
    Args:
        json_dir (str): Directory containing JSON files
        db_path (str): SQLite database path
    
    Returns:
        list: List of files that failed migration
    """
    db_manager = DatabaseManager(db_path)
    
    # Check if directory exists
    if not os.path.exists(json_dir):
        logger.error(f"Directory {json_dir} does not exist")
        return []
    
    # Get list of JSON files
    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    total_files = len(json_files)
    
    if total_files == 0:
        logger.info("No JSON files found for verification")
        return []
    
    logger.info(f"Verifying migration of {total_files} JSON files")
    
    failed_files = []
    
    for file_name in tqdm(json_files, desc="Verifying files"):
        record_id = file_name.replace('.json', '')
        
        # Load JSON file to check record type
        try:
            with open(os.path.join(json_dir, file_name), 'r') as f:
                data = json.load(f)
                
            # Determine record type
            if 'video_id' in data or record_id.startswith('youtube-'):
                record_type = 'youtube'
                if 'video_id' in data:
                    record_id = data['video_id']
                elif record_id.startswith('youtube-'):
                    record_id = record_id.replace('youtube-', '')
            else:
                record_type = 'twitter'
                if 'tweet_id' in data:
                    record_id = data['tweet_id']
            
            # Check if record exists in database
            records = db_manager.get_sentiment_data(record_type=record_type, record_id=record_id)
            
            if not records:
                failed_files.append(file_name)
                
        except Exception as e:
            logger.error(f"Error verifying {file_name}: {str(e)}")
            failed_files.append(file_name)
    
    # Report results
    if failed_files:
        logger.warning(f"Verification found {len(failed_files)} files not properly migrated")
    else:
        logger.info("All files successfully migrated")
    
    return failed_files

def backup_json_files(json_dir="sentiment_data", backup_dir="sentiment_data_backup"):
    """
    Create a backup of JSON files before deletion.
    
    Args:
        json_dir (str): Source directory containing JSON files
        backup_dir (str): Destination directory for backup
    
    Returns:
        int: Number of files backed up
    """
    import shutil
    
    # Check if source directory exists
    if not os.path.exists(json_dir):
        logger.error(f"Source directory {json_dir} does not exist")
        return 0
    
    # Create backup directory if it doesn't exist
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
        logger.info(f"Created backup directory {backup_dir}")
    
    # Count JSON files
    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    total_files = len(json_files)
    
    if total_files == 0:
        logger.info("No JSON files found for backup")
        return 0
    
    logger.info(f"Backing up {total_files} JSON files to {backup_dir}")
    
    # Copy files
    for file_name in tqdm(json_files, desc="Backing up files"):
        source_path = os.path.join(json_dir, file_name)
        destination_path = os.path.join(backup_dir, file_name)
        
        try:
            shutil.copy2(source_path, destination_path)
        except Exception as e:
            logger.error(f"Error backing up {file_name}: {str(e)}")
    
    logger.info(f"Backup completed: {total_files} files backed up to {backup_dir}")
    
    return total_files

def clean_up_json_files(json_dir="sentiment_data"):
    """
    Delete JSON files after successful migration.
    
    Args:
        json_dir (str): Directory containing JSON files
    
    Returns:
        int: Number of files deleted
    """
    # Check if directory exists
    if not os.path.exists(json_dir):
        logger.error(f"Directory {json_dir} does not exist")
        return 0
    
    # Count JSON files
    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    total_files = len(json_files)
    
    if total_files == 0:
        logger.info("No JSON files found for cleanup")
        return 0
    
    logger.info(f"Cleaning up {total_files} JSON files from {json_dir}")
    
    # Delete files
    deleted_count = 0
    for file_name in tqdm(json_files, desc="Deleting files"):
        file_path = os.path.join(json_dir, file_name)
        
        try:
            os.remove(file_path)
            deleted_count += 1
        except Exception as e:
            logger.error(f"Error deleting {file_name}: {str(e)}")
    
    logger.info(f"Cleanup completed: {deleted_count} files deleted from {json_dir}")
    
    return deleted_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate sentiment data from JSON files to SQLite database")
    parser.add_argument("--source", type=str, default="sentiment_data", help="Source directory containing JSON files")
    parser.add_argument("--db", type=str, default="sqlite:///sentiment_database.db", help="SQLite database path")
    parser.add_argument("--batch-size", type=int, default=100, help="Number of records to process in a batch")
    parser.add_argument("--verify", action="store_true", help="Verify migration after completion")
    parser.add_argument("--backup", action="store_true", help="Backup JSON files before migration")
    parser.add_argument("--cleanup", action="store_true", help="Delete JSON files after successful migration")
    parser.add_argument("--backup-dir", type=str, default="sentiment_data_backup", help="Backup directory")
    
    args = parser.parse_args()
    
    print(f"Starting migration from {args.source} to {args.db}")
    
    # Backup if requested
    if args.backup:
        backup_count = backup_json_files(args.source, args.backup_dir)
        print(f"Backed up {backup_count} files to {args.backup_dir}")
    
    # Perform migration
    success_count, failed_count = migrate_sentiment_data(args.source, args.db, args.batch_size)
    print(f"Migration completed: {success_count} successful, {failed_count} failed")
    
    # Verify if requested
    if args.verify:
        failed_files = verify_migration(args.source, args.db)
        if failed_files:
            print(f"Verification found {len(failed_files)} files not properly migrated:")
            for file in failed_files[:10]:  # Show first 10
                print(f"  - {file}")
            if len(failed_files) > 10:
                print(f"  ... and {len(failed_files) - 10} more")
        else:
            print("Verification successful: All files properly migrated")
    
    # Clean up if requested and migration was successful
    if args.cleanup and failed_count == 0:
        deleted_count = clean_up_json_files(args.source)
        print(f"Cleaned up {deleted_count} JSON files")
    elif args.cleanup and failed_count > 0:
        print("Skipping cleanup due to migration failures")
    
    print("Migration process completed")