import os
import json
import argparse
import pandas as pd
from tabulate import tabulate
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from database_manager import DatabaseManager, SentimentRecord

# Import the centralized logging configuration
from utils.logging_config import get_module_logger

# Get logger for this module
logger = get_module_logger(__name__)


def list_records(db_manager, record_type=None, days=None, limit=10):
    """List sentiment records."""
    records = db_manager.get_sentiment_data(record_type=record_type, days=days, limit=limit)
    
    if not records:
        print("No records found matching criteria")
        return
    
    # Convert to DataFrame for better display
    df = pd.DataFrame(records)
    
    # Extract VADER sentiment from nested dict
    if 'vader_sentiment' in df.columns:
        for component in ['neg', 'neu', 'pos', 'compound']:
            df[f'vader_{component}'] = df['vader_sentiment'].apply(lambda x: x.get(component, 0) if isinstance(x, dict) else 0)
        df.drop('vader_sentiment', axis=1, inplace=True)
    
    # Format dates
    for col in ['processed_date', 'publish_date']:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: x.strftime('%Y-%m-%d %H:%M') if x else None)
    
    # Select columns to display
    display_cols = ['record_id', 'record_type', 'source', 'processed_date', 
                   'combined_score', 'bullish_keywords', 'bearish_keywords']
    
    # Add title if available
    if 'title' in df.columns:
        display_cols.insert(3, 'title')
    
    # Display results
    print(tabulate(df[display_cols].head(limit), headers='keys', tablefmt='psql'))
    print(f"Showing {len(df[:limit])} of {len(df)} records")

def show_record(db_manager, record_id):
    """Show detailed information for a specific record."""
    records = db_manager.get_sentiment_data(record_id=record_id)
    
    if not records:
        print(f"No record found with ID: {record_id}")
        return
    
    record = records[0]
    
    # Format dates
    for key in ['processed_date', 'publish_date']:
        if key in record and record[key]:
            if isinstance(record[key], datetime):
                record[key] = record[key].strftime('%Y-%m-%d %H:%M:%S')
    
    # Print record details
    print("\n=== Record Details ===")
    print(f"Record ID: {record['record_id']}")
    print(f"Record Type: {record['record_type']}")
    print(f"Source: {record['source']}")
    
    if 'title' in record and record['title']:
        print(f"Title: {record['title']}")
    
    if 'username' in record and record['username']:
        print(f"Username: {record['username']}")
    
    if 'channel_id' in record and record['channel_id']:
        print(f"Channel ID: {record['channel_id']}")
    
    if 'publish_date' in record and record['publish_date']:
        print(f"Publish Date: {record['publish_date']}")
    
    print(f"Processed Date: {record['processed_date']}")
    
    print("\n=== Sentiment Analysis ===")
    if 'vader_sentiment' in record:
        print("VADER Sentiment:")
        for component, value in record['vader_sentiment'].items():
            print(f"  - {component}: {value:.4f}")
    
    print(f"Bullish Keywords: {record['bullish_keywords']}")
    print(f"Bearish Keywords: {record['bearish_keywords']}")
    print(f"Keyword Sentiment: {record['keyword_sentiment']:.4f}")
    print(f"Combined Score: {record['combined_score']:.4f}")
    
    if 'text_length' in record:
        print(f"Text Length: {record['text_length']} characters")

def show_stats(db_manager, days=30):
    """Show sentiment database statistics."""
    from sqlalchemy import func, text
    
    session = db_manager.get_session()
    
    try:
        # Get overall counts
        from database_manager import SentimentRecord
        total_count = session.query(func.count(SentimentRecord.id)).scalar()
        youtube_count = session.query(func.count(SentimentRecord.id)).filter(SentimentRecord.record_type == 'youtube').scalar()
        twitter_count = session.query(func.count(SentimentRecord.id)).filter(SentimentRecord.record_type == 'twitter').scalar()
        
        # Get date range
        oldest = session.query(func.min(SentimentRecord.processed_date)).scalar()
        newest = session.query(func.max(SentimentRecord.processed_date)).scalar()
        
        # Get recent counts
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_count = session.query(func.count(SentimentRecord.id)).filter(SentimentRecord.processed_date >= cutoff_date).scalar()
        
        # Get sentiment distribution
        query = """
        SELECT 
            CASE 
                WHEN combined_score < -5 THEN 'Very Negative (-10 to -5)'
                WHEN combined_score < 0 THEN 'Negative (-5 to 0)'
                WHEN combined_score < 5 THEN 'Positive (0 to 5)'
                ELSE 'Very Positive (5 to 10)'
            END as sentiment_range,
            COUNT(*) as count,
            AVG(combined_score) as avg_score
        FROM sentiment_records
        GROUP BY sentiment_range
        ORDER BY avg_score
        """
        
        distribution = session.execute(text(query)).fetchall()
        
        # Get recent sentiment trend
        trend_query = """
        SELECT 
            date(processed_date) as date,
            AVG(combined_score) as avg_score,
            COUNT(*) as count
        FROM sentiment_records
        WHERE processed_date >= :cutoff_date
        GROUP BY date(processed_date)
        ORDER BY date(processed_date)
        """
        
        trend = session.execute(text(trend_query), {"cutoff_date": cutoff_date}).fetchall()
        
        # Print statistics
        print("\n=== Sentiment Database Statistics ===")
        print(f"Total Records: {total_count}")
        print(f"YouTube Records: {youtube_count}")
        print(f"Twitter Records: {twitter_count}")
        print(f"Date Range: {oldest.strftime('%Y-%m-%d') if oldest else 'N/A'} to {newest.strftime('%Y-%m-%d') if newest else 'N/A'}")
        print(f"Records in last {days} days: {recent_count}")
        
        print("\n=== Sentiment Distribution ===")
        for row in distribution:
            print(f"{row[0]}: {row[1]} records (Avg: {row[2]:.2f})")
        
        print(f"\n=== Recent {days}-Day Sentiment Trend ===")
        trend_df = pd.DataFrame(trend, columns=['date', 'avg_score', 'count'])
        if not trend_df.empty:
            trend_df['date'] = pd.to_datetime(trend_df['date'])
            trend_df = trend_df.sort_values('date')
            
            # Print trend
            for _, row in trend_df.iterrows():
                print(f"{row['date'].strftime('%Y-%m-%d')}: {row['avg_score']:.2f} (from {row['count']} records)")
        else:
            print("No trend data available")
            
    except Exception as e:
        print(f"Error retrieving statistics: {e}")
    finally:
        session.close()

def export_data(db_manager, output_file, record_type=None, days=None):
    """Export sentiment data to CSV or JSON file."""
    records = db_manager.get_sentiment_data(record_type=record_type, days=days)
    
    if not records:
        print("No records found matching criteria")
        return
    
    # Determine export format from file extension
    file_ext = os.path.splitext(output_file)[1].lower()
    
    try:
        if file_ext == '.csv':
            # Convert to DataFrame for CSV export
            df = pd.DataFrame(records)
            
            # Handle nested dictionaries (vader_sentiment)
            if 'vader_sentiment' in df.columns:
                for component in ['neg', 'neu', 'pos', 'compound']:
                    df[f'vader_{component}'] = df['vader_sentiment'].apply(lambda x: x.get(component, 0) if isinstance(x, dict) else 0)
                df.drop('vader_sentiment', axis=1, inplace=True)
            
            # Convert dates to strings
            for col in df.columns:
                if df[col].dtype == 'datetime64[ns]' or isinstance(df.iloc[0][col], datetime):
                    df[col] = df[col].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S') if x else None)
            
            df.to_csv(output_file, index=False)
            print(f"Exported {len(df)} records to CSV: {output_file}")
            
        elif file_ext == '.json':
            # Convert datetime objects to strings for JSON serialization
            for record in records:
                for key, value in record.items():
                    if isinstance(value, datetime):
                        record[key] = value.strftime('%Y-%m-%d %H:%M:%S')
            
            with open(output_file, 'w') as f:
                json.dump(records, f, indent=2)
            
            print(f"Exported {len(records)} records to JSON: {output_file}")
            
        else:
            print(f"Unsupported file format: {file_ext}")
            print("Supported formats: .csv, .json")
    
    except Exception as e:
        print(f"Error exporting data: {e}")

def main():
    parser = argparse.ArgumentParser(description="Sentiment Database Administration Tool")
    parser.add_argument("--db", type=str, default="sqlite:///sentiment_database.db", help="Database connection string")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # List records command
    list_parser = subparsers.add_parser("list", help="List sentiment records")
    list_parser.add_argument("--type", choices=["youtube", "twitter"], help="Filter by record type")
    list_parser.add_argument("--days", type=int, help="Filter by days back")
    list_parser.add_argument("--limit", type=int, default=10, help="Limit number of records")
    
    # Show record command
    show_parser = subparsers.add_parser("show", help="Show details for a specific record")
    show_parser.add_argument("record_id", help="Record ID to show")
    
    # Statistics command
    stats_parser = subparsers.add_parser("stats", help="Show database statistics")
    stats_parser.add_argument("--days", type=int, default=30, help="Days to include in trend")
    
    # Export data command
    export_parser = subparsers.add_parser("export", help="Export data to CSV or JSON")
    export_parser.add_argument("output_file", help="Output file path (.csv or .json)")
    export_parser.add_argument("--type", choices=["youtube", "twitter"], help="Filter by record type")
    export_parser.add_argument("--days", type=int, help="Filter by days back")
    
    # Optimize database command
    optimize_parser = subparsers.add_parser("optimize", help="Optimize database")
    optimize_parser.add_argument("--cleanup-days", type=int, default=90, help="Remove records older than days")
    
    args = parser.parse_args()
    
    # Initialize database manager
    db_manager = DatabaseManager(args.db)
    
    # Execute command
    if args.command == "list":
        list_records(db_manager, record_type=args.type, days=args.days, limit=args.limit)
    
    elif args.command == "show":
        show_record(db_manager, args.record_id)
    
    elif args.command == "stats":
        show_stats(db_manager, days=args.days)
    
    elif args.command == "export":
        export_data(db_manager, args.output_file, record_type=args.type, days=args.days)
    
    elif args.command == "optimize":
        # Vacuum database
        print("Optimizing database...")
        db_manager.vacuum_database()
        
        # Clean up old records if specified
        if args.cleanup_days:
            deleted = db_manager.cleanup_old_records(days=args.cleanup_days)
            print(f"Cleaned up {deleted} records older than {args.cleanup_days} days")
        
        print("Database optimization complete")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()