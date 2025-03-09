import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from datetime import datetime, timedelta
import seaborn as sns

def analyze_sentiment_data(db_path="sqlite:///sentiment_database.db", json_dir="sentiment_data"):
    """
    Analyze sentiment data quality from both the database and JSON files
    """
    print(f"Analyzing sentiment data at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Connect to database
    try:
        engine = create_engine(db_path)
        db_data = pd.read_sql("SELECT * FROM sentiment_youtube", engine)
        print(f"Successfully loaded {len(db_data)} records from database")
    except Exception as e:
        print(f"Error loading database data: {str(e)}")
        db_data = pd.DataFrame()
    
    # Load JSON files
    json_data = []
    if os.path.exists(json_dir):
        json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
        print(f"Found {len(json_files)} JSON files in {json_dir}")
        
        for filename in json_files[:100]:  # Limit to 100 files for performance
            try:
                with open(os.path.join(json_dir, filename), 'r') as f:
                    data = json.load(f)
                    json_data.append(data)
            except Exception as e:
                print(f"Error reading {filename}: {str(e)}")
    else:
        print(f"JSON directory {json_dir} not found")
    
    # Create dataframe from JSON data
    if json_data:
        json_df = pd.DataFrame(json_data)
        print(f"Loaded {len(json_df)} records from JSON files")
    else:
        json_df = pd.DataFrame()
    
    # Analyze database data
    if not db_data.empty:
        print("\n--- Database Data Analysis ---")
        
        # Time range
        if 'processed_date' in db_data.columns:
            db_data['processed_date'] = pd.to_datetime(db_data['processed_date'])
            earliest = db_data['processed_date'].min()
            latest = db_data['processed_date'].max()
            print(f"Time range: {earliest} to {latest}")
            
            # Data by day
            db_data['day'] = db_data['processed_date'].dt.date
            daily_counts = db_data.groupby('day').size()
            print(f"Average videos processed per day: {daily_counts.mean():.2f}")
        
        # Channel distribution
        if 'channel_id' in db_data.columns:
            channel_counts = db_data['channel_id'].value_counts()
            print(f"Number of unique channels: {len(channel_counts)}")
            print("Top 5 channels by video count:")
            for channel, count in channel_counts.head(5).items():
                print(f"  - {channel}: {count} videos")
        
        # Sentiment score stats
        if 'combined_score' in db_data.columns:
            print(f"\nSentiment score statistics:")
            print(f"  Min: {db_data['combined_score'].min():.2f}")
            print(f"  Max: {db_data['combined_score'].max():.2f}")
            print(f"  Mean: {db_data['combined_score'].mean():.2f}")
            print(f"  Median: {db_data['combined_score'].median():.2f}")
            
            # Sentiment distribution
            plt.figure(figsize=(10, 6))
            sns.histplot(db_data['combined_score'], kde=True)
            plt.title('Distribution of Sentiment Scores')
            plt.xlabel('Sentiment Score')
            plt.ylabel('Count')
            plt.savefig('sentiment_distribution.png')
            print(f"Saved sentiment distribution chart to sentiment_distribution.png")
        
        # Text length analysis
        if 'text_length' in db_data.columns:
            print(f"\nTranscript length statistics:")
            print(f"  Min: {db_data['text_length'].min()} characters")
            print(f"  Max: {db_data['text_length'].max()} characters")
            print(f"  Mean: {db_data['text_length'].mean():.2f} characters")
            print(f"  Median: {db_data['text_length'].median()} characters")
            
            # Find videos with suspiciously short transcripts
            short_transcripts = db_data[db_data['text_length'] < 100]
            print(f"Videos with very short transcripts (<100 chars): {len(short_transcripts)}")
    
    # Cross-check database and JSON files
    if not db_data.empty and not json_df.empty and 'video_id' in db_data.columns and 'video_id' in json_df.columns:
        db_ids = set(db_data['video_id'])
        json_ids = set(json_df['video_id'])
        
        only_in_db = db_ids - json_ids
        only_in_json = json_ids - db_ids
        
        print(f"\n--- Data Consistency Check ---")
        print(f"Records in DB but not JSON: {len(only_in_db)}")
        print(f"Records in JSON but not DB: {len(only_in_json)}")
    
    # Generate time series plot of sentiment
    if not db_data.empty and 'processed_date' in db_data.columns and 'combined_score' in db_data.columns:
        plt.figure(figsize=(12, 6))
        
        # Group by day and calculate average sentiment
        daily_sentiment = db_data.groupby('day')['combined_score'].mean().reset_index()
        daily_sentiment['day'] = pd.to_datetime(daily_sentiment['day'])
        
        # Plot
        plt.plot(daily_sentiment['day'], daily_sentiment['combined_score'], marker='o')
        plt.title('Average Daily Sentiment Score')
        plt.xlabel('Date')
        plt.ylabel('Sentiment Score')
        plt.grid(True, alpha=0.3)
        plt.savefig('daily_sentiment.png')
        print(f"Saved daily sentiment chart to daily_sentiment.png")
    
    print("\nData quality analysis complete!")

if __name__ == "__main__":
    analyze_sentiment_data()