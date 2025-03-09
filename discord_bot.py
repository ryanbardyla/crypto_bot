# discord_bot.py (modified to use token from .env)

import os
import re
import json
import time
import logging
import discord
import asyncio
from discord.ext import commands
from dotenv import load_dotenv
from datetime import datetime
from youtube_transcript_api import YouTubeTranscriptApi
from sentiment_analyzer import SentimentAnalyzer
from simple_backtester import SimpleBacktester

# Import the centralized logging configuration
from utils.logging_config import get_module_logger

# Load environment variables
load_dotenv()

# Get logger for this module
logger = get_module_logger(__name__)

# Set up Discord bot
intents = discord.Intents.default()
bot = commands.Bot(command_prefix='!', intents=intents)

# Initialize sentiment analyzer
sentiment_analyzer = SentimentAnalyzer()

# Define file paths
sentiment_dir = "sentiment_data"

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("discord_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DiscordBot")

# Discord setup
intents = discord.Intents.default()
bot = commands.Bot(command_prefix='!', intents=intents)
sentiment_analyzer = SentimentAnalyzer()

# YouTube regex pattern
youtube_regex = r'(?:youtube\.com\/(?:[^\/]+\/.+\/|(?:v|e(?:mbed)?)\/|.*[?&]v=)|youtu\.be\/)([^"&?\/\s]{11})'

# Sentiment data directory
sentiment_dir = "sentiment_data"

# Help message
help_message = """
**Crypto Sentiment Bot Commands**

`!analyze [youtube_url]` - Analyze the sentiment of a YouTube video
`!status` - Show current status of sentiment data
`!backtest [symbol]` - Run a backtest using sentiment data (default: BTC)
`!commands` - Show this help message
"""

@bot.event
async def on_ready():
    logger.info(f'{bot.user} has connected to Discord!')

@bot.command(name='analyze')
async def analyze_video(ctx, url: str):
    await ctx.send(f"Analyzing YouTube video: {url}")
    try:
        video_id = extract_youtube_id(url)
        if not video_id:
            await ctx.send("Could not extract YouTube video ID. Please provide a valid YouTube URL.")
            return
            
        await ctx.send("Fetching transcript...")
        transcript = get_youtube_transcript(video_id)
        if not transcript:
            await ctx.send("Could not retrieve transcript. The video might not have subtitles or is unavailable.")
            return
            
        await ctx.send("Analyzing sentiment...")
        sentiment = sentiment_analyzer.analyze_text(transcript, source=f"youtube-{video_id}")
        
        # Format result message
        result_message = f"**Sentiment Analysis Results**\n"
        result_message += f"Bullish keywords: {sentiment.get('bullish_keywords', 0)}\n"
        result_message += f"Bearish keywords: {sentiment.get('bearish_keywords', 0)}\n"
        result_message += f"Combined sentiment score: {sentiment.get('combined_score', 0):.2f} (-10 to +10 scale)\n"
        
        if sentiment.get('combined_score', 0) > 3:
            result_message += "🟢 Overall sentiment: **Bullish**"
        elif sentiment.get('combined_score', 0) < -3:
            result_message += "🔴 Overall sentiment: **Bearish**"
        else:
            result_message += "🟡 Overall sentiment: **Neutral**"
            
        await ctx.send(result_message)
        
        # Save sentiment data
        save_sentiment_data(video_id, sentiment)
        
        # Ask if user wants to run a backtest
        await ctx.send("Would you like to run a backtest using this sentiment data? (yes/no)")
        try:
            def check(m):
                return m.author == ctx.author and m.channel == ctx.channel and m.content.lower() in ['yes', 'no', 'y', 'n']
                
            msg = await bot.wait_for('message', check=check, timeout=30.0)
            if msg.content.lower() in ['yes', 'y']:
                await ctx.send("Starting backtest...")
                await run_backtest(ctx, sentiment)
            else:
                await ctx.send("Backtest cancelled.")
        except asyncio.TimeoutError:
            await ctx.send("Timed out waiting for a response. Backtest cancelled.")
            
    except Exception as e:
        logger.error(f"Error analyzing video: {str(e)}")
        await ctx.send(f"Error analyzing video: {str(e)}")

@bot.command(name='commands')
async def commands_help(ctx):
    await ctx.send(help_message)

@bot.command(name='status')
async def status(ctx):
    try:
        sentiment_data = load_all_sentiment_data()
        if not sentiment_data:
            await ctx.send("No sentiment data available. Use `!analyze [youtube_url]` to analyze videos.")
            return
            
        status_message = "**Sentiment Analysis Status**\n\n"
        status_message += f"Analyzed videos: {len(sentiment_data)}\n"
        status_message += "\n**Recent Analysis:**\n"
        
        for i, data in enumerate(sentiment_data[-3:]):
            source = data.get('source', 'Unknown')
            score = data.get('combined_score', 0)
            bullish = data.get('bullish_keywords', 0)
            bearish = data.get('bearish_keywords', 0)
            
            status_message += f"{i+1}. Source: {source}\n"
            status_message += f"   Score: {score:.2f} (Bullish: {bullish}, Bearish: {bearish})\n"
            
        await ctx.send(status_message)
    except Exception as e:
        logger.error(f"Error getting status: {str(e)}")
        await ctx.send(f"Error getting status: {str(e)}")

@bot.command(name='backtest')
async def backtest(ctx, symbol: str = "BTC"):
    try:
        symbol = symbol.upper()
        await ctx.send(f"Starting backtest for {symbol}...")
        
        # Load sentiment data
        sentiment_data = load_all_sentiment_data()
        if not sentiment_data:
            await ctx.send("No sentiment data available. Please analyze some videos first.")
            return
            
        # Calculate average sentiment
        total_score = sum(data.get('combined_score', 0) for data in sentiment_data)
        sentiment_score = total_score / len(sentiment_data)
        
        # Run backtest
        await run_backtest(ctx, sentiment_score, symbol)
    except Exception as e:
        logger.error(f"Error running backtest: {str(e)}")
        await ctx.send(f"Error running backtest: {str(e)}")

async def run_backtest(ctx, sentiment, symbol="BTC"):
    # Create backtest instance
    backtester = SimpleBacktester()
    
    # Calculate sentiment score if passed directly as sentiment data
    if isinstance(sentiment, dict):
        score = sentiment.get('combined_score', 0)
    else:
        score = sentiment
    
    await ctx.send(f"Running backtest for {symbol} with sentiment score {score:.2f}...")
    
    # Define sentiment-based strategy
    def sentiment_strategy(df):
        # Get basic strategy signals
        base_strategy = backtester.simple_strategy(df)
        
        # Modify signals based on sentiment
        if score > 0:  # Bullish sentiment
            # Make strategy more aggressive on buys
            base_strategy['signal'] = base_strategy['signal'].apply(lambda x: 1 if x >= 0 else -1)
        elif score < 0:  # Bearish sentiment
            # Make strategy more aggressive on sells
            base_strategy['signal'] = base_strategy['signal'].apply(lambda x: -1 if x <= 0 else 1)
            
        # Add sentiment to signal strength
        for i in range(len(base_strategy) - 1):
            if base_strategy.iloc[i]['signal'] != 0:
                base_strategy.iloc[i, base_strategy.columns.get_loc('signal_strength')] += abs(score) / 10
        
        return base_strategy
    
    # Run backtest with sentiment strategy
    results = backtester.backtest(symbol, strategy_func=sentiment_strategy)
    
    if results:
        # Generate chart
        os.makedirs("charts", exist_ok=True)
        chart_path = f"charts/backtest_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        backtester.plot_backtest_results(symbol, save_to_file=chart_path)
        
        # Send chart if available
        if os.path.exists(chart_path):
            await ctx.send(file=discord.File(chart_path))
        
        # Format results message
        result_message = "**Backtest Results**\n\n"
        result_message += f"Symbol: {symbol}\n"
        result_message += f"Starting Balance: ${results['starting_balance']:.2f}\n"
        result_message += f"Ending Balance: ${results['ending_balance']:.2f}\n"
        result_message += f"Return: {results['return_pct']:.2f}%\n"
        result_message += f"Total Trades: {results['total_trades']}\n"
        
        if results.get('winning_trades') is not None:
            win_rate = (results['winning_trades'] / results['total_trades'] * 100) if results['total_trades'] > 0 else 0
            result_message += f"Win Rate: {win_rate:.2f}% ({results['winning_trades']}/{results['total_trades']})\n"
        
        result_message += f"Max Drawdown: {results.get('max_drawdown_pct', 0):.2f}%\n"
        
        await ctx.send(result_message)
        
        # Save results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_filename = f"backtest_{symbol}_sentiment_{timestamp}.json"
        backtester.save_backtest_results(symbol, filename=results_filename)
        await ctx.send(f"Backtest results saved to {results_filename}")
    else:
        await ctx.send("Backtest failed. Please check logs for details.")

def extract_youtube_id(url):
    match = re.search(youtube_regex, url)
    if match:
        return match.group(1)
    return None

def get_youtube_transcript(video_id):
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = ' '.join([item['text'] for item in transcript_list])
        return transcript_text
    except Exception as e:
        logger.error(f"Error getting YouTube transcript: {e}")
        return None

def save_sentiment_data(video_id, sentiment):
    os.makedirs(sentiment_dir, exist_ok=True)
    file_path = os.path.join(sentiment_dir, f"{video_id}.json")
    
    sentiment['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sentiment['video_id'] = video_id
    
    with open(file_path, 'w') as f:
        json.dump(sentiment, f, indent=2)
    
    logger.info(f"Saved sentiment data to {file_path}")

def load_all_sentiment_data():
    sentiment_data = []
    
    if not os.path.exists(sentiment_dir):
        return sentiment_data
    
    for filename in os.listdir(sentiment_dir):
        if filename.endswith('.json'):
            try:
                with open(os.path.join(sentiment_dir, filename), 'r') as f:
                    data = json.load(f)
                    sentiment_data.append(data)
            except Exception as e:
                logger.error(f"Error loading sentiment data: {e}")
    
    sentiment_data.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    return sentiment_data

def main():
    try:
        # Get Discord token from environment variable
        token = os.getenv('DISCORD_TOKEN')
        
        if not token:
            logger.error("Discord token not found in environment variables")
            print("Error: Discord token not found in environment variables")
            print("Please add DISCORD_TOKEN to your .env file")
            return
            
        # Run the bot
        bot.run(token)
    except Exception as e:
        logger.error(f"Error running Discord bot: {e}")
        print(f"Error running Discord bot: {e}")

if __name__ == "__main__":
    main()