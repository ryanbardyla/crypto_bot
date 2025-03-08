# discord_bot.py
import discord
from discord.ext import commands
import os
import asyncio
import json
import re
from datetime import datetime
import logging

# Import from your existing code
from sentiment_analyzer import SentimentAnalyzer
from simple_backtester import SimpleBacktester
from youtube_transcript_api import YouTubeTranscriptApi

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

# Configure bot
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

# Initialize sentiment analyzer
sentiment_analyzer = SentimentAnalyzer()

@bot.event
async def on_ready():
    logger.info(f'{bot.user} has connected to Discord!')

@bot.command(name='analyze')
async def analyze_video(ctx, url: str):
    """Analyze a YouTube video transcript for trading sentiment"""
    await ctx.send(f"Analyzing YouTube video: {url}")
    
    try:
        # Extract YouTube video ID
        video_id = extract_youtube_id(url)
        if not video_id:
            await ctx.send("Could not extract YouTube video ID. Please provide a valid YouTube URL.")
            return
            
        # Get transcript
        await ctx.send("Fetching transcript...")
        transcript = get_youtube_transcript(video_id)
        if not transcript:
            await ctx.send("Could not retrieve transcript. The video might not have subtitles or is unavailable.")
            return
            
        # Analyze sentiment
        await ctx.send("Analyzing sentiment...")
        sentiment = sentiment_analyzer.analyze_text(transcript, source=f"youtube-{video_id}")
        
        # Format and send results
        result_message = f"**Sentiment Analysis Results**\n"
        result_message += f"Overall Score: {sentiment['combined_score']:.2f} (-10 to +10 scale)\n"
        result_message += f"Bullish Keywords: {sentiment['bullish_keywords']}\n"
        result_message += f"Bearish Keywords: {sentiment['bearish_keywords']}\n"
        
        await ctx.send(result_message)
        
        # Save sentiment to a file for the trading bot to use
        save_sentiment_data(video_id, sentiment)
        
        # Ask if user wants to run a backtest with this sentiment
        await ctx.send("Would you like to run a backtest using this sentiment data? (yes/no)")
        
        # Wait for user response
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
    """Show help message with available commands"""
    help_message = "**Trading Bot Discord Commands**\n\n"
    help_message += "`!analyze [youtube_url]` - Analyze a YouTube video for trading sentiment\n"
    help_message += "`!status` - Show current trading bot status\n"
    help_message += "`!backtest [symbol]` - Run a backtest on the specified symbol\n"
    help_message += "`!commands` - Show this help message\n"
    
    await ctx.send(help_message)

@bot.command(name='status')
async def status(ctx):
    """Show current trading bot status and sentiment data"""
    try:
        # Load current sentiment data
        sentiment_data = load_all_sentiment_data()
        
        if not sentiment_data:
            await ctx.send("No sentiment data available. Use `!analyze [youtube_url]` to analyze videos.")
            return
        
        # Calculate combined sentiment score
        total_score = 0
        total_weight = 0
        
        for data in sentiment_data:
            weight = 1
            total_score += data['combined_score'] * weight
            total_weight += weight
        
        combined_score = total_score / total_weight if total_weight > 0 else 0
        
        # Format and send status
        status_message = "**Trading Bot Status**\n\n"
        status_message += f"Analyzed videos: {len(sentiment_data)}\n"
        status_message += f"Combined sentiment score: {combined_score:.2f}\n\n"
        
        status_message += "**Recent Sentiment Analysis**\n"
        
        # Show most recent 3 sentiment analyses
        for i, data in enumerate(sentiment_data[-3:]):
            source = data.get('source', 'Unknown')
            score = data.get('combined_score', 0)
            bullish = data.get('bullish_keywords', 0)
            bearish = data.get('bearish_keywords', 0)
            
            status_message += f"{i+1}. Source: {source}\n"
            status_message += f"   Score: {score:.2f}, Bullish: {bullish}, Bearish: {bearish}\n"
        
        await ctx.send(status_message)
        
    except Exception as e:
        logger.error(f"Error getting status: {str(e)}")
        await ctx.send(f"Error getting status: {str(e)}")

@bot.command(name='backtest')
async def backtest(ctx, symbol: str = "BTC"):
    """Run a backtest on the specified symbol"""
    try:
        symbol = symbol.upper()
        await ctx.send(f"Starting backtest for {symbol}...")
        
        # Get sentiment score
        sentiment_data = load_all_sentiment_data()
        sentiment_score = 0
        
        if sentiment_data:
            total_score = sum(data.get('combined_score', 0) for data in sentiment_data)
            sentiment_score = total_score / len(sentiment_data)
            
        sentiment = {'combined_score': sentiment_score}
        await run_backtest(ctx, sentiment, symbol)
        
    except Exception as e:
        logger.error(f"Error running backtest: {str(e)}")
        await ctx.send(f"Error running backtest: {str(e)}")

async def run_backtest(ctx, sentiment, symbol="BTC"):
    """Run a backtest using the sentiment data"""
    backtester = SimpleBacktester()
    
    # Modify strategy based on sentiment
    score = sentiment['combined_score']
    
    await ctx.send(f"Running backtest for {symbol} with sentiment score {score:.2f}...")
    
    # Run backtest with sentiment-adjusted strategy
    def sentiment_strategy(df):
        """Strategy that incorporates sentiment"""
        # Get the base strategy result
        base_strategy = backtester.simple_strategy(df)
        
        if base_strategy is None:
            return None
            
        # Adjust signals based on sentiment
        if score > 5:  # Strong bullish sentiment
            # Increase buy signals
            base_strategy['signal'] = base_strategy['signal'].apply(lambda x: 1 if x >= 0 else -1)
        elif score < -5:  # Strong bearish sentiment
            # Increase sell signals
            base_strategy['signal'] = base_strategy['signal'].apply(lambda x: -1 if x <= 0 else 1)
        elif score > 2:  # Moderately bullish
            # Slightly increase buy signals
            for i in range(len(base_strategy) - 1):
                if base_strategy['signal'].iloc[i] == 1:
                    # Add another buy signal in the next row if none exists
                    if base_strategy['signal'].iloc[i+1] == 0:
                        base_strategy.loc[base_strategy.index[i+1], 'signal'] = 1
        elif score < -2:  # Moderately bearish
            # Slightly increase sell signals
            for i in range(len(base_strategy) - 1):
                if base_strategy['signal'].iloc[i] == -1:
                    # Add another sell signal in the next row if none exists
                    if base_strategy['signal'].iloc[i+1] == 0:
                        base_strategy.loc[base_strategy.index[i+1], 'signal'] = -1
        
        return base_strategy
    
    # Run the backtest with our sentiment-adjusted strategy
    results = backtester.backtest(symbol, strategy_func=sentiment_strategy)
    
    if results:
        # Create chart
        os.makedirs("charts", exist_ok=True)
        chart_path = f"charts/{symbol}_sentiment_backtest.png"
        backtester.plot_backtest_results(symbol, save_to_file=chart_path)
        
        # Send results
        result_message = f"**Backtest Results for {symbol}**\n"
        result_message += f"Starting Balance: ${results['starting_balance']:.2f}\n"
        result_message += f"Ending Balance: ${results['ending_balance']:.2f}\n"
        result_message += f"Return: {results['return_pct']:.2f}%\n"
        result_message += f"Total Trades: {results['total_trades']}\n"
        
        if results['total_trades'] > 0:
            win_rate = results['winning_trades'] / results['total_trades'] * 100
            result_message += f"Win Rate: {win_rate:.2f}% ({results['winning_trades']}/{results['total_trades']})\n"
            
        result_message += f"Max Drawdown: {results['max_drawdown_pct']:.2f}%\n"
        result_message += f"Sentiment Score Used: {score:.2f}"
        
        await ctx.send(result_message)
        
        # Send chart
        if os.path.exists(chart_path):
            await ctx.send(file=discord.File(chart_path))
        
        # Save backtest results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_filename = f"backtest_{symbol}_sentiment_{timestamp}.json"
        backtester.save_backtest_results(symbol, filename=results_filename)
        
        await ctx.send(f"Backtest results saved to {results_filename}")
    else:
        await ctx.send("Backtest failed. Please check logs for details.")

def extract_youtube_id(url):
    """Extract YouTube video ID from URL"""
    youtube_regex = r"(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})"
    match = re.search(youtube_regex, url)
    
    if match:
        return match.group(1)
    return None

def get_youtube_transcript(video_id):
    """Get transcript from a YouTube video"""
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = ' '.join([item['text'] for item in transcript_list])
        return transcript_text
    except Exception as e:
        logger.error(f"Error getting YouTube transcript: {e}")
        return None

def save_sentiment_data(video_id, sentiment):
    """Save sentiment data to a file for the trading bot to use"""
    sentiment_dir = "sentiment_data"
    os.makedirs(sentiment_dir, exist_ok=True)
    
    # Add timestamp to the sentiment data
    sentiment['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sentiment['video_id'] = video_id
    
    file_path = f"{sentiment_dir}/{video_id}.json"
    
    with open(file_path, 'w') as f:
        json.dump(sentiment, f, indent=2)
    
    logger.info(f"Saved sentiment data to {file_path}")

def load_all_sentiment_data():
    """Load all sentiment data files"""
    sentiment_dir = "sentiment_data"
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
    
    # Sort by timestamp if available
    sentiment_data.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    
    return sentiment_data

# Run the bot
def main():
    try:
        # Load Discord token from config
        with open('config.json', 'r') as f:
            config = json.load(f)
            token = config.get('discord_token')
            
        if not token:
            logger.error("Discord token not found in config.json")
            print("Error: Discord token not found in config.json")
            print("Please add a 'discord_token' field to your config.json file")
            return
            
        bot.run(token)
    except Exception as e:
        logger.error(f"Error running Discord bot: {e}")
        print(f"Error running Discord bot: {e}")

if __name__ == "__main__":
    main()