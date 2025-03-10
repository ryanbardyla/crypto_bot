# app.py
import os
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from dash import Dash, html, dcc, callback, Output, Input
from sqlalchemy import create_engine, text
from database_manager import DatabaseManager

# Initialize dashboard
app = Dash(__name__, external_stylesheets=['https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css'])

# Set up database connection
db_manager = DatabaseManager(
    host=os.environ.get("POSTGRES_HOST", "localhost"),
    port=os.environ.get("POSTGRES_PORT", "5432"),
    database=os.environ.get("POSTGRES_DB", "trading_db"),
    user=os.environ.get("POSTGRES_USER", "bot_user"),
    password=os.environ.get("POSTGRES_PASSWORD", "secure_password")
)
engine = db_manager.engine

# Helper function to get cutoff date based on time range
def get_cutoff_date(time_range):
    if time_range == '24h':
        return datetime.now() - timedelta(hours=24)
    elif time_range == '7d':
        return datetime.now() - timedelta(days=7)
    elif time_range == '30d':
        return datetime.now() - timedelta(days=30)
    else:  # 'all'
        return datetime(2000, 1, 1)  # All time

# Dashboard layout
app.layout = html.Div([
    html.Div([
        html.H1("Crypto Sentiment Analysis Dashboard", className="text-center my-4"),
        
        # Sidebar with filters
        html.Div([
            html.Div([
                html.H4("Filters", className="card-header"),
                html.Div([
                    html.Label("Time Range:"),
                    dcc.Dropdown(
                        id='time-range',
                        options=[
                            {'label': 'Last 24 Hours', 'value': '24h'},
                            {'label': 'Last 7 Days', 'value': '7d'},
                            {'label': 'Last 30 Days', 'value': '30d'},
                            {'label': 'All Time', 'value': 'all'}
                        ],
                        value='7d'
                    ),
                    
                    html.Label("Symbol:", className="mt-3"),
                    dcc.Dropdown(
                        id='symbol-selector',
                        options=[
                            {'label': 'Bitcoin (BTC)', 'value': 'BTC'},
                            {'label': 'Ethereum (ETH)', 'value': 'ETH'},
                            {'label': 'Solana (SOL)', 'value': 'SOL'},
                            {'label': 'Dogecoin (DOGE)', 'value': 'DOGE'}
                        ],
                        value='BTC'
                    ),
                    
                    html.Button("Refresh Data", id="refresh-button", className="btn btn-primary mt-3 w-100")
                ], className="card-body")
            ], className="card mb-4")
        ], className="col-md-3"),
        
        # Main content
        html.Div([
            # Sentiment gauge
            html.Div([
                html.Div([
                    html.H3("Sentiment Overview", className="card-header"),
                    html.Div([
                        dcc.Graph(id='sentiment-gauge')
                    ], className="card-body")
                ], className="card mb-4")
            ]),
            
            # Sentiment vs Price chart
            html.Div([
                html.Div([
                    html.H3("Sentiment Trend vs. Price", className="card-header"),
                    html.Div([
                        dcc.Graph(id='sentiment-price-chart')
                    ], className="card-body")
                ], className="card mb-4")
            ]),
            
            # Channel comparison
            html.Div([
                html.Div([
                    html.H3("Channel Sentiment Comparison", className="card-header"),
                    html.Div([
                        dcc.Graph(id='channel-sentiment-chart')
                    ], className="card-body")
                ], className="card mb-4")
            ]),
            
            # Recent videos
            html.Div([
                html.Div([
                    html.H3("Recent Video Analysis", className="card-header"),
                    html.Div([
                        html.Div(id="video-table")
                    ], className="card-body")
                ], className="card mb-4")
            ])
        ], className="col-md-9")
    ], className="row")
], className="container-fluid")

# Callbacks
@app.callback(
    Output('sentiment-gauge', 'figure'),
    [Input('refresh-button', 'n_clicks'),
     Input('time-range', 'value')]
)
def update_sentiment_gauge(n_clicks, time_range):
    try:
        # Get cutoff date
        cutoff_date = get_cutoff_date(time_range)
        
        # Query for average sentiment
        query = """
        SELECT AVG(combined_score) as avg_score
        FROM sentiment_youtube
        WHERE processed_date >= :cutoff_date
        """
        
        with engine.connect() as conn:
            result = conn.execute(text(query), {"cutoff_date": cutoff_date})
            row = result.fetchone()
            avg_sentiment = row[0] if row and row[0] is not None else 0
            
            return go.Figure(go.Indicator(
                mode="gauge+number",
                value=avg_sentiment,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Average Sentiment Score"},
                gauge={
                    'axis': {'range': [-10, 10]},
                    'bar': {'color': "#1f77b4"},
                    'steps': [
                        {'range': [-10, -5], 'color': 'red'},
                        {'range': [-5, 0], 'color': 'indianred'},
                        {'range': [0, 5], 'color': 'lightgreen'},
                        {'range': [5, 10], 'color': 'green'}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': avg_sentiment
                    }
                }
            ))
    except Exception as e:
        # Fallback for errors
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=0,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Average Sentiment Score (Error)"},
            gauge={'axis': {'range': [-10, 10]}}
        ))
        print(f"Error updating sentiment gauge: {e}")
        return fig

@app.callback(
    Output('sentiment-price-chart', 'figure'),
    [Input('refresh-button', 'n_clicks'),
     Input('time-range', 'value'),
     Input('symbol-selector', 'value')]
)
def update_sentiment_price_chart(n_clicks, time_range, symbol):
    try:
        cutoff_date = get_cutoff_date(time_range)
        
        # Query for hourly sentiment data
        query = """
        SELECT 
            DATE_TRUNC('hour', processed_date) as date,
            AVG(combined_score) as sentiment_score,
            COUNT(*) as record_count,
            SUM(CASE WHEN record_type = 'twitter' THEN 1 ELSE 0 END) as has_twitter,
            SUM(CASE WHEN record_type = 'youtube' THEN 1 ELSE 0 END) as has_youtube
        FROM 
            sentiment_youtube
        WHERE 
            processed_date >= :cutoff_date
        GROUP BY 
            DATE_TRUNC('hour', processed_date)
        ORDER BY 
            date
        """
        
        with engine.connect() as conn:
            sentiment_df = pd.read_sql(query, conn, params={"cutoff_date": cutoff_date})
        
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
        
        # Get price data
        price_df = pd.DataFrame()
        try:
            # First try to get from database
            price_query = f"""
            SELECT 
                DATE_TRUNC('hour', timestamp) as time_period,
                AVG(price) as price
            FROM 
                price_history
            WHERE 
                symbol = '{symbol}' AND
                timestamp >= :cutoff_date
            GROUP BY 
                DATE_TRUNC('hour', timestamp)
            ORDER BY 
                time_period
            """
            
            with engine.connect() as conn:
                price_df = pd.read_sql(price_query, conn, params={"cutoff_date": cutoff_date})
            
            # If database query returned no results, fall back to JSON file
            if len(price_df) == 0:
                raise ValueError("No price data in database")
                
        except Exception as e:
            # Fall back to reading from price_history.json
            try:
                with open("price_history.json", "r") as f:
                    price_history = json.load(f)
                    price_data = price_history.get(symbol, [])
                    
                    if price_data:
                        price_df = pd.DataFrame(price_data)
                        price_df['timestamp'] = pd.to_datetime(price_df['timestamp'])
                        
                        # Resample to hourly
                        price_df['time_period'] = price_df['timestamp'].dt.strftime('%Y-%m-%d %H:00:00')
                        price_df = price_df.groupby('time_period').agg({'price': 'mean'}).reset_index()
                        price_df['time_period'] = pd.to_datetime(price_df['time_period'])
                    else:
                        # If no hourly data, go with daily
                        price_df['time_period'] = price_df['timestamp'].dt.strftime('%Y-%m-%d')
                        price_df = price_df.groupby('time_period').agg({'price': 'mean'}).reset_index()
                        price_df['time_period'] = pd.to_datetime(price_df['time_period'])
            except Exception as e2:
                print(f"Error loading price data: {e2}")
        
        # Create plot
        fig = go.Figure()
        
        # Check what data sources we have
        if not sentiment_df.empty:
            has_twitter = sentiment_df['has_twitter'].max() > 0
            has_youtube = sentiment_df['has_youtube'].max() > 0
            
            data_source = "Twitter & YouTube" if has_twitter and has_youtube else "YouTube only" if has_youtube else "Twitter only"
            
            # Add sentiment line
            fig.add_trace(go.Scatter(
                x=sentiment_df['date'],
                y=sentiment_df['sentiment_score'],
                name='Sentiment Score',
                line=dict(color='blue', width=2),
                yaxis='y'
            ))
        
        # Add price line if available
        if not price_df.empty:
            fig.add_trace(go.Scatter(
                x=price_df['time_period'],
                y=price_df['price'],
                name=f'{symbol} Price',
                line=dict(color='green', width=2, dash='dot'),
                yaxis='y2'
            ))
        
        # Update layout
        fig.update_layout(
            title=f"Sentiment vs {symbol} Price ({data_source if 'data_source' in locals() else 'No Data'})",
            xaxis=dict(title='Time'),
            yaxis=dict(
                title='Sentiment Score',
                titlefont=dict(color='blue'),
                tickfont=dict(color='blue'),
                range=[-10, 10]
            ),
            yaxis2=dict(
                title=f'{symbol} Price (USD)',
                titlefont=dict(color='green'),
                tickfont=dict(color='green'),
                anchor="x",
                overlaying="y",
                side="right"
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=50, r=50, t=80, b=50),
        )
        
        # Add trade markers if available
        try:
            # Query trade data from database
            trade_query = f"""
            SELECT
                timestamp,
                price,
                type,
                profit_loss,
                profit_loss_pct
            FROM
                trade_history
            WHERE
                symbol = '{symbol}' AND
                timestamp >= :cutoff_date
            ORDER BY
                timestamp
            """
            
            with engine.connect() as conn:
                trades_df = pd.read_sql(trade_query, conn, params={"cutoff_date": cutoff_date})
                
            if len(trades_df) > 0:
                # Add buy markers
                buys = trades_df[trades_df['type'] == 'BUY']
                if len(buys) > 0:
                    fig.add_trace(go.Scatter(
                        x=buys['timestamp'],
                        y=buys['price'],
                        mode='markers',
                        name='Buy',
                        marker=dict(symbol='triangle-up', size=12, color='green'),
                        yaxis='y2'
                    ))
                
                # Add sell markers
                sells = trades_df[trades_df['type'] == 'SELL']
                if len(sells) > 0:
                    fig.add_trace(go.Scatter(
                        x=sells['timestamp'],
                        y=sells['price'],
                        mode='markers',
                        name='Sell',
                        marker=dict(symbol='triangle-down', size=12, color='red'),
                        yaxis='y2'
                    ))
                
                # Add annotations for profit/loss
                for _, trade in sells.iterrows():
                    pnl = trade['profit_loss']
                    pnl_pct = trade['profit_loss_pct'] * 100
                    
                    fig.add_annotation(
                        x=trade['timestamp'],
                        y=trade['price'],
                        text=f"${pnl:.2f}<br>({pnl_pct:.2f}%)",
                        showarrow=True,
                        arrowhead=2,
                        ax=0,
                        ay=-40,
                        font=dict(color='red' if pnl < 0 else 'green')
                    )
            
        except Exception as e:
            # Fallback to JSON file if database query fails
            try:
                with open("paper_trading/trade_history.json", "r") as f:
                    trades = json.load(f)
                    if trades and symbol in [trade.get('symbol') for trade in trades]:
                        symbol_trades = [t for t in trades if t.get('symbol') == symbol]
                        
                        # Add buy markers
                        buys = [t for t in symbol_trades if t.get('type') == 'BUY']
                        if buys:
                            buy_times = pd.to_datetime([trade['timestamp'] for trade in buys])
                            buy_values = [trade.get('price', 0) for trade in buys]
                            
                            fig.add_trace(go.Scatter(
                                x=buy_times,
                                y=buy_values,
                                mode='markers',
                                name='Buy',
                                marker=dict(symbol='triangle-up', size=12, color='green'),
                                yaxis='y2'
                            ))
                        
                        # Add sell markers
                        sells = [t for t in symbol_trades if t.get('type') == 'SELL']
                        if sells:
                            sell_times = pd.to_datetime([trade['timestamp'] for trade in sells])
                            sell_values = [trade.get('price', 0) for trade in sells]
                            
                            fig.add_trace(go.Scatter(
                                x=sell_times,
                                y=sell_values,
                                mode='markers',
                                name='Sell',
                                marker=dict(symbol='triangle-down', size=12, color='red'),
                                yaxis='y2'
                            ))
                            
                            # Add annotations for profit/loss
                            for trade in sells:
                                time = pd.to_datetime(trade['timestamp'])
                                price = trade.get('price', 0)
                                pnl = trade.get('profit_loss', 0)
                                pnl_pct = trade.get('profit_loss_pct', 0) * 100
                                
                                fig.add_annotation(
                                    x=time,
                                    y=price,
                                    text=f"${pnl:.2f}<br>({pnl_pct:.2f}%)",
                                    showarrow=True,
                                    arrowhead=2,
                                    ax=0,
                                    ay=-40,
                                    font=dict(color='red' if pnl < 0 else 'green')
                                )
            except Exception as e2:
                print(f"Error adding trade markers: {e2}")
        
        return fig
        
    except Exception as e:
        print(f"Error updating sentiment-price chart: {e}")
        return go.Figure(data=[go.Scatter(x=[0], y=[0])], layout=dict(title=f"Error loading data: {str(e)}"))

@app.callback(
    Output('channel-sentiment-chart', 'figure'),
    [Input('refresh-button', 'n_clicks'),
     Input('time-range', 'value')]
)
def update_channel_sentiment_chart(n_clicks, time_range):
    try:
        cutoff_date = get_cutoff_date(time_range)
        
        # Query for channel-based sentiment
        query = """
        SELECT 
            channel_id,
            AVG(combined_score) as avg_score,
            COUNT(*) as video_count
        FROM 
            sentiment_youtube
        WHERE 
            processed_date >= :cutoff_date
            AND channel_id IS NOT NULL
        GROUP BY 
            channel_id
        ORDER BY 
            avg_score DESC
        LIMIT 15
        """
        
        with engine.connect() as conn:
            df = pd.read_sql(query, conn, params={"cutoff_date": cutoff_date})
        
        if df.empty:
            return px.bar(title="No sentiment data available")
        
        # Load channel names if available
        channel_map = {}
        try:
            # Try to get channel names from database
            channel_query = """
            SELECT DISTINCT
                channel_id,
                MAX(title) as channel_name
            FROM
                sentiment_youtube
            WHERE
                channel_id IS NOT NULL
            GROUP BY
                channel_id
            """
            
            with engine.connect() as conn:
                channel_df = pd.read_sql(channel_query, conn)
                channel_map = dict(zip(channel_df['channel_id'], channel_df['channel_name']))
            
            # If we don't have enough channel names, try the config file
            if len(channel_map) < len(df['channel_id']):
                with open("youtube_tracker_config.json", "r") as f:
                    config = json.load(f)
                    config_channel_map = config.get("channel_names", {})
                    # Update with any missing channels
                    for channel_id in df['channel_id']:
                        if channel_id not in channel_map and channel_id in config_channel_map:
                            channel_map[channel_id] = config_channel_map[channel_id]
        except Exception as e:
            print(f"Error loading channel names: {e}")
        
        # Apply channel names
        df['channel'] = df['channel_id'].apply(lambda x: channel_map.get(x, f"Channel {x[-6:]}"))
        
        # Create the bar chart
        fig = px.bar(
            df,
            x='channel',
            y='avg_score',
            color='avg_score',
            text='video_count',
            color_continuous_scale=px.colors.diverging.RdBu,
            color_continuous_midpoint=0,
            title=f"Channel Sentiment Comparison (Last {time_range})",
            labels={
                'channel': 'Channel',
                'avg_score': 'Average Sentiment Score',
                'video_count': 'Video Count'
            }
        )
        
        fig.update_traces(
            texttemplate='%{text} videos',
            textposition='outside'
        )
        
        fig.update_layout(
            xaxis_tickangle=-45,
            yaxis_range=[-10, 10]
        )
        
        return fig
    
    except Exception as e:
        print(f"Error updating channel sentiment chart: {e}")
        return px.bar(title="Error loading sentiment data")

@app.callback(
    Output('video-table', 'children'),
    [Input('refresh-button', 'n_clicks'),
     Input('time-range', 'value')]
)
def update_video_table(n_clicks, time_range):
    try:
        cutoff_date = get_cutoff_date(time_range)
        
        # Query for recent videos
        query = """
        SELECT 
            video_id,
            channel_id,
            title,
            bullish_keywords,
            bearish_keywords,
            combined_score,
            processed_date
        FROM 
            sentiment_youtube
        WHERE 
            processed_date >= :cutoff_date
            AND video_id IS NOT NULL
            AND record_type = 'youtube'
        ORDER BY 
            processed_date DESC
        LIMIT 20
        """
        
        with engine.connect() as conn:
            df = pd.read_sql(query, conn, params={"cutoff_date": cutoff_date})
        
        if df.empty:
            return html.Div("No recent videos found", className="text-center p-4")
        
        # Load channel names
        channel_map = {}
        try:
            with open("youtube_tracker_config.json", "r") as f:
                config = json.load(f)
                channel_map = config.get("channel_names", {})
        except Exception as e:
            print(f"Error loading channel names: {e}")
        
        # Format the data for display
        df['channel'] = df['channel_id'].apply(lambda x: channel_map.get(x, f"Channel {x[-6:]}"))
        
        # Build the table
        rows = []
        for _, row in df.iterrows():
            # Calculate sentiment class
            score = row['combined_score']
            sentiment_class = (
                "text-success" if score > 3 else 
                "text-danger" if score < -3 else 
                "text-warning" if -3 <= score < 0 else 
                "text-info"
            )
            
            # Create YouTube link
            video_url = f"https://www.youtube.com/watch?v={row['video_id']}"
            
            rows.append(html.Tr([
                html.Td(html.A(row['title'], href=video_url, target="_blank")),
                html.Td(row['channel']),
                html.Td(f"{row['bullish_keywords']} 🔼 / {row['bearish_keywords']} 🔽"),
                html.Td(f"{score:.2f}", className=sentiment_class),
                html.Td(row['processed_date'].strftime('%Y-%m-%d %H:%M') if isinstance(row['processed_date'], (datetime, pd.Timestamp)) else row['processed_date'])
            ]))
        
        # Return the complete table
        table = html.Table([
            html.Thead(
                html.Tr([
                    html.Th("Title"), 
                    html.Th("Channel"),
                    html.Th("Keywords"),
                    html.Th("Score"),
                    html.Th("Processed")
                ])
            ),
            html.Tbody(rows)
        ], className="table table-striped table-hover")
        
        return table
    
    except Exception as e:
        print(f"Error updating video table: {e}")
        return html.Div(f"Error loading data: {str(e)}", className="text-center p-4 text-danger")

if __name__ == '__main__':
    app.run_server(debug=True)