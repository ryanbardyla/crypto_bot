# app.py
import os
import json
import sqlite3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from dash import Dash, html, dcc, callback, Output, Input
from dash.dependencies import Input, Output, State
from sqlalchemy import create_engine, text

# Initialize the Dash app
app = Dash(__name__, external_stylesheets=['https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css'])
app.title = "Crypto Sentiment Dashboard"

# Database connection
DB_PATH = "sqlite:///sentiment_database.db"
engine = create_engine(DB_PATH)

# Helper function to get cutoff date based on time range
def get_cutoff_date(time_range):
    if time_range == '24h':
        return datetime.now() - timedelta(hours=24)
    elif time_range == '7d':
        return datetime.now() - timedelta(days=7)
    elif time_range == '30d':
        return datetime.now() - timedelta(days=30)
    else:
        return datetime(2000, 1, 1)  # All time

# Layout
app.layout = html.Div([
    html.Div([
        html.H1("Crypto Sentiment Analysis Dashboard", className="text-center my-4"),
        
        html.Div([
            html.Div([
                html.H4("Filters", className="card-header"),
                html.Div([
                    html.Label("Time Range:"),
                    dcc.Dropdown(
                        id='time-range',
                        options=[
                            {'label': '24 hours', 'value': '24h'},
                            {'label': '7 days', 'value': '7d'},
                            {'label': '30 days', 'value': '30d'},
                            {'label': 'All time', 'value': 'all'}
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
        
        html.Div([
            html.Div([
                html.Div([
                    html.H3("Sentiment Overview", className="card-header"),
                    html.Div([
                        dcc.Graph(id='sentiment-gauge')
                    ], className="card-body")
                ], className="card mb-4")
            ], className="row"),
            
            html.Div([
                html.Div([
                    html.H3("Sentiment Trend vs. Price", className="card-header"),
                    html.Div([
                        dcc.Graph(id='sentiment-price-chart')
                    ], className="card-body")
                ], className="card mb-4")
            ], className="row"),
            
            html.Div([
                html.Div([
                    html.H3("Channel Sentiment Comparison", className="card-header"),
                    html.Div([
                        dcc.Graph(id='channel-sentiment-chart')
                    ], className="card-body")
                ], className="card mb-4")
            ], className="row"),
            
            html.Div([
                html.Div([
                    html.H3("Recent Video Analysis", className="card-header"),
                    html.Div([
                        html.Div(id="video-table")
                    ], className="card-body")
                ], className="card mb-4")
            ], className="row")
        ], className="col-md-9")
    ], className="row")
], className="container-fluid p-4")

# Callback for sentiment gauge
@app.callback(
    Output('sentiment-gauge', 'figure'),
    [Input('refresh-button', 'n_clicks'),
     Input('time-range', 'value')]
)
def update_sentiment_gauge(n_clicks, time_range):
    try:
        if time_range == '24h':
            cutoff_date = datetime.now() - timedelta(hours=24)
        elif time_range == '7d':
            cutoff_date = datetime.now() - timedelta(days=7)
        elif time_range == '30d':
            cutoff_date = datetime.now() - timedelta(days=30)
        else:
            cutoff_date = datetime(2000, 1, 1)  # All time
        
        # Modified query to handle the absence of Twitter data
        query = """
        SELECT 
            avg(combined_score) as avg_score,
            count(*) as record_count,
            count(distinct case when source like '%youtube%' then 1 else null end) as youtube_count,
            count(distinct case when source like '%twitter%' then 1 else null end) as twitter_count
        FROM sentiment_records
        WHERE processed_date >= :cutoff_date
        """
        
        with engine.connect() as conn:
            result = conn.execute(text(query), {"cutoff_date": cutoff_date})
            row = result.fetchone()
        
        if not row or row['record_count'] == 0:
            return go.Figure(go.Indicator(
                mode="gauge+number",
                value=0,
                title={'text': "No Data Available"},
                gauge={'axis': {'range': [-10, 10]},
                       'bar': {'color': "gray"},
                       'steps': [
                           {'range': [-10, -5], 'color': "red"},
                           {'range': [-5, 0], 'color': "orange"},
                           {'range': [0, 5], 'color': "lightgreen"},
                           {'range': [5, 10], 'color': "green"}
                       ]}))
        
        sentiment_value = row['avg_score']
        record_count = row['record_count']
        has_youtube = row['youtube_count'] > 0
        has_twitter = row['twitter_count'] > 0
        
        # Determine color based on sentiment
        if sentiment_value < -5:
            color = "red"
        elif sentiment_value < 0:
            color = "orange"
        elif sentiment_value < 5:
            color = "lightgreen"
        else:
            color = "green"
        
        # Create title with data source info
        title = "Overall Sentiment"
        if has_youtube and not has_twitter:
            title += " (YouTube Only)"
        
        # Create and return the gauge figure
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=sentiment_value,
            title={'text': title},
            gauge={
                'axis': {'range': [-10, 10], 'tickwidth': 1},
                'bar': {'color': color},
                'steps': [
                    {'range': [-10, -5], 'color': "red"},
                    {'range': [-5, 0], 'color': "orange"},
                    {'range': [0, 5], 'color': "lightgreen"},
                    {'range': [5, 10], 'color': "green"}
                ]
            },
            number={'suffix': f" ({record_count} sources)"}
        ))
        
        return fig
    except Exception as e:
        print(f"Error updating sentiment gauge: {e}")
        return go.Figure(go.Indicator(
            mode="gauge+number",
            value=0,
            title={'text': "Error Loading Data"},
            gauge={'axis': {'range': [-10, 10]}, 'bar': {'color': "gray"}}))

# Callback for sentiment and price chart
@app.callback(
    Output('sentiment-price-chart', 'figure'),
    [Input('refresh-button', 'n_clicks'),
     Input('time-range', 'value'),
     Input('symbol-selector', 'value')]
)
def update_sentiment_price_chart(n_clicks, time_range, symbol):
    try:
        cutoff_date = get_cutoff_date(time_range)
        
        # Modified query to handle YouTube-only data
        query = """
        SELECT 
            date(processed_date) as date,
            avg(combined_score) as avg_sentiment,
            count(*) as record_count,
            max(case when source like '%youtube%' then 1 else 0 end) as has_youtube,
            max(case when source like '%twitter%' then 1 else 0 end) as has_twitter
        FROM sentiment_records
        WHERE processed_date >= :cutoff_date
        GROUP BY date(processed_date)
        ORDER BY date
        """
        
        with engine.connect() as conn:
            sentiment_df = pd.read_sql(query, conn, params={"cutoff_date": cutoff_date})
        
        # Convert date string to datetime
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
        
        # Load price data
        price_df = pd.DataFrame()
        try:
            with open("price_history.json", "r") as f:
                price_history = json.load(f)
                price_data = price_history.get(symbol, [])
                
                if price_data:
                    price_df = pd.DataFrame(price_data)
                    price_df['timestamp'] = pd.to_datetime(price_df['timestamp'])
                    
                    # Group by day or hour depending on time range
                    if time_range == '24h':
                        price_df['time_period'] = price_df['timestamp'].dt.strftime('%Y-%m-%d %H:00:00')
                        price_df = price_df.groupby('time_period').agg({'price': 'mean'}).reset_index()
                        price_df['time_period'] = pd.to_datetime(price_df['time_period'])
                    else:
                        price_df['time_period'] = price_df['timestamp'].dt.strftime('%Y-%m-%d')
                        price_df = price_df.groupby('time_period').agg({'price': 'mean'}).reset_index()
                        price_df['time_period'] = pd.to_datetime(price_df['time_period'])
        except Exception as e:
            print(f"Error loading price data: {e}")
            price_df = pd.DataFrame()
        
        # Create the figure
        fig = go.Figure()
        
        # Add sentiment data
        if not sentiment_df.empty:
            # Check if we have both data sources or just YouTube
            has_twitter = sentiment_df['has_twitter'].max() > 0
            has_youtube = sentiment_df['has_youtube'].max() > 0
            
            # Determine title based on available data sources
            chart_title = f"{symbol} Price vs. Sentiment"
            if has_youtube and not has_twitter:
                chart_title += " (YouTube Only)"
            
            # Add sentiment trace
            fig.add_trace(go.Scatter(
                x=sentiment_df['date'],
                y=sentiment_df['avg_sentiment'],
                name='Sentiment',
                mode='lines+markers',
                line=dict(color='blue', width=2),
                yaxis='y1'
            ))
        
        # Add price data
        if not price_df.empty:
            fig.add_trace(go.Scatter(
                x=price_df['time_period'],
                y=price_df['price'],
                name=f'{symbol} Price',
                mode='lines',
                line=dict(color='green', width=2, dash='dot'),
                yaxis='y2'
            ))
        
        # Update layout
        fig.update_layout(
            title=chart_title if sentiment_df.empty else chart_title,
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
                overlaying='y',
                side='right'
            ),
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02
            ),
            margin=dict(l=50, r=50, t=80, b=50),
            hovermode='x unified'
        )
        
        # Add trade markers if available
        try:
            with open("paper_trading/trade_history.json", "r") as f:
                trades = json.load(f)
                
                if trades and symbol in [trade.get('symbol') for trade in trades]:
                    # Filter trades for the selected symbol
                    symbol_trades = [t for t in trades if t.get('symbol') == symbol]
                    
                    # Process buy trades
                    buys = [t for t in symbol_trades if t.get('type') == 'BUY']
                    if buys:
                        buy_times = pd.to_datetime([trade['timestamp'] for trade in buys])
                        buy_values = [trade.get('price', 0) for trade in buys]
                        fig.add_trace(go.Scatter(
                            x=buy_times,
                            y=buy_values,
                            mode='markers',
                            marker=dict(symbol='triangle-up', size=12, color='green'),
                            name='Buy',
                            yaxis='y2'
                        ))
                    
                    # Process sell trades
                    sells = [t for t in symbol_trades if t.get('type') == 'SELL']
                    if sells:
                        sell_times = pd.to_datetime([trade['timestamp'] for trade in sells])
                        sell_values = [trade.get('price', 0) for trade in sells]
                        fig.add_trace(go.Scatter(
                            x=sell_times,
                            y=sell_values,
                            mode='markers',
                            marker=dict(symbol='triangle-down', size=12, color='red'),
                            name='Sell',
                            yaxis='y2'
                        ))
                        
                        # Add profit/loss annotations
                        for trade in sells:
                            if 'profit_loss' in trade and 'profit_loss_pct' in trade:
                                time = pd.to_datetime(trade['timestamp'])
                                price = trade.get('price', 0)
                                pnl = trade.get('profit_loss', 0)
                                pnl_pct = trade.get('profit_loss_pct', 0) * 100
                                
                                fig.add_annotation(
                                    x=time,
                                    y=price,
                                    text=f"${pnl:.0f}\n({pnl_pct:.1f}%)",
                                    showarrow=True,
                                    arrowhead=4,
                                    ax=0,
                                    ay=-40,
                                    font=dict(color='red' if pnl < 0 else 'green')
                                )
        except Exception as e:
            print(f"Error adding trade markers: {e}")
        
        return fig
    except Exception as e:
        print(f"Error updating sentiment-price chart: {e}")
        return go.Figure(data=[go.Scatter(x=[0], y=[0])], layout=dict(title=f"Error loading data: {str(e)}"))

# Callback for channel sentiment comparison chart
@app.callback(
    Output('channel-sentiment-chart', 'figure'),
    [Input('refresh-button', 'n_clicks'),
     Input('time-range', 'value')]
)
def update_channel_sentiment_chart(n_clicks, time_range):
    try:
        cutoff_date = get_cutoff_date(time_range)
        
        # Modified query to handle potential absence of Twitter data
        query = """
        SELECT 
            channel_id,
            source,
            avg(combined_score) as avg_score,
            count(*) as count
        FROM sentiment_records
        WHERE processed_date >= :cutoff_date
        GROUP BY channel_id, source
        """
        
        with engine.connect() as conn:
            df = pd.read_sql(query, conn, params={"cutoff_date": cutoff_date})
        
        if df.empty:
            return px.bar(title="No sentiment data available")
        
        # Check which sources we have data for
        has_twitter = 'twitter' in df['source'].str.lower().values
        has_youtube = 'youtube' in df['source'].str.lower().values
        
        # Load channel names from config
        try:
            with open("youtube_tracker_config.json", "r") as f:
                config = json.load(f)
                channel_map = config.get("channel_names", {})
        except:
            channel_map = {}
        
        # Apply channel names
        for i, row in df.iterrows():
            channel_id = row['channel_id']
            if channel_id in channel_map:
                df.at[i, 'channel_name'] = channel_map[channel_id]
            else:
                df.at[i, 'channel_name'] = f"Channel {channel_id[-6:] if channel_id else ''}"
        
        title = "Channel Sentiment Comparison"
        if not has_twitter:
            title += " (YouTube Only)"
        
        # Create the chart
        fig = px.bar(
            df, 
            x='channel_name', 
            y='avg_score',
            color='avg_score',
            title=title,
            labels={'avg_score': 'Sentiment Score', 'channel_name': 'Channel'},
            color_continuous_scale=px.colors.diverging.RdBu,
            color_continuous_midpoint=0,
            hover_data=['count']
        )
        
        fig.update_traces(
            marker_line_width=1,
            marker_line_color="black"
        )
        
        return fig
    except Exception as e:
        print(f"Error updating channel sentiment chart: {e}")
        return px.bar(title="Error loading sentiment data")

# Callback for video table
@app.callback(
    Output('video-table', 'children'),
    [Input('refresh-button', 'n_clicks'),
     Input('time-range', 'value')]
)
def update_video_table(n_clicks, time_range):
    try:
        cutoff_date = get_cutoff_date(time_range)
        
        # Query to get recent videos - only YouTube now
        query = """
        SELECT 
            video_id,
            title, 
            channel_id,
            combined_score,
            bullish_keywords,
            bearish_keywords,
            processed_date
        FROM sentiment_records
        WHERE 
            processed_date >= :cutoff_date
            AND source LIKE '%youtube%'
        ORDER BY processed_date DESC
        LIMIT 10
        """
        
        with engine.connect() as conn:
            df = pd.read_sql(query, conn, params={"cutoff_date": cutoff_date})
        
        if df.empty:
            return html.Div("No recent videos found", className="text-center p-4")
        
        # Load channel mapping info
        try:
            with open("youtube_tracker_config.json", "r") as f:
                config = json.load(f)
                channel_map = config.get("channel_names", {})
        except:
            channel_map = {}
        
        # Add channel name column
        df['channel'] = df['channel_id'].apply(lambda x: channel_map.get(x, f"Channel {x[-6:]}"))
        
        # Format the data for display
        rows = []
        for _, row in df.iterrows():
            video_id = row['video_id']
            video_url = f"https://www.youtube.com/watch?v={video_id}" if video_id else "#"
            
            score = row['combined_score']
            sentiment_class = ""
            if score < -5:
                sentiment_class = "text-danger fw-bold"
            elif score < 0:
                sentiment_class = "text-danger"
            elif score > 5:
                sentiment_class = "text-success fw-bold"
            elif score > 0:
                sentiment_class = "text-success"
            
            rows.append(html.Tr([
                html.Td(html.A(row['title'], href=video_url, target="_blank")),
                html.Td(row['channel']),
                html.Td(f"{row['bullish_keywords']} 🔼 / {row['bearish_keywords']} 🔽"),
                html.Td(f"{score:.2f}", className=sentiment_class),
                html.Td(row['processed_date'].strftime('%Y-%m-%d %H:%M') if isinstance(row['processed_date'], (datetime, pd.Timestamp)) else row['processed_date'])
            ]))
        
        # Create the table
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

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)