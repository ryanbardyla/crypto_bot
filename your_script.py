import os
import time
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

# Configuration - you can put these in your .env file later
TOKEN = os.environ.get("INFLUXDB_TOKEN")  # Make sure this is set before running
ORG = "Trading"
URL = "http://localhost:8086"
BUCKET = "trading_data"

def test_influxdb_connection():
    """Test connecting to InfluxDB and performing basic operations."""
    print(f"Connecting to InfluxDB at {URL}")
    
    try:
        # Create client
        client = InfluxDBClient(url=URL, token=TOKEN, org=ORG)
        
        # Test health
        health = client.health()
        print(f"Connection health: {health.status}")
        
        # Create write API
        write_api = client.write_api(write_options=SYNCHRONOUS)
        
        # Write some test data
        print("\nWriting test data points...")
        for value in range(5):
            point = (
                Point("cryptocurrency")
                .tag("symbol", "BTC")
                .field("price", 30000 + value * 100)  # Simulating prices
                .field("volume", 1000 + value * 50)   # Simulating volume
            )
            write_api.write(bucket=BUCKET, org=ORG, record=point)
            print(f"Wrote point {value+1}/5: BTC price={30000 + value * 100}, volume={1000 + value * 50}")
            time.sleep(1)  # Space the points 1 second apart
        
        # Wait for data to be fully processed
        time.sleep(1)
        
        # Create query API
        query_api = client.query_api()
        
        # 1. Execute a simple query to retrieve the data
        print("\nSimple query results:")
        query = f'''
        from(bucket: "{BUCKET}")
          |> range(start: -1h)
          |> filter(fn: (r) => r._measurement == "cryptocurrency" and r.symbol == "BTC")
        '''
        
        tables = query_api.query(query=query, org=ORG)
        
        for table in tables:
            for record in table.records:
                field = record.get_field()
                value = record.get_value()
                timestamp = record.get_time()
                print(f"Time: {timestamp}, Field: {field}, Value: {value}")
        
        # 2. Execute an aggregate query
        print("\nAggregate query results:")
        agg_query = f'''
        from(bucket: "{BUCKET}")
          |> range(start: -1h)
          |> filter(fn: (r) => r._measurement == "cryptocurrency" and r.symbol == "BTC")
          |> group(columns: ["_field"])
          |> mean()
        '''
        
        agg_tables = query_api.query(query=agg_query, org=ORG)
        
        for table in agg_tables:
            for record in table.records:
                field = record.get_field()
                value = record.get_value()
                print(f"Average {field}: {value}")
        
        print("\nInfluxDB test completed successfully!")
        
        # Close the client
        client.close()
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    if not TOKEN:
        print("ERROR: INFLUXDB_TOKEN environment variable is not set.")
        print("Please set it with: export INFLUXDB_TOKEN='your-token-here'")
        exit(1)
    
    test_influxdb_connection()