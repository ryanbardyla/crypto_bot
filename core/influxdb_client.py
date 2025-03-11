"""
InfluxDB Client: Handles connections and operations with InfluxDB for historical data storage.
"""

import logging
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
from core.config_loader import load_config
import os
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

class InfluxDBClient:
    """
    A client for interacting with InfluxDB.
    """
    def __init__(self):
        """
        Initialize the InfluxDB client using settings from config.
        """
        config = load_config()
        self.url = config["influxdb"]["url"]
        self.token = config["influxdb"]["token"]
        self.org = config["influxdb"]["org"]
        self.bucket = config["influxdb"]["bucket"]
        self.client = InfluxDBClient(url=self.url, token=self.token, org=self.org)
        self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
        try:
            self.client.ping()
            logger.info(f"Connected to InfluxDB at {self.url}")
        except Exception as e:
            logger.error(f"Failed to connect to InfluxDB: {e}")
            raise

    def write_point(self, measurement: str, tags: dict, fields: dict) -> None:
        """
        Write a data point to InfluxDB.

        Args:
            measurement (str): The measurement name (e.g., 'trades').
            tags (dict): Tags to categorize the data (e.g., {'symbol': 'BTC'}).
            fields (dict): Fields to store (e.g., {'price': 50000}).
        """
        try:
            point = Point(measurement).tag(**tags).field(**fields).time(time=None, write_precision=WritePrecision.S)
            self.write_api.write(bucket=self.bucket, record=point)
            logger.debug(f"Wrote point to InfluxDB: {measurement}, tags={tags}, fields={fields}")
        except Exception as e:
            logger.error(f"Failed to write point to InfluxDB: {e}")
            raise

    def query_data(self, query: str):
        """
        Query data from InfluxDB.

        Args:
            query (str): The Flux query to execute.

        Returns:
            The query result.
        """
        try:
            query_api = self.client.query_api()
            result = query_api.query(org=self.org, query=query)
            logger.debug(f"Executed query: {query}")
            return result
        except Exception as e:
            logger.error(f"Failed to query InfluxDB: {e}")
            raise

# Example usage (for testing purposes)
if __name__ == "__main__":
    try:
        influx_client = InfluxDBClient()
        # Example write
        influx_client.write_point(
            measurement="trades",
            tags={"symbol": "BTC"},
            fields={"price": 50000, "volume": 1.5}
        )
        # Example query
        query = f'from(bucket:"{influx_client.bucket}") |> range(start: -1h)'
        result = influx_client.query_data(query)
        print(result)
    except Exception as e:
        print(f"Error: {e}")