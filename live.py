import requests
import json
from datetime import datetime


def fetch_live_data(api_url, params):
    """
    Fetches live stock data from an API.
    """
    try:
        response = requests.get(api_url, params=params)
        response.raise_for_status()
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        return {'error': str(e)}


def parse_live_data(data):
    """
    Parses the live data to extract relevant information.
    Assumes the data contains 'timestamp' and 'price' fields.
    """
    try:
        timestamp = datetime.fromtimestamp(data['timestamp'])
        price = data['price']
        return {
            'timestamp': timestamp,
            'price': price
        }
    except KeyError as e:
        return {'error': f'Missing key in data: {e}'}
    except Exception as e:
        return {'error': str(e)}


def get_live_stock_data(api_url, params):
    """
    Fetches and parses live stock data.
    """
    data = fetch_live_data(api_url, params)
    if 'error' in data:
        return data

    parsed_data = parse_live_data(data)
    return parsed_data

def fetch_live_data():
    # Example live data (replace with actual live data fetching logic)
    return {'current_price': 145.50, 'timestamp': '2024-09-04T12:00:00Z'}



# Example usage
if __name__ == "__main__":
    # Replace with your actual API URL and parameters
    api_url = "http://api.example.com/live_stock_data"
    params = {"symbol": "AAPL"}

    live_data = get_live_stock_data(api_url, params)
    print(json.dumps(live_data, indent=2))
