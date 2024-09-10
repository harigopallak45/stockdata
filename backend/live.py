import requests
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_live_data(api_url, params):
    try:
        response = requests.get(api_url, params=params)
        response.raise_for_status()
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        logging.error(f'Error fetching live data: {e}')
        return {'error': str(e)}

def get_live_stock_data(api_url, params):
    data = fetch_live_data(api_url, params)
    if 'error' in data:
        return data

    # Extract relevant information based on the API response format
    # Adapt the following line to match the response format of the chosen API
    return parse_live_data(data)

def parse_live_data(data):
    """
    Parses the live data to extract relevant information.
    Adapt this function according to the specific API response structure.
    """
    try:
        # Example for Alpha Vantage (adjust for your chosen API)
        timestamp = list(data['Time Series (1min)'].keys())[0]
        price = data['Time Series (1min)'][timestamp]['1. open']
        return {
            'timestamp': timestamp,
            'price': price
        }
    except KeyError as e:
        logging.error(f'Missing key in data: {e}')
        return {'error': f'Missing key in data: {e}'}
    except Exception as e:
        logging.error(f'Error parsing live data: {e}')
        return {'error': str(e)}
