import requests
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_live_data(api_url, params):
    """
    Fetches live stock data from the API.

    :param api_url: The URL of the API endpoint.
    :param params: Parameters to be sent with the request.
    :return: Parsed JSON data from the API response or an error message.
    """
    try:
        response = requests.get(api_url, params=params)
        response.raise_for_status()  # Raise an exception for HTTP errors
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        logging.error(f'Error fetching live data: {e}')
        return {'error': str(e)}

def get_live_stock_data(api_url, params):
    """
    Retrieves and parses live stock data.

    :param api_url: The URL of the API endpoint.
    :param params: Parameters to be sent with the request.
    :return: Parsed live stock data or an error message.
    """
    data = fetch_live_data(api_url, params)
    if 'error' in data:
        return data

    return parse_live_data(data)

def parse_live_data(data):
    """
    Parses the live data to extract relevant information.

    :param data: The JSON data from the API response.
    :return: A dictionary with the timestamp and price, or an error message.
    """
    try:
        # Adjust the parsing logic according to the API you're using
        # Example below assumes a common structure; adjust if necessary
        timestamp = list(data['Time Series (1min)'].keys())[0]  # Adjust key if different
        price = data['Time Series (1min)'][timestamp]['1. open']  # Adjust key if different
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
