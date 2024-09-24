import requests
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_live_data(api_url, params):
    
    try:
        response = requests.get(api_url, params=params)
        logging.info(f'Response Status Code: {response.status_code}')
        logging.info(f'Response Content: {response.text}')  # Log full response content for debugging

        if response.status_code == 429:
            logging.error('Rate limit exceeded. Please wait before making more requests.')
            return {'error': 'Rate limit exceeded. Please try again later.'}

        response.raise_for_status()  # Raise an exception for HTTP errors

        data = response.json()
        return data

    except requests.exceptions.HTTPError as http_err:
        logging.error(f'HTTP error occurred: {http_err}')
        return {'error': f'HTTP error occurred: {http_err}'}
    except requests.exceptions.RequestException as err:
        logging.error(f'Error fetching live data: {err}')
        return {'error: str(err)'}

def get_live_stock_data(api_url, params):

    data = fetch_live_data(api_url, params)
    if 'error' in data:
        return data

    return parse_live_data(data)

def parse_live_data(data):
    
    try:
        # Adjust the parsing logic according to the API you're using
        time_series = data.get('Time Series (1min)', {})
        if not time_series:
            raise KeyError('Time Series (1min) missing in the data')

        # Get the most recent timestamp
        timestamp = list(time_series.keys())[0]

        # Get the stock price at the timestamp
        price = time_series[timestamp]['1. open']

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

if __name__ == '__main__':
    # Example usage of the code
    api_url = "https://www.alphavantage.co/query"  # Example API URL
    params = {
        'apikey': 'your_api_key',    # Replace with your API key
        'symbol': 'AAPL',            # Replace with the stock symbol you're querying
        'interval': '1min',          # Set the interval (e.g., 1min, 5min, 15min)
        'function': 'TIME_SERIES_INTRADAY'
    }

    # Fetch and display live stock data
    live_data = get_live_stock_data(api_url, params)
    print(live_data)
