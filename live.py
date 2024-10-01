import yfinance as yf
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_live_data(symbol):
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period='1d', interval='1m')  # Get intraday data (1-minute interval)
        
        if data.empty:
            raise ValueError(f"No data found for symbol {symbol}")

        # Get the latest price from the most recent minute
        latest_data = data.iloc[-1]
        return {
            'timestamp': latest_data.name,  # Latest timestamp
            'price': latest_data['Close'],  # Latest close price
        }
    except Exception as e:
        logging.error(f'Error fetching live data: {e}')
        return {'error': str(e)}

def get_live_stock_data(symbol):
    data = fetch_live_data(symbol)
    if 'error' in data:
        return data

    return data  # This returns the live stock data in the format {timestamp, price}

# # Example usage
# if __name__ == '__main__':
#     symbol = 'AAPL'  # Apple's stock symbol
#     stock_data = get_live_stock_data(symbol)
    
#     if 'error' in stock_data:
#         print(f"Error: {stock_data['error']}")
#     else:
#         print(f"Latest stock data for {symbol}:")
#         print(f"Timestamp: {stock_data['timestamp']}")
#         print(f"Price: ${stock_data['price']}")
