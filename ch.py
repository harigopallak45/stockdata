import yfinance as yf

# Define the stock symbol
stock_symbol = 'AAPL'

# Get live stock data
stock = yf.Ticker(stock_symbol)

# Fetch stock info
live_data = stock.info

# Print key stock data
print(f"Stock Price: {live_data['currentPrice']}")
print(f"Market Cap: {live_data['marketCap']}")
print(f"PE Ratio: {live_data['trailingPE']}")
