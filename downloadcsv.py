import yfinance as yf

def download_stock_data(ticker, start_date, end_date):
    # Fetch historical market data for the ticker symbol
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    
    # Save the data to a CSV file
    file_name = f"{ticker}_stock_data.csv"
    stock_data.to_csv(file_name)
    
    print(f"Stock data for {ticker} from {start_date} to {end_date} saved to {file_name}")

# Example usage
if __name__ == "__main__":
    ticker_symbol = input("Enter the stock ticker symbol (e.g., AAPL for Apple): ")
    start_date = input("Enter the start date (YYYY-MM-DD): ")
    end_date = input("Enter the end date (YYYY-MM-DD): ")
    
    download_stock_data(ticker_symbol, start_date, end_date)
