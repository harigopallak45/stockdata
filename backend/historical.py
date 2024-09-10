import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import matplotlib
import io
import mplcursors
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Use 'Agg' to avoid GUI requirement (suitable for Flask)
matplotlib.use('Agg')

def plot_historical_data(file_path=None, buffer=None):
    """
    Plots historical stock data from a given CSV file and saves the plot to the provided buffer.
    
    :param file_path: Path to the CSV file containing historical stock data.
    :param buffer: A BytesIO buffer where the image is saved.
    """
    try:
        # Read the CSV file, ensure 'Date' is parsed as a date column and set as index
        df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
        logging.info(f'Dataset loaded successfully from {file_path}')
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        return {'error': f"File not found: {file_path}"}
    except pd.errors.EmptyDataError:
        logging.error("No data in the file.")
        return {'error': "No data in the file."}
    except pd.errors.ParserError:
        logging.error("Error parsing the file. Please check the CSV format.")
        return {'error': "Error parsing the file. Please check the CSV format."}

    if 'Close' not in df.columns:
        logging.error("Column 'Close' not found in the CSV file.")
        return {'error': "Column 'Close' not found in the CSV file."}

    close_prices = df['Close']

    # Set up the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    line, = ax.plot(df.index, close_prices, label='Historical Close Prices', color='blue')

    # Format x-axis dates
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()

    # Set y-axis limits with some margin
    min_y = close_prices.min()
    max_y = close_prices.max()
    y_margin = (max_y - min_y) * 0.1
    ax.set_ylim(min_y - y_margin, max_y + y_margin)

    # Set labels and title
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.set_title('Historical Stock Price')
    ax.legend()
    ax.grid(True)

    # Add interactive cursor
    cursor = mplcursors.cursor(line, hover=True)

    @cursor.connect("add")
    def on_add(sel):
        date_str = mdates.num2date(sel.target[0]).strftime('%Y-%m-%d')
        price = sel.target[1]
        sel.annotation.set_text(f"{date_str}\nPrice: {price:.2f}")

    # Save plot to the buffer if provided
    if buffer is not None:
        plt.savefig(buffer, format='png')
        plt.close(fig)  # Close the figure to avoid memory leaks
        logging.info("Plot saved successfully to buffer.")
    else:
        logging.error("No buffer provided for saving the plot.")
        return {'error': "No buffer provided for saving the plot."}

    return {'success': True}
