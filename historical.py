# historical.py

import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend

import matplotlib.pyplot as plt
import io
import pandas as pd

def plot_historical_data(file_path, buf):
    """
    Plots historical data from the given CSV file and saves the plot to the provided buffer.

    :param file_path: Path to the CSV file containing historical data.
    :param buf: A buffer object to save the plot image.
    """
    try:
        # Load historical data
        df = pd.read_csv(file_path)

        # Validate required columns
        if 'Date' not in df.columns or 'Value' not in df.columns:
            raise ValueError("CSV file must contain 'Date' and 'Value' columns.")

        # Create the plot
        plt.figure(figsize=(12, 6))
        plt.plot(pd.to_datetime(df['Date']), df['Value'])
        plt.title('Historical Data Plot')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.grid(True)

        # Save the plot to the buffer
        plt.savefig(buf, format='png')
        buf.seek(0)  # Go to the start of the buffer
    except Exception as e:
        print(f"Error occurred while plotting data: {e}")
        raise  # Re-raise the exception to be caught by the caller

# Example usage
if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python historical.py <path_to_csv>")
        sys.exit(1)

    csv_path = sys.argv[1]

    # Create a buffer to hold the image
    buf = io.BytesIO()

    try:
        # Plot data and save to buffer
        plot_historical_data(csv_path, buf)

        # Save buffer to a file for demonstration purposes
        with open("plot.png", "wb") as f:
            f.write(buf.getvalue())
    except Exception as e:
        print(f"Failed to generate plot: {e}")
        sys.exit(1)
