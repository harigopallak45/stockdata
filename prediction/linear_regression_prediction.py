import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)

def load_and_prepare_data(file_path):
    """Load and prepare the stock data."""
    try:
        # Load the data from CSV
        df = pd.read_csv(file_path)
        
        # Assuming the CSV has 'Date' and 'Close' columns
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        
        logging.info("Data loaded and prepared successfully")
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return None

def create_features(df):
    """Create features for Linear Regression model."""
    df['Date_ordinal'] = pd.to_datetime(df.index).map(pd.Timestamp.toordinal)
    return df[['Date_ordinal']], df['Close']

def split_data(X, y):
    """Split the data into training and test sets."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    return X_train, X_test, y_train, y_test

def train_linear_regression(X_train, y_train):
    """Train a Linear Regression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    logging.info("Linear Regression model trained successfully")
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model performance."""
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    logging.info(f"Model Mean Squared Error: {mse}")
    logging.info(f"Model R-squared: {r2}")
    return predictions, mse, r2

def plot_predictions(X_train, y_train, X_test, y_test, predictions, plot_path):
    """Plot the actual vs predicted values."""
    plt.figure(figsize=(10,6))
    plt.plot(X_train, y_train, label="Training Data")
    plt.plot(X_test, y_test, label="Actual Stock Price", color='blue')
    plt.plot(X_test, predictions, label="Predicted Stock Price", color='red', linestyle='dashed')
    plt.title('Stock Price Prediction using Linear Regression')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()

    # Save the plot
    if os.path.exists(plot_path):
        os.remove(plot_path)  # Remove existing plot to update with new one
    
    plt.savefig(plot_path, format='png')
    plt.close()
    logging.info(f"Plot saved to {plot_path}")

def linear_regression_prediction(file_path, plot_path):
    """Main function to run Linear Regression prediction and plotting."""
    try:
        # Step 1: Load and prepare data
        df = load_and_prepare_data(file_path)
        if df is None:
            logging.error("Failed to load data.")
            return
        
        # Step 2: Create features
        X, y = create_features(df)
        
        # Step 3: Split data into train and test sets
        X_train, X_test, y_train, y_test = split_data(X, y)
        
        # Step 4: Train the Linear Regression model
        model = train_linear_regression(X_train, y_train)
        
        # Step 5: Evaluate the model
        predictions, mse, r2 = evaluate_model(model, X_test, y_test)
        
        # Step 6: Plot actual vs predicted stock prices
        plot_predictions(X_train, y_train, X_test, y_test, predictions, plot_path)
        
    except Exception as e:
        logging.error(f"Error during Linear Regression prediction: {e}")

# Example usage
if __name__ == "__main__":
    file_path = "uploads/GOOGL.csv"  # Example file path
    plot_path = "plots/linear_regression_plot.png"  # Example plot path

    linear_regression_prediction(file_path, plot_path)
