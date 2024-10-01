import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from prophet import Prophet
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

uploads_folder = './uploads/'

# Function to display values on cursor hover
def display_hover_values(ax, x_data, y_data, predictions=None):
    annotation = ax.annotate("", xy=(0, 0), xytext=(20, 20),
                             textcoords="offset points", bbox=dict(boxstyle="round", fc="w"),
                             arrowprops=dict(arrowstyle="->"))
    annotation.set_visible(False)

    def update_annotation(event):
        if event.inaxes == ax:
            x, y = event.xdata, event.ydata
            distances = np.sqrt((x_data - x)**2 + (y_data - y)**2)
            idx = np.argmin(distances)
            annotation.xy = (x_data[idx], y_data[idx])
            text = f"Date: {pd.to_datetime(x_data[idx]).date()}\nActual: {y_data[idx]:.2f}"
            if predictions is not None:
                text += f"\nPredicted: {predictions[idx]:.2f}"
            annotation.set_text(text)
            annotation.set_visible(True)
            plt.draw()

    fig = ax.figure
    fig.canvas.mpl_connect("motion_notify_event", update_annotation)

# Linear Regression Analysis
def analyze_data(filepath):
    try:
        df = pd.read_csv(filepath)
        logging.info(f'Dataset loaded successfully with {len(df)} rows.')
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')

        df['DateOrdinal'] = df['Date'].map(pd.Timestamp.toordinal)
        X = df[['DateOrdinal']].values
        y = df['Close'].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = LinearRegression()
        model.fit(X_scaled, y)

        future_dates = pd.date_range(start=df['Date'].max() + pd.Timedelta(days=1), periods=30)
        future_dates_ordinal = future_dates.map(pd.Timestamp.toordinal).values.reshape(-1, 1)
        future_dates_scaled = scaler.transform(future_dates_ordinal)
        predictions = model.predict(future_dates_scaled)

        # Plotting historical data and predictions
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.lineplot(x=df['Date'], y=df['Close'], ax=ax, label='Historical Data')
        ax.plot(future_dates, predictions, label='Predictions', color='red')
        ax.set_xlabel('Date')
        ax.set_ylabel('Close Price')
        ax.set_title('Stock Price Prediction (Linear Regression)')
        ax.legend()
        ax.grid(True)

        display_hover_values(ax, df['Date'].map(pd.Timestamp.toordinal), df['Close'].values, predictions)

        # Save the plot
        plot_filename = os.path.join(uploads_folder, 'linear_regression_prediction.png')
        plt.savefig(plot_filename)
        logging.info(f'Linear regression plot saved as {plot_filename}')
        plt.show()

    except Exception as e:
        logging.error(f'Error during data analysis: {e}')

# Prophet Prediction
def prophet_prediction(filepath):
    try:
        df = pd.read_csv(filepath)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.rename(columns={'Date': 'ds', 'Close': 'y'})

        model_prophet = Prophet()
        model_prophet.fit(df)

        future = model_prophet.make_future_dataframe(periods=30)
        forecast = model_prophet.predict(future)

        fig, ax = plt.subplots(figsize=(12, 6))
        model_prophet.plot(forecast, ax=ax)

        # Use the forecast for hover interaction
        display_hover_values(ax, forecast['ds'].map(pd.Timestamp.toordinal), forecast['yhat'].values)

        # Save the plot
        plot_filename = os.path.join(uploads_folder, 'prophet_prediction.png')
        plt.savefig(plot_filename)
        logging.info(f'Prophet plot saved as {plot_filename}')
        plt.show()

    except Exception as e:
        logging.error(f'Error generating Prophet prediction: {e}')

# LSTM Prediction
def lstm_prediction(filepath):
    try:
        df = pd.read_csv(filepath)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df[['Close']])

        train_size = int(len(scaled_data) * 0.8)
        train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

        def create_sequences(data, steps):
            X, y = [], []
            for i in range(steps, len(data)):
                X.append(data[i-steps:i])
                y.append(data[i])
            return np.array(X), np.array(y)

        X_train, y_train = create_sequences(train_data, 60)
        X_test, y_test = create_sequences(test_data, 60)

        model_lstm = Sequential()
        model_lstm.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model_lstm.add(Dropout(0.2))
        model_lstm.add(LSTM(units=50))
        model_lstm.add(Dropout(0.2))
        model_lstm.add(Dense(units=1))

        model_lstm.compile(optimizer='adam', loss='mean_squared_error')
        model_lstm.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

        predicted_prices = model_lstm.predict(X_test)
        predicted_prices = scaler.inverse_transform(predicted_prices)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df.index[train_size+60:], df['Close'][train_size+60:], label='Actual Price')
        ax.plot(df.index[train_size+60:], predicted_prices, label='Predicted Price', color='red')
        ax.set_xlabel('Date')
        ax.set_ylabel('Close Price')
        ax.set_title('LSTM Prediction')
        ax.legend()
        ax.grid(True)

        display_hover_values(ax, df.index[train_size+60:].map(pd.Timestamp.toordinal), df['Close'][train_size+60:].values, predicted_prices.flatten())

        # Save the plot
        plot_filename = os.path.join(uploads_folder, 'lstm_prediction.png')
        plt.savefig(plot_filename)
        logging.info(f'LSTM plot saved as {plot_filename}')
        plt.show()

    except Exception as e:
        logging.error(f'Error generating LSTM prediction: {e}')

# XGBoost Prediction
def xgboost_prediction(filepath):
    try:
        df = pd.read_csv(filepath)
        df['Date'] = pd.to_datetime(df['Date'])

        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df = df.dropna()

        X = df[['SMA_10', 'SMA_50']]
        y = df['Close']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        model_xgb = XGBRegressor()
        model_xgb.fit(X_train, y_train)

        predicted_prices_xgb = model_xgb.predict(X_test)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df['Date'][len(X_train):], y_test, label='Actual Price')
        ax.plot(df['Date'][len(X_train):], predicted_prices_xgb, label='Predicted Price', color='red')
        ax.set_xlabel('Date')
        ax.set_ylabel('Close Price')
        ax.set_title('XGBoost Prediction')
        ax.legend()
        ax.grid(True)

        display_hover_values(ax, df['Date'][len(X_train):].map(pd.Timestamp.toordinal), y_test.values, predicted_prices_xgb)

        # Save the plot
        plot_filename = os.path.join(uploads_folder, 'xgboost_prediction.png')
        plt.savefig(plot_filename)
        logging.info(f'XGBoost plot saved as {plot_filename}')
        plt.show()

    except Exception as e:
        logging.error(f'Error generating XGBoost prediction: {e}')

# Add additional prediction method function here

# Main function to call prediction methods
def main(filepath):
    analyze_data(filepath)
    prophet_prediction(filepath)
    lstm_prediction(filepath)
    xgboost_prediction(filepath)

# Uncomment below for testing standalone functionality
if __name__ == "__main__":
    main('./IBM.csv')
