import logging
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from sklearn.preprocessing import MinMaxScaler
from prophet import Prophet

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def analyze_data(filepath):
    """
    Analyzes the dataset to generate historical data, predictions, month-wise differences, and plots.
    Assumes dataset contains columns like 'Date', 'Close', etc.
    """
    try:
        df = pd.read_csv(filepath)
        logging.info(f'Dataset loaded successfully with {len(df)} rows.')

        # Validate required columns
        required_columns = ['Date', 'Close']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f'Missing required column: {col}')

        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')

        # Historical data
        historical_data = df[['Date', 'Close']].to_dict(orient='records')
        logging.info('Historical data extracted successfully.')

        # Prepare data for prediction
        df['DateOrdinal'] = df['Date'].map(pd.Timestamp.toordinal)
        X = df[['DateOrdinal']].values
        y = df['Close'].values

        # Scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Linear Regression Model
        model = LinearRegression()
        model.fit(X_scaled, y)
        logging.info('Linear Regression model trained successfully.')

        # Prediction
        future_dates = pd.date_range(start=df['Date'].max() + pd.Timedelta(days=1), periods=30)
        future_dates_ordinal = future_dates.map(pd.Timestamp.toordinal).values.reshape(-1, 1)
        future_dates_scaled = scaler.transform(future_dates_ordinal)
        predictions = model.predict(future_dates_scaled)

        prediction_data = [{'Date': date.strftime('%Y-%m-%d'), 'Prediction': pred} for date, pred in
                           zip(future_dates, predictions)]
        logging.info('Linear Regression predictions generated successfully.')

        # Month-wise data difference
        df['Month'] = df['Date'].dt.to_period('M')
        month_wise_diff = df.groupby('Month')['Close'].agg(['min', 'max']).reset_index()
        month_wise_diff['Difference'] = month_wise_diff['max'] - month_wise_diff['min']
        
        # Plotting month-wise differences
        month_wise_diff_plot_path = os.path.join('uploads', 'month_wise_diff_plot.png')
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Month', y='Difference', data=month_wise_diff)
        plt.xlabel('Month')
        plt.ylabel('Difference')
        plt.title('Month-Wise Differences')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(month_wise_diff_plot_path, format='png')
        plt.close()
        logging.info('Month-wise differences plot generated successfully.')

        # Plotting historical data and predictions
        historical_vs_predictions_plot_path = os.path.join('uploads', 'historical_vs_predictions_plot.png')
        plt.figure(figsize=(12, 6))
        sns.lineplot(x='Date', y='Close', data=df, label='Historical Data')
        plt.plot(future_dates, predictions, label='Predictions', color='red')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.title('Stock Price Prediction')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(historical_vs_predictions_plot_path, format='png')
        plt.close()
        logging.info('Plot of historical data and predictions generated successfully.')

        # Return data
        return {
            'historical_data': historical_data,
            'prediction_data': prediction_data,
            'month_wise_diff_plot': month_wise_diff_plot_path,
            'historical_vs_predictions_plot': historical_vs_predictions_plot_path
        }

    except ValueError as ve:
        logging.error(f'Value error during data analysis: {ve}')
        return {'error': str(ve)}
    except Exception as e:
        logging.error(f'Error during data analysis: {e}')
        return {'error': str(e)}

def prophet_prediction(filepath):
    """
    Generates predictions using Prophet and saves the plot.
    """
    try:
        df = pd.read_csv(filepath)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.rename(columns={'Date': 'ds', 'Close': 'y'})
        
        model_prophet = Prophet()
        model_prophet.fit(df)
        
        future = model_prophet.make_future_dataframe(periods=30)
        forecast = model_prophet.predict(future)
        
        prophet_plot_path = os.path.join('uploads', 'prophet_prediction.png')
        fig = model_prophet.plot(forecast)
        plt.savefig(prophet_plot_path, format='png')
        plt.close(fig)
        logging.info('Prophet prediction plot generated successfully.')
        
        return prophet_plot_path

    except Exception as e:
        logging.error(f'Error generating Prophet prediction: {e}')
        return {'error': str(e)}

def lstm_prediction(filepath):
    """
    Generates predictions using LSTM and saves the plot.
    """
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
        
        lstm_plot_path = os.path.join('uploads', 'lstm_prediction.png')
        plt.figure(figsize=(12, 6))
        plt.plot(df.index[train_size+60:], df['Close'][train_size+60:], label='Actual Price')
        plt.plot(df.index[train_size+60:], predicted_prices, label='Predicted Price')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.title('LSTM Prediction')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(lstm_plot_path, format='png')
        plt.close()
        logging.info('LSTM prediction plot generated successfully.')
        
        return lstm_plot_path

    except Exception as e:
        logging.error(f'Error generating LSTM prediction: {e}')
        return {'error': str(e)}

def xgboost_prediction(filepath):
    """
    Generates predictions using XGBoost and saves the plot.
    """
    try:
        df = pd.read_csv(filepath)
        df['Date'] = pd.to_datetime(df['Date'])
        
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df = df.dropna()
        
        X = df[['SMA_10', 'SMA_50']]
        y = df['Close']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        model_xgb = XGBRegressor(n_estimators=100, learning_rate=0.1)
        model_xgb.fit(X_train, y_train)
        
        predicted_prices_xgb = model_xgb.predict(X_test)
        
        xgboost_plot_path = os.path.join('uploads', 'xgboost_prediction.png')
        plt.figure(figsize=(12, 6))
        plt.plot(y_test.values, label='Actual Price')
        plt.plot(predicted_prices_xgb, label='Predicted Price')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.title('XGBoost Prediction')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(xgboost_plot_path, format='png')
        plt.close()
        logging.info('XGBoost prediction plot generated successfully.')
        
        return xgboost_plot_path

    except Exception as e:
        logging.error(f'Error generating XGBoost prediction: {e}')
        return {'error': str(e)}