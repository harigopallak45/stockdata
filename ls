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

# Ensure uploads directory exists
if not os.path.exists('uploads'):
    os.makedirs('uploads')

def log_plot_success(plot_path, model_name):
    try:
        if os.path.exists(plot_path):
            file_size = os.path.getsize(plot_path)
            if file_size > 0:
                logging.info(f'{model_name} plot saved successfully at {plot_path} (File size: {file_size} bytes).')
            else:
                logging.error(f'{model_name} plot at {plot_path} is empty (File size: 0 bytes).')
        else:
            logging.error(f'{model_name} plot at {plot_path} was not found.')
    except Exception as e:
        logging.error(f'Error while checking {model_name} plot file at {plot_path}: {e}')

def analyze_data(filepath):
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

        # Check predictions
        if not np.isfinite(predictions).all():
            raise ValueError('Predictions contain non-finite values.')

        # Month-wise data difference
        df['Month'] = df['Date'].dt.to_period('M')
        month_wise_diff = df.groupby('Month')['Close'].agg(['min', 'max']).reset_index()
        month_wise_diff['Difference'] = month_wise_diff['max'] - month_wise_diff['min']

        # Plotting month-wise differences
        month_wise_diff_plot_path = os.path.join('uploads', 'month_wise_diff_plot.png')
        try:
            plt.figure(figsize=(12, 6))
            sns.barplot(x='Month', y='Difference', data=month_wise_diff)
            plt.xlabel('Month')
            plt.ylabel('Difference')
            plt.title('Month-Wise Differences')
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.savefig(month_wise_diff_plot_path, format='png')
            plt.close()
            log_plot_success(month_wise_diff_plot_path, "Month-Wise Difference")
        except Exception as e:
            logging.error(f'Error generating Month-Wise Difference plot: {e}')

        # Plotting historical data and predictions
        historical_vs_predictions_plot_path = os.path.join('uploads', 'historical_vs_predictions_plot.png')
        try:
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
            log_plot_success(historical_vs_predictions_plot_path, "Historical vs Predictions")
        except Exception as e:
            logging.error(f'Error generating Historical vs Predictions plot: {e}')

        return {
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
    try:
        # Load data
        df = pd.read_csv(filepath)
        logging.info(f'Prophet model data loaded with {len(df)} rows.')

        # Ensure required columns exist
        required_columns = ['Date', 'Close']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f'Missing required column: {col}')

        df['Date'] = pd.to_datetime(df['Date'])
        df = df.rename(columns={'Date': 'ds', 'Close': 'y'})

        # Fit the Prophet model
        model = Prophet()
        model.fit(df)

        # Create future dates for prediction
        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)

        # Check for valid predictions
        if not np.isfinite(forecast['yhat']).all():
            raise ValueError('Prophet predictions contain non-finite values.')

        # Plotting the forecast
        prophet_plot_path = os.path.join('uploads', 'prophet_prediction.png')
        try:
            fig = model.plot(forecast)
            plt.title('Prophet Stock Price Prediction')
            plt.savefig(prophet_plot_path, format='png')
            plt.close()
            log_plot_success(prophet_plot_path, "Prophet Prediction")
        except Exception as e:
            logging.error(f'Error generating Prophet prediction plot: {e}')
            raise

        return prophet_plot_path

    except ValueError as ve:
        logging.error(f'Value error during Prophet prediction: {ve}')
        return {'error': str(ve)}
    except Exception as e:
        logging.error(f'Error during Prophet prediction: {e}')
        return {'error': str(e)}

def lstm_prediction(filepath):
    try:
        # Load data
        df = pd.read_csv(filepath)
        logging.info(f'LSTM model data loaded with {len(df)} rows.')

        # Ensure required columns exist
        required_columns = ['Date', 'Close']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f'Missing required column: {col}')

        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)

        # Scale the data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df[['Close']].values)

        # Prepare training data
        X, y = [], []
        for i in range(60, len(scaled_data)):
            X.append(scaled_data[i-60:i, 0])
            y.append(scaled_data[i, 0])
        X, y = np.array(X), np.array(y)

        # Reshape for LSTM input
        X = X.reshape(X.shape[0], X.shape[1], 1)

        # Define the LSTM model
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))  # Predicting the next closing value

        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model
        model.fit(X, y, epochs=50, batch_size=32)

        # Prepare for prediction
        inputs = scaled_data[len(scaled_data) - 60:]
        inputs = inputs.reshape(-1, 1)
        inputs = scaler.transform(inputs)

        X_test = []
        X_test.append(inputs)
        X_test = np.array(X_test)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        # Predict and inverse scale
        predicted_price = model.predict(X_test)
        predicted_price = scaler.inverse_transform(predicted_price)

        # Plotting the LSTM prediction
        lstm_plot_path = os.path.join('uploads', 'lstm_prediction.png')
        try:
            plt.figure(figsize=(12, 6))
            plt.plot(df.index, df['Close'], label='Historical Prices', color='blue')
            plt.axvline(x=df.index[-1], color='red', linestyle='--', label='Prediction Start')
            plt.plot(df.index[-1] + pd.DateOffset(days=1), predicted_price[0], label='Predicted Price', color='orange')
            plt.title('LSTM Stock Price Prediction')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()
            plt.savefig(lstm_plot_path, format='png')
            plt.close()
            log_plot_success(lstm_plot_path, "LSTM Prediction")
        except Exception as e:
            logging.error(f'Error generating LSTM prediction plot: {e}')
            raise

        return lstm_plot_path

    except ValueError as ve:
        logging.error(f'Value error during LSTM prediction: {ve}')
        return {'error': str(ve)}
    except Exception as e:
        logging.error(f'Error during LSTM prediction: {e}')
        return {'error': str(e)}

def xgboost_prediction(filepath):
    try:
        # Load data
        df = pd.read_csv(filepath)
        logging.info(f'XGBoost model data loaded with {len(df)} rows.')

        # Ensure required columns exist
        required_columns = ['Date', 'Close']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f'Missing required column: {col}')

        # Convert Date to datetime and create features
        df['Date'] = pd.to_datetime(df['Date'])
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day

        # Define features and target
        X = df[['Year', 'Month', 'Day']]
        y = df['Close']

        # Split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define and train the XGBoost model
        model = XGBRegressor(objective='reg:squarederror')
        model.fit(X_train, y_train)

        # Make predictions
        predictions = model.predict(X_test)

        # Check for valid predictions
        if not np.isfinite(predictions).all():
            raise ValueError('XGBoost predictions contain non-finite values.')

        # Calculate and log the mean squared error
        mse = mean_squared_error(y_test, predictions)
        logging.info(f'Mean Squared Error for XGBoost: {mse}')

        # Plotting the predictions
        xgboost_plot_path = os.path.join('uploads', 'xgboost_prediction.png')
        try:
            plt.figure(figsize=(12, 6))
            plt.plot(df['Date'], df['Close'], label='Historical Prices', color='blue')
            plt.scatter(X_test['Year'] + (X_test['Month'] - 1) / 12, predictions, label='Predicted Prices', color='orange')
            plt.title('XGBoost Stock Price Prediction')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()
            plt.savefig(xgboost_plot_path, format='png')
            plt.close()
            log_plot_success(xgboost_plot_path, "XGBoost Prediction")
        except Exception as e:
            logging.error(f'Error generating XGBoost prediction plot: {e}')
            raise

        return xgboost_plot_path

    except ValueError as ve:
        logging.error(f'Value error during XGBoost prediction: {ve}')
        return {'error': str(ve)}
    except Exception as e:
        logging.error(f'Error during XGBoost prediction: {e}')
        return {'error': str(e)}
    
def run_all_models(filepath):
    """
    Runs all prediction models and returns only the paths of successfully generated plots.
    """
    results = {}
    
    # Running Prophet model
    prophet_path = prophet_prediction(filepath)
    if prophet_path:
        results['prophet'] = prophet_path
    else:
        logging.error('Prophet model failed to generate the plot.')
    
    # Running LSTM model
    lstm_path = lstm_prediction(filepath)
    if lstm_path:
        results['lstm'] = lstm_path
    else:
        logging.error('LSTM model failed to generate the plot.')
    
    # Running XGBoost model
    xgboost_path = xgboost_prediction(filepath)
    if xgboost_path:
        results['xgboost'] = xgboost_path
    else:
        logging.error('XGBoost model failed to generate the plot.')
    
    return results
