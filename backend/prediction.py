import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import io

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def analyze_data(filepath):
    """
    Analyzes the dataset to generate historical data, predictions, and month-wise differences.
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
        logging.info('Model trained successfully.')

        # Prediction
        future_dates = pd.date_range(start=df['Date'].max() + pd.Timedelta(days=1), periods=30)
        future_dates_ordinal = future_dates.map(pd.Timestamp.toordinal).values.reshape(-1, 1)
        future_dates_scaled = scaler.transform(future_dates_ordinal)
        predictions = model.predict(future_dates_scaled)

        prediction_data = [{'Date': date.strftime('%Y-%m-%d'), 'Prediction': pred} for date, pred in
                           zip(future_dates, predictions)]
        logging.info('Predictions generated successfully.')

        # Month-wise data difference
        df['Month'] = df['Date'].dt.to_period('M')
        month_wise_diff = df.groupby('Month')['Close'].agg(['min', 'max']).reset_index()
        month_wise_diff['Difference'] = month_wise_diff['max'] - month_wise_diff['min']
        month_wise_diff = month_wise_diff.to_dict(orient='records')
        logging.info('Month-wise differences calculated successfully.')

        # Plotting
        buf = io.BytesIO()
        plt.figure(figsize=(12, 6))
        sns.lineplot(x='Date', y='Close', data=df, label='Historical Data')
        plt.plot(future_dates, predictions, label='Predictions', color='red')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.title('Stock Price Prediction')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(buf, format='png')
        buf.seek(0)
        logging.info('Plot generated successfully.')

        # Return data
        return historical_data, prediction_data, month_wise_diff, buf

    except ValueError as ve:
        logging.error(f'Value error during data analysis: {ve}')
        return {'error': str(ve)}
    except Exception as e:
        logging.error(f'Error during data analysis: {e}')
        return {'error': str(e)}
