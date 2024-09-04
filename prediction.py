import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import json
import pandas as pd

def analyze_data(filepath):
    df = pd.read_csv(filepath)
    # Example analysis (replace with actual analysis)
    historical_data = df[['Date', 'Close']].to_dict(orient='records')
    prediction_data = [{'Date': '2024-01-01', 'Predicted_Close': 150}]  # Example prediction
    month_wise_diff = {'January': 2.5, 'February': -1.2}  # Example month-wise data
    return historical_data, prediction_data, month_wise_diff

def get_company_info(filepath):
    df = pd.read_csv(filepath)
    company_name = df['Company'].iloc[0] if 'Company' in df.columns else 'Unknown'
    description = 'No description available'
    return {'company_name': company_name, 'description': description}


def get_company_info(filepath):
    """
    Extracts company information from the dataset.
    Assumes the dataset contains columns like 'Company', 'Description', etc.
    """
    try:
        df = pd.read_csv(filepath)
        company_name = df['Company'].iloc[0]  # Example
        description = df.get('Description', 'No description available').iloc[0]
        return {
            'company_name': company_name,
            'description': description
        }
    except Exception as e:
        return {'error': str(e)}


def analyze_data(filepath):
    """
    Analyzes the dataset to generate historical data, predictions, and month-wise differences.
    Assumes dataset contains columns like 'Date', 'Close', etc.
    """
    try:
        df = pd.read_csv(filepath)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')

        # Historical data
        historical_data = df[['Date', 'Close']].to_dict(orient='records')

        # Prepare data for prediction
        df['Date'] = df['Date'].map(pd.Timestamp.toordinal)
        X = df[['Date']].values
        y = df['Close'].values

        # Scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Linear Regression Model
        model = LinearRegression()
        model.fit(X_scaled, y)

        # Prediction
        future_dates = pd.date_range(start=df['Date'].max() + pd.Timedelta(days=1), periods=30)
        future_dates_ordinal = future_dates.map(pd.Timestamp.toordinal).values.reshape(-1, 1)
        future_dates_scaled = scaler.transform(future_dates_ordinal)
        predictions = model.predict(future_dates_scaled)

        prediction_data = [{'Date': date, 'Prediction': pred} for date, pred in zip(future_dates, predictions)]

        # Month-wise data difference
        df['Month'] = df['Date'].dt.to_period('M')
        month_wise_diff = df.groupby('Month')['Close'].agg(['min', 'max']).reset_index()
        month_wise_diff['Difference'] = month_wise_diff['max'] - month_wise_diff['min']
        month_wise_diff = month_wise_diff.to_dict(orient='records')

        return historical_data, prediction_data, month_wise_diff

    except Exception as e:
        return {'error': str(e)}

# You can add more functions here as needed
