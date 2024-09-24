import os
import pandas as pd
import logging
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_data(filepath):
    """
    Analyzes the dataset to generate historical data, month-wise differences, and plots.
    Assumes dataset contains columns like 'Date', 'Close', etc.
    """
    try:
        df = pd.read_csv(filepath)
        logging.info(f'Dataset loaded successfully with {len(df)} rows.')

        required_columns = ['Date', 'Close']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f'Missing required column: {col}')

        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')

        historical_data = df[['Date', 'Close']].to_dict(orient='records')
        logging.info('Historical data extracted successfully.')

        # Month-wise data difference
        df['Month'] = df['Date'].dt.to_period('M')
        month_wise_diff = df.groupby('Month')['Close'].agg(['min', 'max']).reset_index()
        month_wise_diff['Difference'] = month_wise_diff['max'] - month_wise_diff['min']
        
        plot_path = os.path.join('uploads', 'month_wise_diff_plot.png')
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Month', y='Difference', data=month_wise_diff)
        plt.xlabel('Month')
        plt.ylabel('Difference')
        plt.title('Month-Wise Differences')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(plot_path, format='png')
        plt.close()
        logging.info(f'Month-wise differences plot saved at {plot_path}.')

        return {
            'historical_data': historical_data,
            'month_wise_diff_plot': plot_path
        }

    except ValueError as ve:
        logging.error(f'Value error during data analysis: {ve}')
        return {'error': str(ve)}
    except Exception as e:
        logging.error(f'Error during data analysis: {e}')
        return {'error': str(e)}
