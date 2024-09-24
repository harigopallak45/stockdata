import os
import pandas as pd
import logging
import matplotlib.pyplot as plt
from prophet import Prophet

def prophet_prediction(filepath):
    """Generates predictions using Prophet and saves the plot."""
    try:
        df = pd.read_csv(filepath)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.rename(columns={'Date': 'ds', 'Close': 'y'})

        model = Prophet()
        model.fit(df)

        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)

        plot_path = os.path.join('uploads', 'prophet_prediction.png')
        fig = model.plot(forecast)
        plt.savefig(plot_path, format='png')
        plt.close(fig)
        logging.info(f'Prophet prediction plot saved at {plot_path}')
        
        return plot_path

    except Exception as e:
        logging.error(f'Error generating Prophet prediction: {e}')
        return {'error': str(e)}
