import os
import pandas as pd
import logging
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from sklearn.preprocessing import MinMaxScaler

def lstm_prediction(filepath):
    """Generates predictions using LSTM and saves the plot."""
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

        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))

        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

        predicted_prices = model.predict(X_test)
        predicted_prices = scaler.inverse_transform(predicted_prices)

        plot_path = os.path.join('uploads', 'lstm_prediction.png')
        plt.figure(figsize=(12, 6))
        plt.plot(df.index[train_size + 60:], df['Close'][train_size + 60:], label='Actual Price')
        plt.plot(df.index[train_size + 60:], predicted_prices, label='Predicted Price')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.title('LSTM Prediction')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plot_path, format='png')
        plt.close()
        logging.info(f'LSTM prediction plot saved at {plot_path}')
        return plot_path

    except Exception as e:
        logging.error(f'Error generating LSTM prediction: {e}')
        return {'error': str(e)}
