import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os

# Define the folder to save the plots
uploads_folder = '.'

# LSTM Prediction for next 365 days (1 year)
def lstm_prediction(filepath):
    try:
        # Load and preprocess data
        df = pd.read_csv(filepath)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)

        # Scale the 'Close' price
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df[['Close']])

        # Split the data into training and testing sets
        train_size = int(len(scaled_data) * 0.8)
        train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

        # Function to create sequences of data
        def create_sequences(data, steps):
            X, y = [], []
            for i in range(steps, len(data)):
                X.append(data[i-steps:i])
                y.append(data[i])
            return np.array(X), np.array(y)

        # Prepare the training and testing data with longer sequences
        X_train, y_train = create_sequences(train_data, 100)  # Using 100 instead of 60 days
        X_test, y_test = create_sequences(test_data, 100)

        # Reshape input data to be 3D (samples, timesteps, features)
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

        # Build the LSTM model
        model_lstm = Sequential()
        model_lstm.add(LSTM(units=64, return_sequences=True, input_shape=(X_train.shape[1], 1)))  # More units
        model_lstm.add(Dropout(0.3))    # Increased dropout
        model_lstm.add(LSTM(units=64))  # Second LSTM layer
        model_lstm.add(Dropout(0.3))
        model_lstm.add(Dense(units=1))

        # Compile the model with a lower learning rate
        model_lstm.compile(optimizer='adam', loss='mean_squared_error')

        # Early stopping and model checkpoint to save the best model
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        checkpoint = ModelCheckpoint('best_lstm_model.keras', save_best_only=True, monitor='val_loss')

        # Train the model with validation set
        history = model_lstm.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test),
                                 callbacks=[early_stop, checkpoint], verbose=1)

        # Predict on the test data
        predicted_prices = model_lstm.predict(X_test)
        predicted_prices = scaler.inverse_transform(predicted_prices)

        # Predict the next 365 days (1 year)
        future_predictions = []
        last_100_days = scaled_data[-100:]  # Last 100 days from the original dataset
        current_input = last_100_days.reshape(1, -1, 1)  # Reshape for the LSTM input

        for _ in range(365):  # Predict the next 365 days
            next_predicted = model_lstm.predict(current_input)  # Predict next day
            future_predictions.append(next_predicted[0][0])  # Store the predicted value

            # Reshape next_predicted to match input shape
            next_predicted = np.reshape(next_predicted, (1, 1, 1))  # Reshape to (1, 1, 1)

            # Update the input for the next prediction
            current_input = np.append(current_input[:, 1:, :], next_predicted, axis=1)

        # Transform predicted values back to original scale
        future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

        # Extend the dates for future predictions
        last_date = df.index[-1]
        future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, 366)]  # Next 365 days

        # Plot the actual, predicted, and future predictions
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df.index[train_size+100:], df['Close'][train_size+100:], label='Actual Price')
        ax.plot(df.index[train_size+100:], predicted_prices, label='Predicted Price', color='red')
        ax.plot(future_dates, future_predictions, label='Future Predicted Price (Next 1 Year)', color='green', linestyle='dashed')
        ax.set_xlabel('Date')
        ax.set_ylabel('Close Price')
        ax.set_title('LSTM Prediction with Future Forecast (Next 1 Year)')
        ax.legend()
        ax.grid(True)

        # Save the plot
        plot_filename = os.path.join(uploads_folder, 'lstm_prediction_future_1_year.png')
        plt.savefig(plot_filename)
        logging.info(f'LSTM future prediction plot saved as {plot_filename}')
        plt.show()

    except Exception as e:
        logging.error(f'Error generating LSTM future prediction: {e}')

def run_all_predictions(filepath):
    lstm_prediction(filepath)

if __name__ == '__main__':
    filepath = 'IBM.csv'
    run_all_predictions(filepath)
