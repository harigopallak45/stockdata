import logging
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for saving plots
import matplotlib.pyplot as plt
from flask import Flask, request, render_template, jsonify, send_file
from werkzeug.utils import secure_filename
import yfinance as yf
from prophet import Prophet
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from sklearn.model_selection import *
from prediction.lstm_prediction import *
from prediction.prophet_prediction import *
from prediction.xgboost_prediction import *
from prediction.linear_regression_prediction import *
from prediction.data_analysis import analyze_data
from prediction.historical import *
from get_company_info import get_company_info
from live import get_live_stock_data

# Initialize the Flask app
app = Flask(__name__)

# Directory to store uploaded files and stock files
UPLOAD_FOLDER = 'uploads'
DOWNLOAD_FOLDER = 'download'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DOWNLOAD_FOLDER'] = DOWNLOAD_FOLDER
ALLOWED_EXTENSIONS = {'csv'}

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)

# Load company info CSV file once
COMPANY_INFO_PATH = 'company_info.csv'

# Utility to check if file type is CSV
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        try:
            logging.info(f'Processing file: {file_path}')
            
            # Fetch company info using the uploaded file name
            company_info = get_company_info(COMPANY_INFO_PATH, filename)
            logging.info(f'Company info: {company_info}')

            if 'error' in company_info:
                raise ValueError(company_info['error'])

            ticker = company_info.get('Ticker', None)

            if ticker:
                live_data_url = f'https://finance.yahoo.com/quote/{ticker}/'  # Updated URL format
                live_data = get_live_stock_data(live_data_url, {'symbol': ticker})
            else:
                live_data = {'error': 'Ticker not found in company info'}

            # Analyze the data
            result = analyze_data(file_path)
            if result == True:
                logging.info('Data Fetched successfully')
            else:
                logging.info('Data Fetch failed')
            logging.info(f'Live data is {result}')

            if 'error' in result:
                raise ValueError(result['error'])

            # Return plot URLs and data
            return jsonify({
                'company_info': company_info,
                'historical_data_plot_url': '/plot_image',
                'month_wise_diff_plot_url': '/month_wise_diff_plot',
                'live_data': live_data,
                'prophet_plot_url': '/prophet_prediction',
                'lstm_plot_url': '/lstm_prediction',
                'xgboost_plot_url': '/xgboost_prediction'
            })

        except Exception as e:
            logging.error(f'Error during file processing: {e}')
            return jsonify({'error': str(e)}), 500
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)
    else:
        return jsonify({'error': 'Invalid file format'}), 400

@app.route('/prophet_prediction', methods=['POST'])
def prophet_prediction_route():
    """Endpoint to generate and serve Prophet prediction plot."""
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_file.csv')  # Use the correct uploaded file path
        plot_path = os.path.join(app.config['UPLOAD_FOLDER'], 'prophet_prediction.png')
        if not os.path.exists(plot_path):
            logging.info("Creating new Prophet prediction plot.")
            plot_path = prophet_prediction(file_path)  # Pass the file path

        return send_file(plot_path, mimetype='image/png')
    except Exception as e:
        logging.error(f'Error generating Prophet prediction: {e}')
        return jsonify({'error': str(e)}), 500

@app.route('/lstm_prediction', methods=['POST'])
def lstm_prediction_route():
    """Endpoint to generate and serve LSTM prediction plot."""
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_file.csv')  # Use the correct uploaded file path
        plot_path = os.path.join(app.config['UPLOAD_FOLDER'], 'lstm_prediction.png')
        if not os.path.exists(plot_path):
            logging.info("Creating new LSTM prediction plot.")
            plot_path = lstm_prediction(file_path)  # Pass the file path

        return send_file(plot_path, mimetype='image/png')
    except Exception as e:
        logging.error(f'Error generating LSTM prediction: {e}')
        return jsonify({'error': str(e)}), 500

@app.route('/xgboost_prediction', methods=['POST'])
def xgboost_prediction_route():
    """Endpoint to generate and serve XGBoost prediction plot."""
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_file.csv')  # Use the correct uploaded file path
        plot_path = os.path.join(app.config['UPLOAD_FOLDER'], 'xgboost_prediction.png')
        if not os.path.exists(plot_path):
            logging.info("Creating new XGBoost prediction plot.")
            plot_path = xgboost_prediction(file_path)  # Pass the file path

        return send_file(plot_path, mimetype='image/png')
    except Exception as e:
        logging.error(f'Error generating XGBoost prediction: {e}')
        return jsonify({'error': str(e)}), 500

@app.route('/plot_image')
def plot_image():
    """Endpoint to serve the historical vs predictions plot image."""
    try:
        plot_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'historical_vs_predictions_plot.png')
        if os.path.exists(plot_image_path):
            return send_file(plot_image_path, mimetype='image/png')
        else:
            return jsonify({'error': 'Plot image not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/month_wise_diff_plot')
def month_wise_diff_plot():
    """Endpoint to serve the month-wise differences plot image."""
    try:
        plot_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'month_wise_diff_plot.png')
        if os.path.exists(plot_image_path):
            return send_file(plot_image_path, mimetype='image/png')
        else:
            return jsonify({'error': 'Month-wise differences plot image not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def download_stock_data(ticker, start_date, end_date):
    """Download stock data from Yahoo Finance."""
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    file_name = f"{ticker}.csv"
    file_path = os.path.join(app.config['DOWNLOAD_FOLDER'], file_name)
    stock_data.to_csv(file_path, index=True)  # Save with index to include dates
    return file_path

@app.route('/download_stock_data', methods=['POST'])
def download_stock():
    data = request.get_json()
    ticker = data.get('ticker')
    start_date = data.get('start_date')
    end_date = data.get('end_date')

    try:
        file_path = download_stock_data(ticker, start_date, end_date)
        return jsonify({'message': f"Stock data for {ticker} downloaded successfully.", 'file_path': file_path})
    except Exception as e:
        logging.error(f'Error downloading stock data: {e}')
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    app.run(debug=True)
