from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import pandas as pd
from prediction import analyze_data, get_company_info
from live import fetch_live_data

app = Flask(__name__)

# Directory to store uploaded datasets
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/fetch_dataset', methods=['POST'])
def fetch_dataset():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        filename = 'dataset.csv'
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Load dataset and get company info
        company_info = get_company_info(filepath)
        return jsonify({'company_info': company_info})

    return jsonify({'error': 'File upload failed'}), 400


@app.route('/run_analysis', methods=['POST'])
def run_analysis():
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'dataset.csv')
    if not os.path.exists(filepath):
        return jsonify({'error': 'Dataset not found'}), 400

    # Perform data analysis
    historical_data, prediction_data, month_wise_diff = analyze_data(filepath)
    return jsonify({
        'historical_data': historical_data,
        'prediction_data': prediction_data,
        'month_wise_diff': month_wise_diff
    })


@app.route('/live_data', methods=['GET'])
def live_data():
    # Fetch live stock data
    live_data = fetch_live_data()
    return jsonify(live_data)


@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True)
