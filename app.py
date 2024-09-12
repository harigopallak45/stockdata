import logging
import os
from flask import Flask, request, render_template, jsonify, send_file
from werkzeug.utils import secure_filename
from historical import plot_historical_data
from prediction import analyze_data
from live import get_live_stock_data
from get_company_info import get_company_info

# Initialize the Flask app
app = Flask(__name__)

# Directory to store uploaded files
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'csv'}

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

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
            company_info = get_company_info(COMPANY_INFO_PATH, filename)
            logging.info(f'Company info: {company_info}')

            if 'error' in company_info:
                raise ValueError(company_info['error'])

            ticker = company_info.get('Ticker', None)

            if ticker:
                live_data = get_live_stock_data('https://api.example.com/endpoint', {'symbol': ticker})
            else:
                live_data = {'error': 'Ticker not found in company info'}

            result = analyze_data(file_path)

            if 'error' in result:
                raise ValueError(result['error'])

            # Return plot URLs instead of JSON data
            return jsonify({
                'company_info': company_info,
                'historical_data_plot_url': '/plot_image',
                'month_wise_diff_plot_url': '/month_wise_diff_plot',
                'live_data': live_data
            })

        except Exception as e:
            logging.error(f'Error during file processing: {e}')
            return jsonify({'error': str(e)}), 500
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)
    else:
        return jsonify({'error': 'Invalid file format'}), 400

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

if __name__ == '__main__':
    app.run(debug=True)
