from flask import Flask, render_template, request, jsonify, send_from_directory, send_file
import os
import logging
import io
from backend.prediction import analyze_data
from backend.live import get_live_stock_data
from backend.historical import plot_historical_data
from backend.get_company_info import get_company_info
from flask_httpauth import HTTPBasicAuth

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set up basic authentication
auth = HTTPBasicAuth()

# Define users for authentication
users = {
    "admin": "password"  # Username: Password
}

@auth.get_password
def get_password(username):
    if username in users:
        return users.get(username)
    return None

# Directory to store uploaded datasets and generated plots
UPLOAD_FOLDER = 'uploads'
PLOTS_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PLOTS_FOLDER'] = PLOTS_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists(PLOTS_FOLDER):
    os.makedirs(PLOTS_FOLDER)

def cleanup_old_plots():
    """
    Removes old plot PNG files from the PLOTS_FOLDER directory.
    """
    for filename in os.listdir(app.config['PLOTS_FOLDER']):
        if filename.endswith('.png'):
            file_path = os.path.join(app.config['PLOTS_FOLDER'], filename)
            try:
                os.remove(file_path)
                app.logger.info(f'Removed old plot file: {file_path}')
            except Exception as e:
                app.logger.error(f'Error removing file {file_path}: {e}')

# Run cleanup on server start
cleanup_old_plots()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/fetch_dataset', methods=['POST'])
def fetch_dataset():
    if 'file' not in request.files:
        app.logger.error('No file part in request')
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        app.logger.error('No selected file')
        return jsonify({'error': 'No selected file'}), 400

    try:
        if file:
            filename = 'dataset.csv'
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            company_info = get_company_info(filepath)
            return jsonify({'company_info': company_info})
    except Exception as e:
        app.logger.error(f'Error processing file: {e}')
        return jsonify({'error': 'File upload failed'}), 500

@app.route('/run_analysis', methods=['POST'])
def run_analysis():
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'dataset.csv')
    if not os.path.exists(filepath):
        return jsonify({'error': 'Dataset not found'}), 400

    try:
        # Perform data analysis
        historical_data, prediction_data, month_wise_diff, plot_buf = analyze_data(filepath)
        
        # Save plot buffer to a file
        plot_path = os.path.join(app.config['PLOTS_FOLDER'], 'plot.png')
        with open(plot_path, 'wb') as f:
            f.write(plot_buf.getvalue())

        return jsonify({
            'historical_data': historical_data,
            'prediction_data': prediction_data,
            'month_wise_diff': month_wise_diff
        })
    except Exception as e:
        app.logger.error(f'Error analyzing data: {e}')
        return jsonify({'error': 'Data analysis failed'}), 500

@app.route('/live_data', methods=['GET'])
def live_data():
    try:
        api_url = "http://api.example.com/live_stock_data"  # Replace with actual API URL
        params = {"symbol": "AAPL"}  # Replace with actual parameters
        live_data = get_live_stock_data(api_url, params)
        return jsonify(live_data)
    except Exception as e:
        app.logger.error(f'Error fetching live data: {e}')
        return jsonify({'error': 'Live data fetch failed'}), 500

@app.route('/plot', methods=['GET'])
def plot():
    try:
        plot_path = os.path.join(app.config['PLOTS_FOLDER'], 'plot.png')

        if not os.path.exists(plot_path):
            return jsonify({'error': 'Plot not found'}), 404

        return send_file(plot_path, mimetype='image/png', as_attachment=False, attachment_filename='plot.png')
    except Exception as e:
        app.logger.error(f'Error generating plot: {e}')
        return jsonify({'error': 'Plot generation failed'}), 500

@app.route('/download/<filename>', methods=['GET'])
@auth.login_required
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
