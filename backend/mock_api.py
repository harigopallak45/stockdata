from flask import Flask, jsonify
from datetime import datetime
import random

app = Flask(__name__)

@app.route('/live_stock_data', methods=['GET'])
def live_stock_data():
    """
    Returns mock live stock data.
    """
    # Simulate a timestamp and random price
    data = {
        'timestamp': datetime.now().isoformat(),
        'price': round(random.uniform(100, 500), 2)  # Random price between 100 and 500
    }
    return jsonify(data)

if __name__ == '__main__':
    app.run(port=5001, debug=True)
