# Stock Data Dashboard

## Overview

The **Stock Data Dashboard** is a Flask-based web application designed to provide a comprehensive view of stock market data. The app allows users to upload stock datasets in CSV format or download stock data directly from the internet, visualize historical data, and predict future stock prices using multiple forecasting methods like Linear Regression, Prophet, LSTM, and XGBoost.

## Features

- **File Upload**: Upload stock data in CSV format for analysis.
- **Data Download**: Option to download stock data directly using stock ticker symbols.
- **Live Data**: Display live stock data for the selected company.
- **Historical Data Plotting**: Visualize historical stock data as line plots.
- **Forecasting Methods**:
  - Linear Regression
  - Prophet
  - LSTM (Long Short-Term Memory)
  - XGBoost
- **Month-wise Differences**: Displays month-wise stock price differences for further analysis.
  
## Project Directory Structure

```plaintext
stock_prediction_project/
|
├── uploads/                    # Directory for storing uploaded files and plots
├── static/                     # Directory for static files (CSS, JavaScript, images)
│   ├── script.js               # JavaScript files
│   ├── style.css               # CSS files
│   
├── templates/                  # Directory for HTML templates
│   ├── index.html              # Home page with file upload and prediction button
│   
├── app.py                      # Main Flask application
├── prediction.py               # Contains functions for predictions and plotting
├── live.py                     # Functions to fetch and handle live stock data
├── historical.py               # Functions to handle and plot historical data
├── get_company_info.py         # Function to extract company info based on dataset
├── company_info.csv            # CSV file with company information
└── plots/                      # Directory for storing prediction plots and images
```

## Setup Instructions

### Prerequisites

Before running this application, ensure you have the following installed:

- Python 3.x
- Flask
- Required Python packages (install using `requirements.txt`)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/stock_prediction_project.git
   cd stock_prediction_project
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Flask application:
   ```bash
   flask run
   ```

4. Access the web app in your browser:
   ```
   http://127.0.0.1:5000/
   ```

## Usage

### Upload Dataset

1. On the home page, upload your stock dataset in CSV format.
2. The app will automatically process the dataset, showing company info, live data, and historical plots.
3. Use the available options to download stock data if you do not have a dataset.

### Download Stock Data

1. Click the "Download Stock Data" button on the homepage.
2. Enter the stock ticker symbol, start date, and end date.
3. The application will download the dataset and use it for analysis.

### Prediction Methods

Once the dataset is uploaded or downloaded, the app will:
- Display historical data.
- Use multiple forecasting methods to predict future stock trends.

Prediction methods supported:
- **Linear Regression**: Predicts future stock prices based on historical data.
- **Prophet**: Time-series forecasting tool for seasonality.
- **LSTM**: Recurrent neural network method for predicting stock prices.
- **XGBoost**: Gradient boosting method optimized for prediction accuracy.

## Contributing

We welcome contributions to enhance the features of this dashboard.

1. Fork the repository.
2. Create your feature branch:
   ```bash
   git checkout -b feature/YourFeature
   ```
3. Commit your changes:
   ```bash
   git commit -m 'Add some feature'
   ```
4. Push to the branch:
   ```bash
   git push origin feature/YourFeature
   ```
5. Open a pull request.

## Contributors

- **Hariharan** - [GitHub Profile](https://github.com/harigopallak45)
- **Chaitra** - [GitHub Profile](https://github.com/chaithra1404)

## License

This project is licensed under the MIT License.
