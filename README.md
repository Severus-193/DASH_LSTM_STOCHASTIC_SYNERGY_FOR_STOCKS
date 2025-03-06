# DASH_LSTM_STOCHASTIC_SYNERGY_FOR_STOCKS

## Project Overview
This project is a web-based application that integrates two DASH-LSTM models for stock prediction and time series analysis of NYSE stocks.This project leverages LSTM neural networks for stock prediction and time series analysis, providing interactive insights through a web-based interface using Dash. The integration of `yfinance` ensures access to real-time and historical financial data, enhancing the accuracy and reliability of the predictive models. The application is hosted in a webpage named `webpage.html` and includes the following components:

1. **Stock Prediction Model (`stock_prediction.py`)**
   - This model predicts stock prices using an LSTM (Long Short-Term Memory) network.
   - It relies on another model, `lstm.py`, to perform stock price predictions.
   - The dataset is obtained using the `yfinance` library, which provides real-time and historical market data for financial analysis.
   - The model processes stock price time series data, normalizes it, and trains an LSTM network to predict future stock values based on historical trends.

2. **Time Series Analysis Model (`time_series.py`)**
   - This model performs time series analysis on NYSE stocks, providing insights into stock trends over time.
   - It utilizes data retrieved from `yfinance` to analyze stock performance, volatility, and trend patterns.
   - Advanced time series decomposition techniques, including moving averages and statistical smoothing, are applied to identify key stock trends.

## Project Files
- `webpage.html`: The front-end webpage that hosts the stock prediction and time series analysis models.
- `stock_prediction.py`: Implements stock price prediction using LSTM and integrates with `lstm.py`.
- `lstm.py`: Contains the core LSTM model for stock price prediction, including the architecture design with multiple LSTM layers, dropout layers to prevent overfitting, and a fully connected output layer.
- `time_series.py`: Implements time series analysis on NYSE stock data, utilizing statistical and machine learning methods for trend extraction and visualization.

## Features
- **Stock Price Prediction**: Predict future stock prices using a deep learning-based LSTM model trained on historical data.
- **Time Series Analysis**: Perform detailed time series analysis using statistical and machine learning approaches.
- **Real-Time Data Retrieval**: Leverage `yfinance` to fetch up-to-date stock price data for dynamic predictions and analysis.
- **DASH Integration**: A user-friendly web interface using Dash for interactive visualization of stock trends and predictions.
- **Data Preprocessing**: Implements feature scaling, trend extraction, and time series segmentation for enhanced model accuracy.
- **Visualization**: Graphical representation of stock trends, moving averages, and model predictions using Dash and Plotly.

## How to Run
1. Install the required dependencies:
   ```sh
   pip install dash pandas numpy tensorflow keras yfinance plotly
   ```
2. Run the Dash applications:
   ```sh
   python stock_prediction.py
   python time_series.py
   ```
3. Open `webpage.html` in a web browser to access the application.

## Requirements
- Python 3.11
- Dash
- TensorFlow/Keras
- Pandas
- NumPy
- `yfinance` for real-time stock data retrieval
- Plotly for advanced data visualization



