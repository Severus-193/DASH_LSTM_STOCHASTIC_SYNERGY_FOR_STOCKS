import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import plotly.express as px
from flask import Flask, render_template_string

# Download historical stock data from Yahoo Finance
def download_stock_data(symbol, start_date, end_date):
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    return stock_data

# Normalize the data
def normalize_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))
    return scaled_data, scaler

# Create sequences for LSTM
def create_sequences(data, look_back):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back), 0])
        y.append(data[i + look_back, 0])
    return np.array(X), np.array(y)

# Build the LSTM model
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(input_shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Train the LSTM model
def train_lstm_model(model, X_train, y_train, epochs, batch_size, validation_data):
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=validation_data, verbose=1)

# Make predictions on the test set
def predict_prices(model, X_test):
    return model.predict(X_test)

# Inverse transform the predictions and actual values
def inverse_transform(scaler, X_test, y_test, predicted_prices_scaled):
    actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))
    predicted_prices = scaler.inverse_transform(predicted_prices_scaled.reshape(-1, 1))
    return actual_prices, predicted_prices

# Calculate and print the Mean Squared Error (MSE)
def calculate_mse(actual_prices, predicted_prices):
    mse = mean_squared_error(actual_prices, predicted_prices)
    print(f'Mean Squared Error (MSE): {mse}')
    return mse

# Function to plot results for multiple symbols using Plotly
def plot_results_plotly(stock_data_list, split_index, look_back, actual_prices_list, predicted_prices_list, symbol_list):
    plots = []
    for i, stock_data in enumerate(stock_data_list):
        actual_prices_flat = actual_prices_list[i].flatten()
        predicted_prices_flat = predicted_prices_list[i].flatten()

        actual_trace = go.Scatter(x=stock_data.index[split_index + look_back:], y=actual_prices_flat, mode='lines', name='Actual Prices', line=dict(color='blue'))
        predicted_trace = go.Scatter(x=stock_data.index[split_index + look_back:], y=predicted_prices_flat, mode='lines', name='Predicted Prices', line=dict(color='orange'))

        # Create layout for the plot
        layout = go.Layout(
            title=f'{symbol_list[i]} Time Series Analysis with LSTM',
            xaxis=dict(title='Date'),
            yaxis=dict(title='Stock Price (USD)'),
            hovermode='closest')
        
        # Create figure and add traces to it
        fig = go.Figure(data=[actual_trace, predicted_trace], layout=layout)


         # Convert Plotly figure to HTML with full HTML content
        plot_html_full = fig.to_html(full_html=True)
        plots.append(plot_html_full)
    
    return '\n'.join(plots)
# Define Flask app
app = Flask(__name__)

# HTML template for rendering the stock price predictions with Plotly plots
template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Stock Price Predictions</title>
    <style>
        h2 {
            color: #483D8B; /* Blue heading color */
        }
        p {
            color: #008080; /* Gray text color */
        }
     </style>
</head>
<body>
    {% for i in range(stock_symbols|length) %}
        <h2>{{ stock_symbols[i] }} Time Series Analysis with LSTM</h2>
        <div>
            <p>Mean Squared Error (MSE): {{ mse_values[stock_symbols[i]] }}</p>
            {{ plot_data_plotly[stock_symbols[i]]|safe }}
        </div>
        <hr>
    {% endfor %}
</body>
</html>
"""

# List of stock symbols
stock_symbols = ['AAPL', 'GOOGL', 'MSFT', 'BA','DIS','SBUX']  # Add more symbols as needed

# Parameters
start_date = '2015-02-01'
end_date = '2025-12-31'
look_back = 30
split_percentage = 0.8
epochs = 2
batch_size = 32

# Lists to store data for each symbol
stock_data_list = []
actual_prices_list = []
predicted_prices_list = []

for symbol in stock_symbols:
    # Download stock data
    stock_data = download_stock_data(symbol, start_date, end_date)

    # Extract closing prices
    data = stock_data['Close'].values.reshape(-1, 1)

    # Normalize the data
    scaled_data, scaler = normalize_data(data)

    # Create sequences for LSTM
    X, y = create_sequences(scaled_data, look_back)

    # Split the data into training and testing sets
    split_index = int(split_percentage * len(X))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # Reshape data for LSTM input (samples, time steps, features)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Build the LSTM model
    input_shape = X_train.shape
    lstm_model = build_lstm_model(input_shape)

    # Train the LSTM model
    train_lstm_model(lstm_model, X_train, y_train, epochs, batch_size, validation_data=(X_test, y_test))

    # Make predictions on the test set
    predicted_prices_scaled = predict_prices(lstm_model, X_test)

    # Inverse transform the predictions and actual values
    actual_prices, predicted_prices = inverse_transform(scaler, X_test, y_test, predicted_prices_scaled)

    # Calculate and print the Mean Squared Error (MSE)
    mse = calculate_mse(actual_prices, predicted_prices)

    # Store data for plotting
    stock_data_list.append(stock_data)
    actual_prices_list.append(actual_prices)
    predicted_prices_list.append(predicted_prices)

# Run Flask app
@app.route('/')
def show_predictions():
    mse_values = {symbol: calculate_mse(actual_prices, predicted_prices) for symbol, actual_prices, predicted_prices in zip(stock_symbols, actual_prices_list, predicted_prices_list)}
    plot_data_plotly = {symbol: plot_results_plotly([stock_data], split_index, look_back, [actual_prices], [predicted_prices], [symbol]) for symbol, stock_data, actual_prices, predicted_prices in zip(stock_symbols, stock_data_list, actual_prices_list, predicted_prices_list)}
    return render_template_string(template, stock_symbols=stock_symbols, mse_values=mse_values, plot_data_plotly=plot_data_plotly)

if __name__ == "__main__":
    app.run(debug=True)
