import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))
    return scaled_data, scaler

def create_dataset(data, look_back=1):
    x, y = [], []
    for i in range(len(data) - look_back):
        x.append(data[i:(i + look_back), 0])
        y.append(data[i + look_back, 0])
    return np.array(x), np.array(y)

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(input_shape, 1)))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_lstm_model(data, look_back=1, epochs=10):
    scaled_data, scaler = preprocess_data(data)
    x, y = create_dataset(scaled_data, look_back)
    x = np.reshape(x, (x.shape[0], x.shape[1], 1))

    model = build_lstm_model(look_back)
    model.fit(x, y, epochs=epochs, batch_size=1, verbose=2)

    return model, scaler

def predict_close_price(model, scaler, data, look_back=1):
    last_data = data[-look_back:]
    last_data_scaled, _ = preprocess_data(last_data)
    x_input = np.array([last_data_scaled])
    x_input = np.reshape(x_input, (1, look_back, 1))

    predicted_price_scaled = model.predict(x_input)
    predicted_price = scaler.inverse_transform(predicted_price_scaled)

    return predicted_price[0, 0]
