import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

def get_realtime_data(ticker, interval='1d', range='5y'):
    data = yf.download(tickers=ticker, interval=interval, period=range)
    return data

def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    return scaled_data, scaler

def create_dataset(dataset, look_back=1, days=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back - days):
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back:i + look_back + days, 0])
    return np.array(X), np.array(Y)

def create_transformer_model(input_shape, num_heads=2, num_layers=2, d_model=32, output_units=1):
    inputs = keras.layers.Input(shape=input_shape)
    x = inputs
    for _ in range(num_layers):
        x = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
    x = keras.layers.Flatten()(x)
    outputs = keras.layers.Dense(output_units)(x)
    model = keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def predict_prices(model, data, scaler, look_back):
    last_data = data.tail(look_back)
    last_data_scaled = scaler.transform(last_data['Close'].values.reshape(-1, 1))
    X_test = np.array([last_data_scaled])
    prediction = model.predict(X_test)
    prediction = scaler.inverse_transform(prediction)
    return prediction[0][0]

def main():
    symbol = input("Enter sales symbol: ")
    days = int(input("Enter number of days to predict: "))  # Added missing closing parenthesis

    data = get_realtime_data(symbol)
    scaled_data, scaler = preprocess_data(data)
    X, Y = create_dataset(scaled_data, look_back=10, days=1)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    model = create_transformer_model(input_shape=(X.shape[1], 1))
    model.fit(X, Y, batch_size=64, epochs=10, verbose=1)
    predicted_price = predict_prices(model, data, scaler, look_back=10)
    print(f"Predicted price for {symbol} in {days} days: {predicted_price:.2f}")

    # Calculate MSE and MAE
    y_true = data['Close'][-days:].values
    y_pred = [predict_prices(model, data, scaler, look_back=10) for _ in range(days)]
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(data.index[-days:], y_true, label='True Price')
    plt.plot(data.index[-days:], y_pred, label='Predicted Price')
    plt.title('True vs Predicted Prices')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

    # Scatter graph
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_true, y=y_pred)
    plt.title('True vs Predicted Prices - Scatter Plot')
    plt.xlabel('True Price')
    plt.ylabel('Predicted Price')
    plt.show()

if __name__ == "__main__":
    main()
