import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, Input
from tensorflow.keras.callbacks import EarlyStopping

def build_lstm_autoencoder(window_size: int, n_features: int) -> Sequential:
    """Builds an LSTM Autoencoder model."""
    model = Sequential([
        Input(shape=(window_size, n_features)),
        LSTM(64, activation='tanh', return_sequences=False),
        RepeatVector(window_size),
        LSTM(64, activation='tanh', return_sequences=True),
        TimeDistributed(Dense(n_features))
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def train_autoencoder(model: Sequential, X_train: np.ndarray, epochs=50, batch_size=32):
    """Trains the LSTM Autoencoder."""
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(
        X_train, X_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1
    )
    return history

def compute_reconstruction_error(model: Sequential, X: np.ndarray) -> np.ndarray:
    """Computes MSE reconstruction error."""
    X_pred = model.predict(X)
    mse = np.mean(np.square(X - X_pred), axis=(1, 2))
    return mse
