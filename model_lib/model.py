import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization,Bidirectional
from tensorflow.keras.optimizers import Adam
#from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import yfinance as yf
from tensorflow.keras.regularizers import l2


def build_lstm_model(input_shape, output_shape, hidden_dim1: int = 128, hidden_dim2: int = 64, hidden_dim3: int = 32, dense_dim: int = 16, reg_lam:float =0.001, dropout1: float = 0.3, dropout2: float = 0.25, dropout3: float = 0.2, lr: float = 0.0005):
    model = Sequential([
        Bidirectional(LSTM(hidden_dim1, activation="tanh", return_sequences=True, input_shape=input_shape, kernel_regularizer=l2(reg_lam))),
        Dropout(dropout1),
        LSTM(hidden_dim2, activation="tanh", return_sequences=True, kernel_regularizer=l2(reg_lam)),
        Dropout(dropout2),
        LSTM(hidden_dim3, activation="tanh", return_sequences=False, kernel_regularizer=l2(reg_lam)),
        Dropout(dropout3),
        Dense(dense_dim, activation='relu', kernel_regularizer=l2(reg_lam)),
        Dense(output_shape , activation ="linear")  # Output dimension will be determined in accordance with the total number of companies and their respective features.
    ])
    
    optimizer = Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='mse')
    
    return model

