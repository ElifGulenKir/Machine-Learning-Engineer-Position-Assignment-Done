from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import numpy as np

from model_lib.model import build_lstm_model
from data.data_stock import get_stock_data 
from utils.prepare_data import prepare_train_test_data, get_input_output_dims
from utils.visualize_data import visualize_data
from utils.visualize_lstm_output import visualize_lstm_output
from tools.macd import get_macd, visualize_macd
from tools.rsi import get_rsi, visualize_rsi
from tools.bollinger_bands import bollinger_bands, visualize_bollinger_bands

#%%
#Getting stock data (initialization and preparing the data)
company_list = ["TSLA"]
start = "2020-01-01"
end = "2025-01-01"
features = ["Close","Low","Open","High","Volume"]


data = get_stock_data(company = company_list, start = start, end = end, features = features)


time_steps_for_prediction = 40 #days
scale_data = True
X_train, X_test, Y_train, Y_test, scaler = prepare_train_test_data(np.array(data), train_test_probability = 0.80, time_steps_for_prediction = time_steps_for_prediction, scale_data = scale_data)

visualize_data(data, company_list, features)
#%%
# Set training parameters
epoch = 100
hidden_dim1 = 50
hidden_dim2 =30
hidden_dim3 = 20
dense_dim = 10
lr = 0.0004

#Do not change unless needed
dropout1 = 0.2
dropout2 = 0.2
dropout3 = 0.15
patience = epoch // 3
reg_lam = 0.0001


X_dim, Y_dim = get_input_output_dims(X_train, Y_train, company_list)
# Initialize model
model = build_lstm_model(input_shape=(X_dim[0], X_dim[1]), output_shape = Y_dim, hidden_dim1 = hidden_dim1, hidden_dim2 = hidden_dim2, dense_dim = 25, lr=lr, reg_lam = reg_lam, dropout1 = dropout1, dropout2 = dropout2)

# Early Stopping Condition in accordance with the validation loss
early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)


#%%

# Train model
history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test),
                    epochs = epoch, batch_size=32, callbacks=[early_stopping],shuffle = False)


#%%

visualize_lstm_output(model = model, data = data, company_list = company_list, features = features, scaler = scaler, X_test = X_test, Y_test = Y_test,Y_train=Y_train, history = history, scale_data = scale_data)


#%%

feature_rsi = "Open"
combined = get_rsi(data = data, rsi_feature = feature_rsi, company_list = company_list, features = features)
visualize_rsi(combined, rsi_feature = feature_rsi, company_list = company_list, savefig=True)

#%%

macd_data = get_macd(data)
visualize_macd(macd_data)

#%%

feature_bollinger = "Open"

bollinger_data = bollinger_bands(data, feature = feature_bollinger, window = 20, num_std = 2)
visualize_bollinger_bands(bollinger_data, feature = feature_bollinger)

#%%
#Calculating the RMSE
rmse = np.sqrt(np.mean((Y_test - Y_pred)**2))
print(f"RMSE: {rmse}")

#%%
#MAPE is found in percantage.
mape = np.mean(np.abs((Y_test - Y_pred) / Y_test)) * 100
print(f"MAPE: {mape}%")

#%%
#Calculating the r2_score (employed r2_score library for my convinience)
from sklearn.metrics import r2_score
r2 = r2_score(Y_test, Y_pred)
print(f"R-Squared: {r2}")
