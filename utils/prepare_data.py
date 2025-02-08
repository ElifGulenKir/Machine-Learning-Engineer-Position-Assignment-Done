import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler


def prepare_train_test_data(data, time_steps_for_prediction: int = 90, train_test_probability: float = 0.8, scale_data: bool = True):
    if (train_test_probability >= 0.999):
        if (train_test_probability >= 1):
            return "train_test_prabability must be between 0-1"
        return f"train_test_prabability parameter must be less than 99%, this is because there should be enough test data to validate. Given: {train_test_probability}"
    if (time_steps_for_prediction >= data.shape[0]):
        return f"time_steps_for_prediction parameter must be smaller than the total time steps in the data. You can either decrease the time_steps_for_prediction parameter or increase the data size by changing the start and end dates of the corresponding company. Given time_steps_for_prediction is: {time_steps_for_prediction}. Data time step size is: {data.shape[0]}"
    
    if (scale_data):
        # Normilizing data. This step is not a mandatory, however, scaling the data will help us in the training process. This leverages our training process by preventing any kind of vanishing and exploiding gradients.
        scaler = MinMaxScaler(feature_range=(0, 1))
        data = scaler.fit_transform(np.array(data))
        
    X, Y = _get_time_sequences(data = data, time_steps_for_prediction = time_steps_for_prediction)
    
    split_index = int(train_test_probability * X.shape[0])
    X_train, X_test = X[:split_index], X[split_index:]
    Y_train, Y_test = Y[:split_index], Y[split_index:]
    #X_train has shape train_test_probability*X.shape[0] x time_steps_for_prediction x # of company*features
    #X_test has shape (1 - train_test_probability)*X.shape[0] x time_steps_for_prediction x # of company*features
    #Y_train has shape train_test_probability*X.shape[0] x # of company*features
    #Y_test has shape (1 - train_test_probability) * X.shape[0] x # of company*features
    if (scale_data):
        return X_train, X_test, Y_train, Y_test, scaler
    else:     
        return X_train, X_test, Y_train, Y_test

def _get_time_sequences(data, time_steps_for_prediction):
    X, Y = [], []
    for i in range(len(data) - time_steps_for_prediction):
        X.append(data[i:i + time_steps_for_prediction])
        Y.append(data[i + time_steps_for_prediction])
    return np.array(X), np.array(Y)


def get_input_output_dims(X_train, Y_train, company_list):
    return (X_train.shape[1], X_train.shape[2]), Y_train.shape[1]
