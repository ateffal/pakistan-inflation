from helpers import WindowGenerator
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import mean_squared_error
import numpy as np

def compare_models(data_rnn_, vars_to_use_, var_to_predict_, models, lags = 6, learning_rate = 1e-6, epochs = 500):
    data_rnn = data_rnn_.copy()
    data_rnn = data_rnn[vars_to_use_]
    data_rnn = (data_rnn - data_rnn.min())/(data_rnn.max()-data_rnn.min())
    n = len(data_rnn)
    n_train = int(n * 0.6)
    n_validation = int(n * 0.2)
    n_test = n - n_train - n_validation
    train_data = data_rnn[0:n_train]
    validation_data = data_rnn[n_train : (n_train + n_validation)]
    test_data = data_rnn[(n_train + n_validation):]

    w2 = WindowGenerator(input_width=lags, label_width=1, shift=1,train_df = train_data, val_df=validation_data, 
                     test_df = test_data, label_columns=[var_to_predict_])

    # input shape
    input_shape_ = (lags, train_data.shape[1])

    if "NNET" in models:

        # classical nnet
        nnet_model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=128, input_shape=input_shape_),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=1)
        ])

        # Set the training parameters
        nnet_model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
                    metrics=[tf.keras.metrics.MeanAbsoluteError()])

        nnet_model.fit(w2.train, validation_data=w2.val, epochs=epochs, verbose=0)

    if "RNN" in models:
        # Build the RNN Model
        rnn_model = tf.keras.models.Sequential([
        tf.keras.layers.SimpleRNN(128, return_sequences=False, input_shape=input_shape_),
        tf.keras.layers.Dense(1)
        ])

        # Set the optimizer 
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        # Set the training parameters
        rnn_model.compile(loss=tf.keras.losses.MeanSquaredError(),
                    optimizer=optimizer,
                    metrics=[tf.keras.metrics.MeanAbsoluteError()])

        # Train the model
        history = rnn_model.fit(w2.train,validation_data=w2.val, epochs=epochs, verbose=0)

    if "LSTM" in models:
        lstm_model = tf.keras.models.Sequential([
        # Shape [batch, time, features] => [batch, time, lstm_units]
        tf.keras.layers.LSTM(128, return_sequences=False),
        # Shape => [batch, time, features]
        tf.keras.layers.Dense(units=1)
        ])

        lstm_model.build(input_shape=(None, input_shape_[0], input_shape_[1]))

        ################################################################################

        lstm_model.compile(loss=tf.keras.losses.MeanSquaredError(),
                    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                    metrics=[tf.keras.metrics.MeanAbsoluteError()])

        history = lstm_model.fit(w2.train, validation_data=w2.val, epochs=epochs,
                        verbose=0)

    # get labels of test
    w2_test = list(w2.test.as_numpy_iterator())
    w2_test[0][1][:,0,0]

    pred_df = pd.DataFrame({'obs' : w2_test[0][1][:,0,0]})

    RMSE = {}

    if "NNET" in models:
        pred_nnet = nnet_model.predict(w2.test)[:,0]
        pred_df['pred nnet'] = pred_nnet
        RMSE['NNET'] = mean_squared_error(pred_df['obs'], pred_df['pred nnet'], squared=False)
        
    if "RNN" in models:
        pred_rnn = rnn_model.predict(w2.test)[:,0]
        pred_df['pred rnn'] = pred_rnn
        RMSE['RNN'] = mean_squared_error(pred_df['obs'], pred_df['pred rnn'], squared=False)

    if "LSTM" in models:
        pred_lstm = lstm_model.predict(w2.test)[:,0]
        pred_df['pred lstm'] = pred_lstm
        RMSE['LSTM'] = mean_squared_error(pred_df['obs'], pred_df['pred lstm'], squared=False)

    
    
    

    return pred_df, RMSE








    