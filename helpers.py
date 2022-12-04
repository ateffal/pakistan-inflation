import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import mean_squared_error
import numpy as np

def unscale_data(df_, mins, maxs):
    df = df_.copy()
    for c in df.columns:
        df[c] = df[c]*(maxs - mins)
        df[c] = df[c] + mins
    return df

def scale_data_func(df_):
    return (df_ - df_.min())/(df_.max()-df_.min())
    


def select_data(data, cmname, mktname, n_obs = None):
    product_market = cmname + '_' + mktname
    
    # select data based on cmname and mktname
    cols = ['date', 'price', 'price_var', 'crimes_annual_percent_change', 'crimes_per_100K_population',
            'terror_attacks_casualties', 'monthly_crimes_calculated_with_noise']
    data_cmname_mktname = data.loc[data['cmname_mktname'] == product_market, :][cols].set_index('date')
    data_cmname_mktname = data_cmname_mktname.dropna()
    
    # use only the first n_obs rows
    if not n_obs is None :
        data_cmname_mktname = data_cmname_mktname.reset_index().iloc[0:n_obs,:]
        # reset date as index
        data_cmname_mktname = data_cmname_mktname.set_index('date')
        
    return data_cmname_mktname




    return data_cmname_mktname, mins, maxs

################################################################################################

def prediction_nnet_dense_layers(data_cmname_mktname_, features, target, lags = 6, nodes_layers=[32, 32, 32], scale_data = True):
    """
    make prediction for given commodity name and market name

    Args:
      cmname (string) - commodity name. Ex : Wheat flour - Retail
      mktname (string) - market name. Ex : Karachi
      features (list of string) - variables to be used for prediction - in fact, lag of these variables 
      target (string) - variable that we want to make predictions of

    Returns:
      data, predictions - 
    """
    data_cmname_mktname = data_cmname_mktname_.copy()
    
    # number of features
    n_features = len(features)
    
    # construct features and labels series
    df_features_unscaled = make_lags(data_cmname_mktname[features], lags).dropna()
    df_target_unscaled = data_cmname_mktname.loc[:,target].iloc[lags:]
    
    #scale data
    if scale_data:
        df_features = (df_features_unscaled - df_features_unscaled.min())/(df_features_unscaled.max()-df_features_unscaled.min())
        df_target = (df_target_unscaled - df_target_unscaled.min())/(df_target_unscaled.max()-df_target_unscaled.min())
    else:
        df_features = df_features_unscaled.copy()
        df_target = df_target_unscaled.cop()
        
    # separate features and target values
    data_cmname_mktname_features = df_features.values
    data_cmname_mktname_labels = df_target.values
    
    # create the model from nodes_layers list
    #input layer
    model = tf.keras.models.Sequential([tf.keras.layers.Dense(nodes_layers[0], input_shape=(None, 1, n_features * lags))])
    # hidden layers
    if len(nodes_layers) > 1:
        for l in nodes_layers[1:]:
            model.add(tf.keras.layers.Dense(32))
    # output layer
    model.add(tf.keras.layers.Dense(1, activation="relu"))
    
    
    # Set the training parameters
    model.compile(loss="mae", optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))
    
    print('shape of x : ', data_cmname_mktname_features.shape)
    print('shape of y : ', data_cmname_mktname_labels.shape)
    
    # Train the model
    model.fit(x = data_cmname_mktname_features,y = data_cmname_mktname_labels, epochs=100, verbose=0)
    
    # get predictions
    forecasts = model.predict(data_cmname_mktname_features)
    
    # get unscaled predictions
    forecasts = forecasts * (df_target_unscaled.max() - df_target_unscaled.min())
    forecasts = forecasts + df_target_unscaled.min()
    
    # Plot the predictions
    col_name_observed = 'observed ' + target
    col_name_predicted = 'predicted ' + target
    
    time = range(len(forecasts))
    plot_series(time, (df_target_unscaled.values, forecasts), (col_name_observed , col_name_predicted))
    
    forecasts_val = pd.DataFrame({col_name_observed : list(df_target_unscaled.values) , col_name_predicted : list(forecasts[:,0])})
    forecasts_val['diff'] = abs(forecasts_val[col_name_observed]-forecasts_val[col_name_predicted])
    forecasts_val['diff (%)'] = forecasts_val['diff']/forecasts_val[col_name_observed]
#     print('used data : ')
    used_df = pd.concat([df_features_unscaled, df_target_unscaled], axis=1)

    print(forecasts_val[['diff', 'diff (%)']].mean())
    rmse = mean_squared_error(forecasts_val[col_name_observed], forecasts_val[col_name_predicted], squared=False)
    print('Root Mean Squared Error : ', rmse)
    
    
    return used_df, forecasts_val, rmse

################################################################################################


# Credit https://colab.research.google.com/github/https-deeplearning-ai/tensorflow-1-public/blob/main/C4/W2/ungraded_labs/C4_W2_Lab_2_single_layer_NN.ipynb#scrollTo=Yw5jEYuBADvA
def plot_series(time, series, series_names=None, format="-", start=0, end=None):
    """
    Visualizes time series data

    Args:
      time (array of int) - contains the time steps
      series (array of int) - contains the measurements for each time step
      series_names (str or list of str) - names of series to be ploted
      format - line style when plotting the graph
      label - tag for the line
      start - first time step to plot
      end - last time step to plot
    """

    # Setup dimensions of the graph figure
    plt.figure(figsize=(15, 8))

    if type(series) is tuple:

        for series_num in series:
            # Plot the time series data
            plt.plot(time[start:end], series_num[start:end], format)
            # legend
            if series_names != None:
                plt.legend([s for s in series_names])
            else:
                plt.legend(['serie_' + str(i) for i in len(series_names)])

    else:
        # Plot the time series data
        plt.plot(time[start:end], series[start:end], format)
        if series_names != None:
            plt.legend(series_names)
        else:
            plt.legend('serie_1')

    # Label the x-axis
    plt.xlabel("Time")

    # Label the y-axis
    plt.ylabel("Value")

    # Overlay a grid on the graph
    plt.grid(True)

    # Draw the graph on screen
    plt.show()

# Credit : https://www.kaggle.com/code/amineteffal/exercise-forecasting-with-machine-learning/edit


def make_lags(ts, lags, lead_time=1):
    return pd.concat(
        {
            f'y_lag_{i}': ts.shift(i)
            for i in range(lead_time, lags + lead_time)
        },
        axis=1)


def make_multistep_target(ts, steps):
    return pd.concat(
        {f'y_step_{i + 1}': ts.shift(-i)
         for i in range(steps)},
        axis=1)
#############################################################################################################

def predict_rnn(data_cmname_mktname_, features, target, lags = 6, nodes_layers=[32, 32, 32], scale_data = True):

    n = len(data_cmname_mktname_)
    n_train = int(n * 0.6)
    n_validation = int(n * 0.2)
    n_test = n - n_train - n_validation
    train_data = data_cmname_mktname_[0:n_train]
    validation_data = data_cmname_mktname_[n_train : (n_train + n_validation)]
    test_data = data_cmname_mktname_[(n_train + n_validation):]

    lags = 6
    w2 = WindowGenerator(input_width=lags, label_width=1, shift=1,train_df = train_data, val_df=validation_data, 
                     test_df = test_data, label_columns=[target])

    # Set the learning rate
    learning_rate = 1e-3

    # set epochs
    epochs = 300

    # input shape
    input_shape_ = (lags, train_data.shape[1])

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

    pred_rnn = np.concatenate((rnn_model.predict(w2.train)[:,0], 
                               rnn_model.predict(w2.test)[:,0], rnn_model.predict(w2.val)[:,0]))
    w2_test = list(w2.test.as_numpy_iterator())[0][1][:,0,0]

    # pred_df = pd.DataFrame({'obs' : w2_test[0][1][:,0,0], 'pred nnet' : pred_nnet, 'pred rnn' : pred_rnn, 'pred lstm' : pred_lstm})

    return pred_rnn



class WindowGenerator():
    def __init__(self, input_width, label_width, shift, train_df, val_df, test_df, label_columns=None):
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
        self.column_indices = {name: i for i, name in enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

def __repr__(self):
    return '\n'.join([
        f'Total window size: {self.total_window_size}',
        f'Input indices: {self.input_indices}',
        f'Label indices: {self.label_indices}',
        f'Label column name(s): {self.label_columns}'])

def split_window(self, features):
  inputs = features[:, self.input_slice, :]
  labels = features[:, self.labels_slice, :]
  if self.label_columns is not None:
    labels = tf.stack(
        [labels[:, :, self.column_indices[name]] for name in self.label_columns],
        axis=-1)

  # Slicing doesn't preserve static shape information, so set the shapes
  # manually. This way the `tf.data.Datasets` are easier to inspect.
  inputs.set_shape([None, self.input_width, None])
  labels.set_shape([None, self.label_width, None])

  return inputs, labels

WindowGenerator.split_window = split_window


def make_dataset(self, data):
  data = np.array(data, dtype=np.float32)
  ds = tf.keras.utils.timeseries_dataset_from_array(
      data=data,
      targets=None,
      sequence_length=self.total_window_size,
      sequence_stride=1,
      shuffle=False, # True
      batch_size=8,)

  ds = ds.map(self.split_window)

  return ds

WindowGenerator.make_dataset = make_dataset


@property
def train(self):
  return self.make_dataset(self.train_df)

@property
def val(self):
  return self.make_dataset(self.val_df)

@property
def test(self):
  return self.make_dataset(self.test_df)

@property
def example(self):
  """Get and cache an example batch of `inputs, labels` for plotting."""
  result = getattr(self, '_example', None)
  if result is None:
    # No example batch was found, so get one from the `.train` dataset
    result = next(iter(self.train))
    # And cache it for next time
    self._example = result
  return result

WindowGenerator.train = train
WindowGenerator.val = val
WindowGenerator.test = test
WindowGenerator.example = example


################################################ plot method ###########################################################
def plot(self, model=None, plot_col='monthly_crimes_calculated_with_noise', max_subplots=3):
  inputs, labels = self.example
  plt.figure(figsize=(12, 8))
  plot_col_index = self.column_indices[plot_col]
  max_n = min(max_subplots, len(inputs))
  for n in range(max_n):
    plt.subplot(max_n, 1, n+1)
    plt.ylabel(f'{plot_col} [normed]')
    plt.plot(self.input_indices, inputs[n, :, plot_col_index],
             label='Inputs', marker='.', zorder=-10)

    if self.label_columns:
      label_col_index = self.label_columns_indices.get(plot_col, None)
    else:
      label_col_index = plot_col_index

    if label_col_index is None:
      continue

    plt.scatter(self.label_indices, labels[n, :, label_col_index],
                edgecolors='k', label='Labels', c='#2ca02c', s=64)
    if model is not None:
      predictions = model(inputs)
      plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                  marker='X', edgecolors='k', label='Predictions',
                  c='#ff7f0e', s=64)

    if n == 0:
      plt.legend()

  plt.xlabel('Time')

WindowGenerator.plot = plot
