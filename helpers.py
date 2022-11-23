import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import mean_squared_error

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
