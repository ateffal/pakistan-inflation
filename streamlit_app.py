import streamlit as st
import pandas as pd
import altair as alt
# from PIL import Image

import helpers as h
import compare_models as cm


prices_terror_attacks = pd.read_excel(
    'data/wfp_food_prices_pakistan_restructured.xlsx', sheet_name='wfp_food_prices_cmname')
prices_terror_attacks = prices_terror_attacks.dropna()
# prices_terror_attacks

crimes = pd.read_excel(
    'data/year_wise_crime_report_v20221114.xlsx', sheet_name='monthly_calculated')

data = prices_terror_attacks.set_index('date').join(
    crimes[['date', 'monthly_crimes_calculated_with_noise']].set_index('date'), how='left').reset_index()
data = data.sort_values(['cmname_mktname', 'date'])

# prediction_nnet(cmname, mktname, features, target, n_obs = None, lags = 6)
cmnames = list(data.cmname.unique())
mktnames = list(data.mktname.unique())

cols = ['price', 'price_var', 'crimes_annual_percent_change', 'crimes_per_100K_population',
        'terror_attacks_casualties', 'monthly_crimes_calculated_with_noise']

crimes_cols = ['terror_attacks_casualties',
               'crimes_per_100K_population',
               'crimes_annual_percent_change',
               'monthly_crimes_calculated_with_noise']


# image = Image.open('pakistan-flag.JPG')
# st.image(image, use_column_width=True)

st.write(
    """
    # Dashboard Building For Growing Inflation in Pakistan
    ***
    """


)


options = st.multiselect(
    "Select variables to use for predicting crimes : ",
    cols, cols[0])

# for k in options:
#     st.write('You selected:', options[k])

var_target = st.selectbox(
    "Select target varibale for crimes levels : ",
    crimes_cols)

st.sidebar.header('Choose a comodity')

comodity = st.sidebar.selectbox(
    'Choose a comodity', cmnames)


st.sidebar.header('Choose a market')

market = st.sidebar.selectbox(
    'Choose a market', mktnames)

choosen_model = st.sidebar.multiselect(
    'Compare models :', ['NNET', 'RNN', 'LSTM'], ['NNET', 'RNN'])


# selected_data = h.select_data(data, comodity, market)

sugar_quetta_data = h.select_data(data, comodity, market)

# ['price', 'monthly_crimes_calculated_with_noise']
if not (var_target in options):
    vars_features = options + [var_target]  # ['price'] + [var_target]
else:
    vars_features = options
# var_target = 'crimes_per_100K_population'


selected_data, predictions, RMSE = h.prediction_nnet_dense_layers(sugar_quetta_data,
                                                                  features=vars_features,
                                                                  target=var_target,
                                                                  lags=6,
                                                                  scale_data=True)
st.write(
    """
    # Predictions
    ***
    """ + var_target
)


st.line_chart(predictions[[predictions.columns[0], predictions.columns[1]]])

st.dataframe(predictions)


st.write(
    """
    # Selected data : 
    ***
    """
)

st.write('Comodity : ' + comodity, ' --- ' + market +
         '--- ' + str(len(selected_data)) + ' records')

st.dataframe(selected_data, 10000, 500)


st.write(
    """
    # Compare models by splitting data into train, validation and test : 
    ***
    """
)

predictions_2, rmses = cm.compare_models(sugar_quetta_data, 
                                                vars_features, var_target, choosen_model)

st.line_chart(predictions_2)

for m in choosen_model:
    st.text('Root Mean Squared Error of ' + m + ' : ' + str(rmses[m]))

st.dataframe(predictions_2)
