from flask import Flask, send_file, render_template
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# # from Historic_Crypto import HistoricalData
# # import pandas as pd
# # import time

# # new =  HistoricalData("BTC-USD", 60, "2021-01-01-00-00")

# # df = pd.DataFrame(new)
import warnings
warnings.filterwarnings('ignore')
import os
import pandas as pd
import numpy as np
import math
import datetime as dt 
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score 
from sklearn.metrics import mean_poisson_deviance, mean_gamma_deviance, accuracy_score
from sklearn.preprocessing import MinMaxScaler

from itertools import product
import statsmodels.api as sm

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LSTM

from itertools import cycle
import plotly.offline as py
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

plt.style.use('seaborn-darkgrid')

root_path = "/major project/btc_23_feb.csv"
btc_input_df = pd.read_csv(root_path)
btc_input_df_datetype = btc_input_df.astype({'time': 'datetime64'})

btc_input_df_datetype['date'] = pd.to_datetime(btc_input_df_datetype['time'],unit='s').dt.date

group = btc_input_df_datetype.groupby('date')

btc_closing_price_groupby_date = group['close'].mean()
# Train Test Split
prediction_days = 60

# Set Train data to be uplo ( Total data length - prediction_days )
df_train= btc_closing_price_groupby_date[:len(btc_closing_price_groupby_date)-prediction_days].values.reshape(-1,1)


# Set Test data to be the last prediction_days (or 60 days in this case)
df_test= btc_closing_price_groupby_date[len(btc_closing_price_groupby_date)-prediction_days:].values.reshape(-1,1)

chosen_col = 'Close'

scaler_train = MinMaxScaler(feature_range=(0, 1))
scaled_train = scaler_train.fit_transform(df_train)

scaler_test = MinMaxScaler(feature_range=(0, 1))
scaled_test = scaler_test.fit_transform(df_test)
def dataset_generator_lstm(dataset, look_back=5):
    # A “lookback period” defines the window-size of how many
    # previous timesteps are used in order to predict
    # the subsequent timestep. 
    dataX, dataY = [], []
    
    for i in range(len(dataset) - look_back):
        window_size_x = dataset[i:(i + look_back), 0]
        dataX.append(window_size_x)
        dataY.append(dataset[i + look_back, 0]) # this is the label or actual y-value
    return np.array(dataX), np.array(dataY)

trainX, trainY = dataset_generator_lstm(scaled_train)

testX, testY = dataset_generator_lstm(scaled_test)

### And now reshape trainX and testX 
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))

testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1 ))

# checking the values for input_shape = (trainX.shape[1], trainX.shape[2])
# Note - `input_shape` of LSTM Model - `input_shape` is supposed to be (timesteps, n_features).

regressor = Sequential()

regressor.add(LSTM(units = 128, activation = 'relu',return_sequences=True, input_shape = (trainX.shape[1], trainX.shape[2])))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 64, input_shape = (trainX.shape[1], trainX.shape[2])))
regressor.add(Dropout(0.2))

regressor.add(Dense(units = 1))

regressor.summary()

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Compiling the LSTM
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

checkpoint_path = 'my_best_model.hdf5'

checkpoint = ModelCheckpoint(filepath=checkpoint_path, 
                             monitor='val_loss',
                             verbose=1, 
                             save_best_only=True,
                             mode='min')


earlystopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

callbacks = [checkpoint, earlystopping]
# callbacks = [checkpoint]


history = regressor.fit(trainX, trainY, batch_size = 32, epochs = 300, verbose=1, shuffle=False, validation_data=(testX, testY), callbacks=callbacks)
from tensorflow.keras.models import load_model

model_from_saved_checkpoint = load_model(checkpoint_path)

# Transformation to original form and making the predictions

# predicted_btc_price_test_data = regressor.predict(testX)
predicted_btc_price_test_data = model_from_saved_checkpoint.predict(testX)

predicted_btc_price_test_data = scaler_test.inverse_transform(predicted_btc_price_test_data.reshape(-1, 1))

test_actual = scaler_test.inverse_transform(testY.reshape(-1, 1))

# Transformation to original form and making the predictions

predicted_btc_price_train_data = model_from_saved_checkpoint.predict(trainX)

predicted_btc_price_train_data = scaler_train.inverse_transform(predicted_btc_price_train_data.reshape(-1, 1))

train_actual = scaler_train.inverse_transform(trainY.reshape(-1, 1))

rmse_lstm_test = math.sqrt(mean_squared_error(test_actual, predicted_btc_price_test_data))


# With 2 Layers + Dropout + lookback=5 => I got - Test RMSE: 1666.162  => This seems best

rmse_lstm_train = math.sqrt(mean_squared_error(train_actual, predicted_btc_price_train_data))


# With 2 Layers + Dropout + lookback=5 => I got - Test RMSE: 1047.916  => This seems best
lookback_period = 5

# That is the original Trading data ended on 30-Oct-2021, but going to forecast for Future 5 days beyond 30-Oct-2021

testX_last_5_days = testX[testX.shape[0] - lookback_period :  ]

predicted_5_days_forecast_price_test_x = []

for i in range(5):  
  predicted_forecast_price_test_x = model_from_saved_checkpoint.predict(testX_last_5_days[i:i+1])
  
  predicted_forecast_price_test_x = scaler_test.inverse_transform(predicted_forecast_price_test_x.reshape(-1, 1))
  # print(predicted_forecast_price_test_x)
  predicted_5_days_forecast_price_test_x.append(predicted_forecast_price_test_x)
  
# That is the original Trading data ended on 30-Oct-2021, but now I am going to forecast beyond 30-Oct-2021
predicted_5_days_forecast_price_test_x = np.array(predicted_5_days_forecast_price_test_x)

predicted_5_days_forecast_price_test_x = predicted_5_days_forecast_price_test_x.flatten()

predicted_btc_price_test_data = predicted_btc_price_test_data.flatten()

predicted_btc_test_concatenated = np.concatenate((predicted_btc_price_test_data, predicted_5_days_forecast_price_test_x))

# plt.figure(figsize=(16,7))

# plt.plot(predicted_btc_test_concatenated, 'r', marker='.', label='Predicted Test')

# plt.plot(test_actual, marker='.', label='Actual Test')

# plt.legend()

# plt.show()

import praw
import pandas as pd
import datetime
from praw.models import MoreComments

reddit = praw.Reddit(client_id='xYZQPkBX6Yn9_5Lo_votVA',
                     client_secret='gFrYmMgncbcBIf9vnyqX_gxewZ5VlQ',
                     user_agent='major_project',
                     username='rashidraxis',
                     password='#Rashid369')

subreddit = reddit.subreddit('bitcoin')

reddit_data = {
    "date": [],
    "title":[],
    "text_inside":[]
}

for submission in subreddit.new(limit=1000):
    reddit_data["date"] += [datetime.datetime.utcfromtimestamp(submission.created).strftime('%Y-%m-%d')]
    reddit_data["title"] += [submission.title]
    reddit_data["text_inside"] += [submission.selftext]
df = pd.DataFrame(reddit_data)
df.loc[2, "text_inside"]
clean_df = df[df["text_inside"] != ""]
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
clean_df['compound'] = [analyzer.polarity_scores(x)['compound'] for x in clean_df['text_inside']]
clean_df["date"] =  pd.to_datetime(clean_df["date"])

result_df = clean_df.groupby("date")["compound"].mean()

result_list = result_df.tolist()

# plt.plot(result_list)
# plt.show()


import seaborn as sns



app = Flask(__name__)

@app.route('/')
def home():
   return render_template("index.html")

@app.route('/visualize_btc')
def visualize_btc():
  fig, ax = plt.subplots(figsize=(8,6))
  ax = sns.set_style(style="darkgrid")
  sns.lineplot(predicted_btc_test_concatenated, color="r")
  sns.lineplot(test_actual)
  canvas = FigureCanvas(fig)
  img = io.BytesIO()
  fig.savefig(img)
  img.seek(0)
  return send_file(img, mimetype='img/png')

@app.route('/visualize_reddit')
def visualize_reddit():
  fig, ax = plt.subplots(figsize=(8,6))
  ax = sns.set_style(style="darkgrid")
  sns.lineplot(result_list)
  canvas = FigureCanvas(fig)
  img = io.BytesIO()
  fig.savefig(img)
  img.seek(0)
  return send_file(img, mimetype='img/png')

if __name__ == "__main__":
   app.run()