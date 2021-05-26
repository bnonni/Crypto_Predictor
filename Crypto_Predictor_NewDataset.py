#!/usr/bin/env python3
# coding: utf-8

from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from connect import updateCSV
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gc
import os
import sys
from sys import argv
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
gc.collect()

if(len(argv) > 1):
    if(argv[1]  == 'y'):
        updateCSV()
        sys.exit(0)
else:
    pass

BTC_df = pd.read_csv('data.csv')
BTC_df.head()

keys = BTC_df.keys()
for k in keys:
    print(f'{k}: {len(BTC_df[k])}')

xp = BTC_df["Datetime"]
yp = BTC_df["Prices"]
plt.plot(xp, yp)
plt.show()

BTC_df['High'] = BTC_df.High.astype('float64')
BTC_df['Low'] = BTC_df.Low.astype('float64')
BTC_df['Volumes'] = BTC_df.Volumes.astype('float64')

ADL_avg = BTC_df.ADL.mean()
RSI_avg = BTC_df.RSI.mean()
ADL_slp_avg = BTC_df.ADL_slope.mean()
OBV_slp_avg = BTC_df.OBV_slope.mean()

values = {'OBV_slope': OBV_slp_avg, 'RSI': RSI_avg }
BTC_df = BTC_df.fillna(value=values)

BTC_df.head()
BTC_df.info()

np.random.seed(1)

try:
    analysis_name = f'\n[High, Low, RSI] vs. Prices Analysis\n'
    print(analysis_name)
    cwd = os.getcwd()
    if os.path.exists(f'{cwd}/results/NewData_OriginalModel/HighLowRSI_vs_Prices'):
        pass
    else:
        os.mkdir(f'{cwd}/results/NewData_OriginalModel/HighLowRSI_vs_Prices')

    f = open('results/NewData_OriginalModel/HighLowRSI_vs_Prices/Results.txt', 'a')
    f.write(analysis_name)
    # 'Unnamed: 0', 'Datetime','Prices','Volumes', 'ADL', 'OBV', 'ADL_slope', 'OBV_slope'
    X = BTC_df[['High','RSI']].values
    y = BTC_df['Prices'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

    scalar = MinMaxScaler(feature_range=(0, 1))
    X_train = scalar.fit_transform(X_train)
    X_test = scalar.fit_transform(X_test)

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    model = Sequential()

    model.add(LSTM(150, input_shape=(X_train.shape[1], 1), return_sequences=True))

    model.add(LSTM(150, return_sequences=True, recurrent_dropout=0.1))
    model.add(Dropout(0.2))

    model.add(LSTM(150, return_sequences=True, recurrent_dropout=0.1))
    model.add(Dropout(0.2))

    model.add(LSTM(150, return_sequences=True, recurrent_dropout=0.1))
    model.add(Dropout(0.2))

    model.add(LSTM(150, return_sequences=True, recurrent_dropout=0.1))
    model.add(Dropout(0.2))

    model.add(LSTM(150, return_sequences=True))
    model.add(Dropout(0.5))

    model.add(LSTM(150))

    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse', 'mape'])
    summary = model.summary()
    print(summary)
    f.write(str(summary))
    f.write('\n')

    model.fit(X_train, y_train, batch_size=50,\
              validation_split=0.2, epochs=200, shuffle=False)

    model_eval = model.evaluate(X_test, y_test, verbose=2)
    print(model_eval)
    f.write('\nModel Evaluation [X_test, y_test]')
    f.write(str(model_eval))
    f.write('\n')

    y_pred = model.predict(X_test, verbose=2)
    f.write('\nModel Predictions [X_test => y_pred]')
    # f.write(str(y_pred))
    for i in y_pred:
        f.write(str(i))
    f.write('\n')

    model_eval_pred = model.evaluate(y_test, y_pred, verbose=2)
    print(model_eval_pred)
    f.write('\nModel Predictions Evaluation [X_test, y_pred]')
    f.write(str(model_eval_pred))
    f.write('\n')

except Exception as e:
    err = open('Errors.txt', 'a')
    err.write(str(e))
    err.write('\n')

# DONE

# print('\n[High, Low] Analysis\n')
# print('\n[High, RSI] Analysis\n')
# print('\n[High, ADL] Analysis\n')
# print('\n[High, ADL_slope] Analysis\n')
# print('\n[High, OBV] Analysis\n')
# print('\n[High, OBV_slope] Analysis\n')

# print('\n[Volumes] Analysis\n')
# print('\n[High] Analysis\n')
# print('\n[Low] Analysis\n')
# print('\n[RSI] Analysis\n')
# print('\n[ADL] Analysis\n')
# print('\n[OBV] Analysis\n')
# print('\n[ADL_slope] Analysis\n')
# print('\n[OBV_slope] Analysis\n')

# print('\n[Volumes, High] Analysis\n')
# print('\n[Volumes, Low] Analysis\n')
# print('\n[Volumes, RSI] Analysis\n')
# print('\n[Volumes, ADL] Analysis\n')
# print('\n[Volumes, ADL_slope] Analysis\n')
# print('\n[Volumes, OBV] Analysis\n')
# print('\n[Volumes, OBV_slope] Analysis\n')
