import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, robust_scale
from sklearn.utils import shuffle
from functools import partial
from datetime import datetime, timedelta
import errpred as erp

def fit_transform_feature(scaler, a):
    b = a.reshape(-1,a.shape[-1])
    b = scaler.fit_transform(b)
    return b.reshape(a.shape)

def inverse_transform_feature(scaler, a):
    b = a.reshape(-1,a.shape[-1])
    b = scaler.inverse_transform(b)
    return b.reshape(a.shape)

def load_stock_timeseries_dataset(file="AAPL.csv", validation_fraction=.15, train_validation_fraction=.1, shuffle_training_set = True):    
    def deviation(window):
        return window[-1] / np.average(window)
    
    df = pd.read_csv(file)
    df["Close"] = df["Adj Close"]
    df["Volume"] = df["Volume"].astype(float)
    df.set_index("Date", inplace=True)
    df = df.drop(['Adj Close'], axis=1)
            
    df["50avg_deviation"] = df["Close"].rolling(window=50).apply(deviation)
    df["100avg_deviation"] = df["Close"].rolling(window=100).apply(deviation)
    df["200avg_deviation"] = df["Close"].rolling(window=200).apply(deviation)
    df["200avg"] = df["Close"].rolling(window=200).mean()
    df = df.ix[:,["50avg_deviation", "100avg_deviation", "200avg_deviation", "200avg","Close"]].dropna()    
    data = df.values

    def unscaler(yscaler, z, prediction):
        # prediction here means offset from 200d moving average
        return inverse_transform_feature(yscaler, prediction.reshape(-1,1)) * z.reshape(-1,1)
    
    def x_sampler(window):
        """ Everything up to, but not including is the input to the prediction """
        window = window[:-1].copy()
        window[:,4] = (window[:,4] - window[:,3]) / window[:,3]
        return window[:,[0,1,2,4]]

    def y_sampler(window):
        """ The last value in the window, is the day to predict """
        # Predict deviation from yesterdays close
        return window[-1,[4]] / window[-2,[4]]

    def z_sampler(window):
        return window[-2,4]

    def numpy_window_func(window_size, window_step, array):
        for i in range(0, len(array) - window_size, window_step):
            yield array[i:i+window_size]  

    # x/y - input/output data for model to learn from
    # z - close at sample time (actual prediction is then y+z)
    day_ahead_sampler = partial(numpy_window_func, 7, 1) # 6 days lookback +  1 day ahead prediction = window size 7
    x = np.array(list(map(x_sampler, day_ahead_sampler(data))), dtype=np.float32)
    y = np.array(list(map(y_sampler, day_ahead_sampler(data))), dtype=np.float32).reshape(-1,1)
    z = np.array(list(map(z_sampler, day_ahead_sampler(data))), dtype=np.float32).reshape(-1,1)
      
    # scale the dataset
    x_scaler = RobustScaler()
    x = fit_transform_feature(x_scaler, x)
    y_scaler = RobustScaler()
    y = fit_transform_feature(y_scaler, y)

    # import matplotlib.pyplot as plt
    # plt.plot(y)
    # plt.show()
    # exit(1)
    
    # Split the dataset into training and validation set. Then, the training set is split again for fit-validation
    x_train, x_test = erp.split_train_test(x, validation_fraction)
    y_train, y_test = erp.split_train_test(y, validation_fraction)
    _, z_test = erp.split_train_test(z, validation_fraction)
    
    # Shuffle and extract validation data
    if shuffle_training_set: x_train, y_train = shuffle(x_train, y_train)
    x_train, x_train_validation = erp.split_train_test(x_train, train_validation_fraction)
    y_train, y_train_validation = erp.split_train_test(y_train, train_validation_fraction)
        
    return erp.normalize_data(x_train, y_train, x_train_validation, y_train_validation, x_test, y_test, z_test) + [partial(unscaler, y_scaler)]
