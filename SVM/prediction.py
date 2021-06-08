#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 09:14:43 2021

@author: nronzoni
"""

import pandas as pd 
import scipy 
import sklearn
import tslearn 
import numpy as np
import random
from toolz.itertoolz import sliding_window, partition
from tslearn.utils import to_time_series, to_time_series_dataset
##### strategy for normalization 
from sklearn.preprocessing import MinMaxScaler, StandardScaler



#define a function to create the daily time series and the scaling 
def daily_series(data,n):
    #normalization of the data 
    data=np.array(data)
    data=data.reshape((len(data), 1))
    #fit data 
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler_series = scaler.fit(data)
    #scale train data 
    normalized_series=scaler_series.transform(data)
    #from array to list 
    normalized_series=normalized_series.tolist()
    len(normalized_series)
    #create daily time series 
    time_series=list(partition(n,normalized_series))
    #from list to multidimensional array 
    time_series=np.asarray(time_series)
    #create univariate series for normalized observations 
    daily_time_series = to_time_series(time_series)
    return daily_time_series, scaler_series

df= pd.read_excel(r"magnan 2019.xlsx") 
df

occupancy_train=df.loc[:,'occupancy']

occupancy_series_train=daily_series(occupancy_train,190)

occupancy_series_train[0].shape

speed_train=df.loc[:,'speed']

speed_series_train=daily_series(speed_train,190)

speed_series_train[0].shape

multivariate=np.dstack((occupancy_series_train[0],speed_series_train[0]))
multivariate_time_series_train = to_time_series(multivariate)
print(multivariate_time_series_train.shape)

df_test= pd.read_excel(r"gloria 2019.xlsx") 
df_test

occupancy_test=df_test.loc[:,'occupancy']

occupancy_series_test=daily_series(occupancy_test,190)

occupancy_series_test[0].shape

speed_test=df_test.loc[:,'speed']

speed_series_test=daily_series(speed_test,190)


multivariate_test=np.dstack((occupancy_series_test[0],speed_series_test[0]))
multivariate_time_series_test = to_time_series(multivariate_test)
print(multivariate_time_series_test.shape)

multivariate_time_series_test.shape

from tslearn.svm import TimeSeriesSVR
from sklearn.metrics import mean_squared_error
from statistics import mean
from tslearn.generators import random_walk_blobs

C=[1,0.01,0.1,10]

epsilon=[0.1,0.01,1,10]

# list of all possible time 
hours=range(20,190)
# randomly choose 50 point from this list 
time=(random.sample(hours,50))
len(time)
# grid search for C, epsilon, window size
# for every time of the day return the suggested values of C and epsilon
def grid_search(data,C, epsilon, window_size):
    df_final=pd.DataFrame(data=None,columns = ['time','C', 'epsilon','mse'])
    for i in range(0,len(time)):
        X_train=data[:,time[i]-window_size:time[i],:]
        # from[ts,n,d] to [ts,] one step ahead,  select the d (detector you want to predict)
        PRED_train=[]
        GROUND_TRUTH=[]
        y_train=data[:,time[i],1]
        ground_truth=speed_series_train[1].inverse_transform(y_train.reshape(1, -1))
        GROUND_TRUTH.append(ground_truth)
        df=pd.DataFrame(data=None,columns = ['time','C', 'epsilon','mse'])
        for j in range(0,len(C)):
            for z in range(0,len(epsilon)):
               reg = TimeSeriesSVR(C=C[j],kernel="gak", gamma="auto",epsilon=epsilon[z])
               y_hat=reg.fit(X_train, y_train).predict(X_train)
               prediction_train=speed_series_train[1].inverse_transform(y_hat.reshape(1, -1))
               PRED_train.append(prediction_train)
               mse=mean_squared_error(ground_truth,prediction_train)
               current=[time[i],C[j],epsilon[z],mse]
               df.loc[len(df)]=current
               df.drop_duplicates(subset = ['mse'], keep = 'first', inplace = True) 
        min_time=df[df.mse == df.mse.min()]
        df_final = df_final.append(min_time)
    return print('C: ', df_final['C'].mode(),'epsilon: ',df_final['epsilon'].mode())

grid_search(multivariate_time_series_train[210:220,:,:], C, epsilon, 20)


#starting time >= window_size
def walk_forward_validation(train,test,window_size,starting_time):
    #define a starting split X and Y for both train and test
    X_train=train[:,0:starting_time,:]
    Y_train=train[:,starting_time:,:]
    X_test=test[:,0:starting_time,:]
    Y_test=test[:,starting_time:,:]
    #initialize the lists
    MSE_train=[]
    MSE_test=[]
    PRED_train=[]
    PRED_test=[]
    GROUND_TRUTH_train=[]
    GROUND_TRUTH_test=[]
    for t in range(0,len(Y_train[1])): # lenghts of Y_train[1] = lenghts of Y_test[1]
        #select most recent observations of X for both train and test 
        x_train=X_train[:,-window_size:,:]
        x_test=X_test[:,-window_size:,:]
        #select the target Train 
        y_train=Y_train[:,t,1]
        #rescale the target Train 
        ground_truth_train=speed_series_train[1].inverse_transform(y_train.reshape(1, -1))
        GROUND_TRUTH_train.append(ground_truth_train)
        #select the target Test
        y_test=Y_test[:,t,1]
        #rescale the target Test 
        ground_truth_test=speed_series_test[1].inverse_transform(y_test.reshape(1, -1))
        GROUND_TRUTH_test.append(ground_truth_test)
        #model 
        reg = TimeSeriesSVR(kernel="gak", gamma="auto")
        #fit the model and predict the Train 
        y_hat_train=reg.fit(x_train, y_train).predict(x_train)
        #rescale the prediction of the Train 
        prediction_train=speed_series_train[1].inverse_transform(y_hat_train.reshape(1, -1))
        PRED_train.append(prediction_train)
        # predict the test 
        y_hat_test=reg.predict(x_test)
        #rescale the prediction of the Test 
        prediction_test=speed_series_test[1].inverse_transform(y_hat_test.reshape(1, -1))
        PRED_test.append(prediction_test)
        #compute the mean square error 
        mse_train=mean_squared_error(ground_truth_train, prediction_train)
        mse_test=mean_squared_error(ground_truth_test, prediction_test)
        MSE_train.append(mse_train)
        MSE_test.append(mse_test)
        #Add the curent observation to the train set 
        obs_train=Y_train[:,t:t+1,:]
        obs_test=Y_test[:,t:t+1,:]
        X_train=np.hstack((X_train,obs_train))
        X_test=np.hstack((X_test,obs_test))
        print(mse_train,mse_test)
    return mean(MSE_train), mean(MSE_test), PRED_train, PRED_test, GROUND_TRUTH_train, GROUND_TRUTH_test
        
prova=walk_forward_validation(multivariate_time_series_train[254:261,:,:],multivariate_time_series_test[261:262,:,:],7,20)      

import matplotlib.pyplot as plt

# plot forecasts against actual outcomes
x= np.arange(7,24,0.1)
len(x)
plt.plot(x,np.concatenate(prova[3], axis=0 ),color='blue', label="prediction test") 
plt.plot(x,np.concatenate(prova[5], axis=0 ), color='red', label="ground truth") 
plt.legend()
plt.show()



