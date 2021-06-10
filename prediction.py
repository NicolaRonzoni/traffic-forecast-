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

#IMPORT DATA

df= pd.read_excel(r"occupancy 2019-2020.xlsx") 

df
#train 
#2/9 to 22/12 
df_train=df[0:21280]

#test
#1/1 to 1/3 
df_test=df[22990:34580]

## create daily time series train 
occupancy_series_train_gloria=daily_series(df_train.loc[:,'gloria'],190)

occupancy_series_train_magnan=daily_series(df_train.loc[:,'magnan'],190)

occupancy_series_train_philippeN=daily_series(df_train.loc[:,'philippe nord'],190)

occupancy_series_train_philippeS=daily_series(df_train.loc[:,'philippe sud '],190)

occupancy_series_train_cimiezN=daily_series(df_train.loc[:,'cimiez nord'],190)

occupancy_series_train_cimiezS=daily_series(df_train.loc[:,'cimiez sud'],190)

occupancy_series_train_augustin=daily_series(df_train.loc[:,'augustin'],190)

occupancy_series_train_grinda=daily_series(df_train.loc[:,'grinda'],190)


multivariate=np.dstack((occupancy_series_train_gloria[0],occupancy_series_train_magnan[0],occupancy_series_train_philippeN[0],occupancy_series_train_philippeS[0],occupancy_series_train_cimiezN[0],occupancy_series_train_cimiezS[0],occupancy_series_train_augustin[0],occupancy_series_train_grinda[0]))
multivariate_time_series_train = to_time_series(multivariate)
print(multivariate_time_series_train.shape)

## create daily time series train 
occupancy_series_test_gloria=daily_series(df_test.loc[:,'gloria'],190)

occupancy_series_test_magnan=daily_series(df_test.loc[:,'magnan'],190)

occupancy_series_test_philippeN=daily_series(df_test.loc[:,'philippe nord'],190)

occupancy_series_test_philippeS=daily_series(df_test.loc[:,'philippe sud '],190)

occupancy_series_test_cimiezN=daily_series(df_test.loc[:,'cimiez nord'],190)

occupancy_series_test_cimiezS=daily_series(df_test.loc[:,'cimiez sud'],190)

occupancy_series_test_augustin=daily_series(df_test.loc[:,'augustin'],190)

occupancy_series_test_grinda=daily_series(df_test.loc[:,'grinda'],190)


multivariate_test=np.dstack((occupancy_series_test_gloria[0],occupancy_series_test_magnan[0],occupancy_series_test_philippeN[0],occupancy_series_test_philippeS[0],occupancy_series_test_cimiezN[0],occupancy_series_test_cimiezS[0],occupancy_series_test_augustin[0],occupancy_series_test_grinda[0]))
multivariate_time_series_test = to_time_series(multivariate_test)
print(multivariate_time_series_test.shape)

#CLUSTERING 
from tslearn.clustering import TimeSeriesKMeans, KernelKMeans, silhouette_score
from tslearn.metrics import gamma_soft_dtw
#estimate the gamma hyperparameter 

gamma_soft_dtw(dataset=multivariate_time_series_train, n_samples=200,random_state=0) 
#fit the model on train data 
km_dba = TimeSeriesKMeans(n_clusters=4, metric="softdtw",metric_params={"gamma":22.44767991014613}, max_iter=5,max_iter_barycenter=5, random_state=0).fit(multivariate_time_series_train)

#predict train 
prediction_train=km_dba.fit_predict(multivariate_time_series_train[0:1,:,:],y=None)


#silhouette

silhouette_score(multivariate_time_series_train, prediction_train, metric="softdtw",metric_params={"gamma":22.44767991014613})

#visualization 
import calplot
#train
train_days= pd.date_range('9/2/2019',periods=112, freq='D')
#assign at every day the cluster 
events_train = pd.Series(prediction_train,index=train_days)
#plot the result 
calplot.calplot(events_train,yearlabel_kws={'color': 'black'}, cmap='cool', suptitle='Clustering of the days softDTW $\gamma$=22 occupancy 8 detectors', linewidth=2.3)

#PREDICTION
from tslearn.svm import TimeSeriesSVR
from sklearn.metrics import mean_squared_error
from statistics import mean

##### GRID SEARCH
# list of all possible time 
hours=range(20,190)
# grid search for C, epsilon
# for every time of the day return the suggested values of C and epsilon
from sklearn.model_selection import GridSearchCV
def grid_search(data,window_size):
    output=[]
    #model 
    reg = TimeSeriesSVR(kernel="gak", gamma="auto")
    #parameter grid 
    p_grid = {"C": [0.1,1,10,100], "epsilon":[0.01,0.1,1,10]}
    for i in range(0,len(hours)):
        X_train=data[:,0:20+i,:]
        #clustering
        km_dba = TimeSeriesKMeans(n_clusters=4, metric="softdtw",metric_params={"gamma":gamma_soft_dtw(dataset=X_train, n_samples=200,random_state=0) }, max_iter=5,max_iter_barycenter=5, random_state=0).fit(X_train)
        #assign the day you want to predict to a cluster
        prediction_train=km_dba.fit_predict(X_train,y=None)
        cluster_train=X_train[prediction_train==3]
        X_train=cluster_train[:,-window_size-1:-1,:]
        Y_train=cluster_train[:,-1:,:]
        #select most recent observations of X for both train and test
        x_train=X_train[:,:,1:2]
        #select the target Train 
        #choose the detector you want to predict 
        y_train=Y_train[:,0,1]
        clf = GridSearchCV(estimator=reg, param_grid=p_grid, scoring='neg_mean_squared_error')
        clf.fit(x_train,y_train)
        print(clf.best_params_)
        output.append(clf.best_params_)
    return output

grid=grid_search(multivariate_time_series_train[10:15,:,:],10)
def most_frequent(List):
    counter = 0
    num = List[0]
      
    for i in List:
        curr_frequency = List.count(i)
        if(curr_frequency> counter):
            counter = curr_frequency
            num = i
  
    return num
  
print(most_frequent(grid))


#starting time >= window_size

#decide which detector you want to predict 

def walk_forward_validation(train,test,window_size,starting_time):
    #define a starting split X and Y for both train and test
    X_test=test[:,0:starting_time,:]
    Y_test=test[:,starting_time:,:]  
    #initialize the lists
    MSE_train=[]
    MSE_test=[]
    PRED_train=[]
    PRED_test=[]
    GROUND_TRUTH_train=[]
    GROUND_TRUTH_test=[]
    for t in range(0,180): # lenghts of Y_train[1] = lenghts of Y_test[1]
        X_train=train[:,0:starting_time+1+t,:]
        #clustering
        km_dba = TimeSeriesKMeans(n_clusters=4, metric="softdtw",metric_params={"gamma":gamma_soft_dtw(dataset=X_train, n_samples=200,random_state=0) }, max_iter=5,max_iter_barycenter=5, random_state=0).fit(X_train)
        #assign the day you want to predict to a cluster
        prediction_train=km_dba.fit_predict(X_train,y=None)
        prediction_test_cluster= km_dba.predict(X_test)
        cluster_train=X_train[prediction_train==prediction_test_cluster]
        X_train=cluster_train[:,-window_size-1:-1,:]
        Y_train=cluster_train[:,-1:,:]
        #select most recent observations of X for both train and test
        x_train=X_train[:,:,1:2]
        x_test=X_test[:,-window_size:,1:2]
        #select the target Train 
        y_train=Y_train[:,0,1]
        #rescale the target Train 
        ground_truth_train=occupancy_series_train_magnan[1].inverse_transform(y_train.reshape(1, -1))
        GROUND_TRUTH_train.append(ground_truth_train)
        #select the target Test
        y_test=Y_test[:,t,1]
        #rescale the target Test 
        ground_truth_test=occupancy_series_test_magnan[1].inverse_transform(y_test.reshape(1, -1))
        GROUND_TRUTH_test.append(ground_truth_test)
        #model 
        reg = TimeSeriesSVR(C=10,kernel="gak", gamma="auto",epsilon=0.1)
        #fit the model and predict the Train 
        y_hat_train=reg.fit(x_train, y_train).predict(x_train)
        #rescale the prediction of the Train 
        prediction_train=occupancy_series_train_magnan[1].inverse_transform(y_hat_train.reshape(1, -1))
        PRED_train.append(prediction_train)
        # predict the test 
        y_hat_test=reg.predict(x_test)
        #rescale the prediction of the Test 
        prediction_test=occupancy_series_test_magnan[1].inverse_transform(y_hat_test.reshape(1, -1))
        PRED_test.append(prediction_test)
        #compute the mean square error 
        mse_train=mean_squared_error(ground_truth_train, prediction_train)
        mse_test=mean_squared_error(ground_truth_test, prediction_test)
        MSE_train.append(mse_train)
        MSE_test.append(mse_test)
        #Add the curent observation to the X set 
        obs_test=Y_test[:,t:t+1,:]
        X_test=np.hstack((X_test,obs_test))
        print(mse_train,mse_test,prediction_test_cluster)
    return mean(MSE_train), mean(MSE_test), PRED_train, PRED_test, GROUND_TRUTH_train, GROUND_TRUTH_test
        
result=walk_forward_validation(multivariate_time_series_train,multivariate_time_series_test[12:13,:,:],10,10)      
result[0]
result[1]
import matplotlib.pyplot as plt
# plot forecasts against actual outcomes
x= np.arange(6,24,0.1)
len(x)
plt.plot(x,np.concatenate(result[3], axis=0 ),color='blue', label="prediction test") 
plt.plot(x,np.concatenate(result[5], axis=0 ), color='red', label="ground truth") 
plt.legend()
plt.title('13/1/2020 Magnan')
plt.show()




