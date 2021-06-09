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
prediction_train=km_dba.fit_predict(multivariate_time_series_train,y=None)
#predict test
prediction_test= km_dba.predict(multivariate_time_series_test)

#silhouette

silhouette_score(multivariate_time_series_train, prediction_train, metric="softdtw",metric_params={"gamma":22.44767991014613})

silhouette_score(multivariate_time_series_test, prediction_test, metric="softdtw",metric_params={"gamma":22.44767991014613})

#visualization 
import calplot
#train
train_days= pd.date_range('9/2/2019',periods=112, freq='D')
#assign at every day the cluster 
events_train = pd.Series(prediction_train,index=train_days)
#plot the result 
calplot.calplot(events_train,yearlabel_kws={'color': 'black'}, cmap='cool', suptitle='Clustering of the days softDTW $\gamma$=22 occupancy 8 detectors', linewidth=2.3)
#test
test_days= pd.date_range('1/1/2020',periods=61, freq='D')
#assign at every day the cluster 
events_test = pd.Series(prediction_test,index=test_days)
#plot the result 
calplot.calplot(events_test,yearlabel_kws={'color': 'black'}, cmap='cool', suptitle='Clustering of the days softDTW $\gamma$=22 occupancy 8 detectors test', linewidth=2.3) 
#softDTW $\gamma$=15#

#PREDICTION
from tslearn.svm import TimeSeriesSVR
from sklearn.metrics import mean_squared_error
from statistics import mean
from tslearn.generators import random_walk_blobs

#select time series assigned to the same cluster 
#train
cluster_train=multivariate_time_series_train[prediction_train==3]
cluster_train.shape
#test
cluster_test=multivariate_time_series_test[prediction_test==3]
cluster_test.shape

cluster_test[:,0:10,:].shape

#starting time >= window_size

#decide which detector you want to predict 

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
        x_train=X_train[:,-window_size:,2:3]
        x_test=X_test[:,-window_size:,2:3]
        #select the target Train 
        y_train=Y_train[:,t,2]
        #rescale the target Train 
        ground_truth_train=occupancy_series_train_magnan[1].inverse_transform(y_train.reshape(1, -1))
        GROUND_TRUTH_train.append(ground_truth_train)
        #select the target Test
        y_test=Y_test[:,t,2]
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
        obs_train=Y_train[:,t:t+1,:]
        obs_test=Y_test[:,t:t+1,:]
        X_train=np.hstack((X_train,obs_train))
        X_test=np.hstack((X_test,obs_test))
        print(mse_train,mse_test)
    return mean(MSE_train), mean(MSE_test), PRED_train, PRED_test, GROUND_TRUTH_train, GROUND_TRUTH_test
        
result=walk_forward_validation(cluster_train,cluster_test[2:3,:,:],10,10)      


result[0]
result[1]

result[3]
import matplotlib.pyplot as plt

# plot forecasts against actual outcomes
x= np.arange(6,24,0.1)
len(x)
plt.plot(x,np.concatenate(result[3], axis=0 ),color='blue', label="prediction test") 
plt.plot(x,np.concatenate(result[5], axis=0 ), color='red', label="ground truth") 
plt.legend()
plt.title('13/1/2020 Magnan')
plt.show()





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

from tslearn.generators import random_walk_blobs
X, y = random_walk_blobs(n_ts_per_blob=10, sz=64, d=2, n_blobs=2)
X.shape
import numpy
y = y.astype(numpy.float) + numpy.random.randn(20) * .1
reg = TimeSeriesSVR(kernel="gak", gamma="auto")
reg.fit(X, y).predict(X)
reg.fit(X,y, sample_weight=())



x_train=cluster_train[:,-window_size:,:]
x_test=X_test[:,-window_size:,:]
#select the target Train 
y_train=Y_train[:,t,3]
#rescale the target Train 
ground_truth_train=occupancy_series_train_philippeS[1].inverse_transform(y_train.reshape(1, -1))
GROUND_TRUTH_train.append(ground_truth_train)
#select the target Test
y_test=Y_test[:,t,3]
#rescale the target Test 
ground_truth_test=occupancy_series_test_philippeS[1].inverse_transform(y_test.reshape(1, -1))
GROUND_TRUTH_test.append(ground_truth_test)
#model 
reg = TimeSeriesSVR(kernel="gak", gamma="auto")
#fit the model and predict the Train 
y_hat_train=reg.fit(x_train, y_train,sample_weight=(34,34)).predict(x_train)
