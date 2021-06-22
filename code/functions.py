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
from tslearn.clustering import TimeSeriesKMeans, KernelKMeans, silhouette_score
from tslearn.metrics import gamma_soft_dtw
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from tslearn.metrics import soft_dtw, gamma_soft_dtw


#speed km/h
def data_split(data):
    data_speed=data['Speed (km/h)']
    #TRAIN
    # 1/1 20/6
    first_period=data_speed[0:41040]
    index_first_period=pd.date_range('2013-01-01',periods=41040, freq='6min')
    first_period=pd.Series(data=first_period.values, index=index_first_period)
    first_period=first_period.between_time('4:59', '22:59')
    #25/6 24/8
    second_period=data_speed[42000:56640]
    index_second_period=pd.date_range('2013-06-25',periods=14640, freq='6min')
    second_period=pd.Series(data=second_period.values, index=index_second_period)
    second_period=second_period.between_time('4:59', '22:59')
    #27/8 7/9
    third_period=data_speed[57120:60000]
    index_third_period=pd.date_range('2013-08-27',periods=2880, freq='6min')
    third_period=pd.Series(data=third_period.values, index=index_third_period)
    third_period=third_period.between_time('4:59', '22:59')
    #10/9 31/12
    fourth_period=data_speed[60480:87600]
    index_fourth_period=pd.date_range('2013-9-10',periods=27120, freq='6min')
    fourth_period=pd.Series(data=fourth_period.values, index=index_fourth_period)
    fourth_period=fourth_period.between_time('4:59', '22:59')
    train_data=pd.concat([first_period,second_period,third_period,fourth_period])
    len(train_data)
    #TEST
    # Mo, 10.02.  – Sun, 16.02.
    index_first_week=pd.date_range('2014-02-10',periods=1680, freq='6min')
    #Mo, 17.03.  – Sun, 23.03.
    index_second_week=pd.date_range('2014-03-17',periods=1680, freq='6min')
    #Mo, 11.08.  – Sun, 17.08.
    index_third_week=pd.date_range('2014-08-11',periods=1680, freq='6min')
    #Mo, 08.09.  – Sun, 14.09.
    index_fourth_week=pd.date_range('2014-09-18',periods=1680, freq='6min')
    #Mo, 03.11.  – Sun, 09.11.
    index_fifth_week=pd.date_range('2014-11-03',periods=1680, freq='6min')

    index_first_week=pd.Series(data=index_first_week)
    index_second_week=pd.Series(data=index_second_week)
    index_third_week=pd.Series(data=index_third_week)
    index_fourth_week=pd.Series(data=index_fourth_week)
    index_fifth_week=pd.Series(data=index_fifth_week)

    index_test=pd.concat([index_first_week,index_second_week,index_third_week,index_fourth_week,index_fifth_week],ignore_index=True)

    test_data=data_speed[87600:96000]
    test_data=pd.Series(data=test_data.values, index=index_test.values)
    test_data=test_data.between_time('4:59', '22:59')
    len(test_data)
    return train_data,test_data

#flow veh/h
def data_split_flow(data):
    data_speed=data['Flow (veh/h)']
    #TRAIN
    # 1/1 20/6
    first_period=data_speed[0:41040]
    index_first_period=pd.date_range('2013-01-01',periods=41040, freq='6min')
    first_period=pd.Series(data=first_period.values, index=index_first_period)
    first_period=first_period.between_time('4:59', '22:59')
    #25/6 24/8
    second_period=data_speed[42000:56640]
    index_second_period=pd.date_range('2013-06-25',periods=14640, freq='6min')
    second_period=pd.Series(data=second_period.values, index=index_second_period)
    second_period=second_period.between_time('4:59', '22:59')
    #27/8 7/9
    third_period=data_speed[57120:60000]
    index_third_period=pd.date_range('2013-08-27',periods=2880, freq='6min')
    third_period=pd.Series(data=third_period.values, index=index_third_period)
    third_period=third_period.between_time('4:59', '22:59')
    #10/9 31/12
    fourth_period=data_speed[60480:87600]
    index_fourth_period=pd.date_range('2013-9-10',periods=27120, freq='6min')
    fourth_period=pd.Series(data=fourth_period.values, index=index_fourth_period)
    fourth_period=fourth_period.between_time('4:59', '22:59')
    train_data=pd.concat([first_period,second_period,third_period,fourth_period])
    len(train_data)
    #TEST
    # Mo, 10.02.  – Sun, 16.02.
    index_first_week=pd.date_range('2014-02-10',periods=1680, freq='6min')
    #Mo, 17.03.  – Sun, 23.03.
    index_second_week=pd.date_range('2014-03-17',periods=1680, freq='6min')
    #Mo, 11.08.  – Sun, 17.08.
    index_third_week=pd.date_range('2014-08-11',periods=1680, freq='6min')
    #Mo, 08.09.  – Sun, 14.09.
    index_fourth_week=pd.date_range('2014-09-18',periods=1680, freq='6min')
    #Mo, 03.11.  – Sun, 09.11.
    index_fifth_week=pd.date_range('2014-11-03',periods=1680, freq='6min')

    index_first_week=pd.Series(data=index_first_week)
    index_second_week=pd.Series(data=index_second_week)
    index_third_week=pd.Series(data=index_third_week)
    index_fourth_week=pd.Series(data=index_fourth_week)
    index_fifth_week=pd.Series(data=index_fifth_week)

    index_test=pd.concat([index_first_week,index_second_week,index_third_week,index_fourth_week,index_fifth_week],ignore_index=True)

    test_data=data_speed[87600:96000]
    test_data=pd.Series(data=test_data.values, index=index_test.values)
    test_data=test_data.between_time('4:59', '22:59')
    len(test_data)
    return train_data,test_data




#define a function to create the daily time series and the scaling ---> clustering 
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

#define a function to create the daily time series but without the scaling for the prediction 
#the scaling require the knowledge of the all test set, instead in the prediction we assume that 
#we don't have all data from the very beginning but they become available step by step 
def daily_series_pred(data,n):
    #normalization of the data 
    data=np.array(data)
    data=data.reshape((len(data), 1))
    #from array to list 
    series=data.tolist()
    len(series)
    #create daily time series 
    time_series=list(partition(n,series))
    #from list to multidimensional array 
    time_series=np.asarray(time_series)
    #create univariate series for normalized observations 
    daily_time_series = to_time_series(time_series)
    return daily_time_series

# Gap Statistic for K means
def optimalK(data, nrefs=3, maxClusters=15):
    """
    Calculates KMeans optimal K using Gap Statistic 
    Params:
        data: numpy (ts,n,d)
        nrefs: number of sample reference datasets to create
        maxClusters: Maximum number of clusters to test for
    Returns: (gaps, optimalK)
    """
    gaps = np.zeros((len(range(1, maxClusters)),))
    resultsdf = pd.DataFrame({'clusterCount':[], 'gap':[]})
    for gap_index, k in enumerate(range(1, maxClusters)):
# Holder for reference dispersion results
        refDisps = np.zeros(nrefs)
# For n references, generate random sample and perform kmeans getting resulting dispersion of each loop
        for i in range(nrefs):
            
            # Create new random reference set
            randomReference=np.random.random_sample(size=data.shape)
            
            # Fit to it
            km =  TimeSeriesKMeans(n_clusters=k, metric="softdtw",metric_params={"gamma":gamma_soft_dtw(dataset=randomReference, n_samples=200,random_state=0) }, max_iter=5,max_iter_barycenter=5, random_state=0)
            km.fit(randomReference)
            
            refDisp = km.inertia_
            refDisps[i] = refDisp
# Fit cluster to original data and create dispersion
        km = TimeSeriesKMeans(n_clusters=k, metric="softdtw",metric_params={"gamma":gamma_soft_dtw(dataset=data, n_samples=200,random_state=0) }, max_iter=5,max_iter_barycenter=5, random_state=0)
        km.fit(data)
        
        origDisp = km.inertia_
# Calculate gap statistic
        gap = np.log(np.mean(refDisps)) - np.log(origDisp)
# Assign this loop's gap statistic to gaps
        gaps[gap_index] = gap
        
        resultsdf = resultsdf.append({'clusterCount':k, 'gap':gap}, ignore_index=True)
    return (gaps.argmax() + 1, resultsdf)

from tslearn.generators import random_walks
X = random_walks(n_ts=50, sz=32, d=2)

score_g, df = optimalK(X, nrefs=5, maxClusters=7)

plt.plot(df['clusterCount'], df['gap'], linestyle='--', marker='o', color='b');
plt.xlabel('K');
plt.ylabel('Gap Statistic');
plt.title('Gap Statistic vs. K');

#find the day closest to the centroid 
def closest(multivariate_time_series_train,prediction_train,centroids,k,events_train):
    if k==0:
        c=k+0.05
    else:
        c=k
    index=events_train[events_train==c].index
    columns=["day","sim"]
    df=pd.DataFrame(columns=columns)
    cluster=multivariate_time_series_train[prediction_train==k]
    for i in range(0,cluster.shape[0]):
        sim = soft_dtw(cluster[i,:,:], centroids[k,:,:], gamma=gamma_soft_dtw(dataset=multivariate_time_series_train, n_samples=200,random_state=0))
        df = df.append({'day': i,'sim': sim}, ignore_index=True)
    df["sim"]=df["sim"].abs()
    df.index=index
    return print(df[df.sim==df.sim.min()])

#PREDICTION
from tslearn.svm import TimeSeriesSVR
from sklearn.metrics import mean_squared_error
from statistics import mean

p_grid = {"C": [0.1,1,10,100], "epsilon":[0.01,0.1,1,10]}

def walk_forward_validation(train,test,window_size,starting_time):
    #define a starting split X and Y for test set 
    X_test=test[:,0:starting_time,:]
    Y_test=test[:,starting_time:,:]  
    #initialize the lists
    MSE_train=[]
    MSE_test=[]
    PRED_train=[]
    PRED_test=[]
    GROUND_TRUTH_train=[]
    GROUND_TRUTH_test=[]
    for t in range(0,30): # number of prediction from starting point to the end of the day
        # take an observation forward: len(X_train)=len(X_test)+1 
        train_set=train[:,0:starting_time+1+t,:]
        #select only most window_size+1 recent observations for train 
        XY_train=train_set[:,-window_size-1:,:]
        #select only most window_size recent observations for test
        X_test=X_test[:,-window_size:,:]
        #clustering
        km_dba = TimeSeriesKMeans(n_clusters=3, metric="softdtw",metric_params={"gamma":gamma_soft_dtw(dataset=XY_train, n_samples=200,random_state=0) }, max_iter=5,max_iter_barycenter=5, random_state=0).fit(XY_train)
        #assign the day you want to predict to a cluster
        prediction_train=km_dba.fit_predict(XY_train,y=None)
        prediction_test_cluster= km_dba.predict(X_test)
        #select train observations that belongs to the same cluster that we want to predict
        cluster_train=XY_train[prediction_train==prediction_test_cluster]
        #divide X and Y in the train set 
        X_train=cluster_train[:,-window_size-1:-1,:]
        Y_train=cluster_train[:,-1:,:]
        #select the detector we want to predict
        x_train=X_train[:,:,:]
        x_test=X_test[:,:,:]
        #select the target Train 
        y_train=Y_train[:,0,0]
        #rescale the target Train 
        GROUND_TRUTH_train.append(y_train)
        #select the target Test
        y_test=Y_test[:,t,0]
        #rescale the target Test 
        GROUND_TRUTH_test.append(y_test)
        #Grid search to tune the parameters
        reg = TimeSeriesSVR(kernel="gak", gamma="auto")
        clf = GridSearchCV(estimator=reg, param_grid=p_grid, scoring='neg_mean_squared_error',refit=True,cv=2)
        #fit the model with the best found parameters
        clf.fit(x_train,y_train)
        #prediction for the train
        y_hat_train=clf.predict(x_train)
        # prediction for the test 
        y_hat_test=clf.predict(x_test)
        #rescale the prediction of the Train 
        PRED_train.append(y_hat_train)
        #rescale the prediction of the Test 
        PRED_test.append(y_hat_test)
        #compute the mean square error 
        mse_train=mean_squared_error(y_train,y_hat_train)
        mse_test=mean_squared_error(y_test,y_hat_test)
        MSE_train.append(mse_train)
        MSE_test.append(mse_test)
        #Add the curent observation to the X set 
        obs_test=Y_test[:,t:t+1,:]
        X_test=np.hstack((X_test,obs_test))
        print(mse_train,mse_test,prediction_test_cluster)
    return mean(MSE_train), mean(MSE_test), PRED_train, PRED_test, GROUND_TRUTH_train, GROUND_TRUTH_test
 


      
def loubes(train,test,window_size,starting_time):
    #define a starting split X and Y for test set 
    X_test=test[:,0:starting_time,:]
    Y_test=test[:,starting_time:,:]  
    #initialize the list
    PRED_test=[]
    for t in range(0,5): # number of prediction from starting point to the end of the day
        # take an observation forward: len(X_train)=len(X_test)+1 
        train_set=train[:,0:starting_time+1+t,:]
        #select only most window_size+1 recent observations for train 
        XY_train=train_set[:,-window_size-1:,:]
        #select only most window_size recent observations for test
        X_test=X_test[:,-window_size:,:]
        #clustering
        km_dba = TimeSeriesKMeans(n_clusters=3, metric="softdtw",metric_params={"gamma":gamma_soft_dtw(dataset=XY_train, n_samples=200,random_state=0) }, max_iter=5,max_iter_barycenter=5, random_state=0).fit(XY_train)
        #assign the day you want to predict to a cluster
        km_dba.fit(XY_train,y=None)
        prediction_test_cluster= km_dba.predict(X_test)
        #select centroids of the cluster in which belong X_test
        centroid_train=km_dba.cluster_centers_[prediction_test_cluster]
        print(centroid_train.shape)
        #select the last observation in the centroid as prediction for X_test 
        prediction_test=centroid_train[:,-1:,:]
        print(prediction_test)
        print(prediction_test.shape)
        PRED_test.append(prediction_test)
        #Add the curent observation to the X set 
        obs_test=Y_test[:,t:t+1,:]
        X_test=np.hstack((X_test,obs_test))
    PRED_test=np.concatenate(PRED_test,axis=0)
    return PRED_test

from tslearn.generators import random_walks
train=random_walks(n_ts=50, sz=30, d=4)
test=random_walks(n_ts=1, sz=30, d=4)


prediction=loubes(train,test,10,25)

prediction.shape
x=np.concatenate(prediction, axis=0 )
x.shape

x[:,:,1] 