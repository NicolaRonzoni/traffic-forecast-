#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 09:14:43 2021

@author: nronzoni
"""
pip install statsmodels
#libraries 
import pandas as pd 
import scipy 
import sklearn
import tslearn 
import numpy as np
import random
from toolz.itertoolz import sliding_window, partition
from tslearn.utils import to_time_series, to_time_series_dataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tslearn.clustering import TimeSeriesKMeans, KernelKMeans, silhouette_score
from tslearn.metrics import gamma_soft_dtw
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from tslearn.metrics import soft_dtw, gamma_soft_dtw,dtw
from sklearn.multioutput import MultiOutputRegressor
from tslearn.generators import random_walks
from sklearn.pipeline import Pipeline
from scipy.spatial import distance
import math 
import statsmodels.api as sm
from statistics import mean
from sklearn.multioutput import RegressorChain,ClassifierChain
from sklearn.svm import SVR
from tslearn.generators import random_walks
from tslearn.svm import TimeSeriesSVR
from sklearn.metrics import mean_squared_error
from sklearn import linear_model 
#FOR EVERY VARIABLES: DELETE DAYS WITH NAN VALUES GREATER THAN THE THRESHOLD, DELETE THE NIGHT, DECLARE THE TIME AGGREGATION
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


#density veh/km
def data_split_density(data):
    data_speed=data['Density (veh/km)']
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




#define a function to create the daily time series and the compute the scaling 
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

#define a function to create the daily time series but without the scaling: never used 
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

#Function to decide the number of cluster: NEVER USED
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

#### EXAMPLE OF USAGE OF optimalK FUNCTION 
X = random_walks(n_ts=50, sz=32, d=2)

score_g, df = optimalK(X, nrefs=5, maxClusters=7)

plt.plot(df['clusterCount'], df['gap'], linestyle='--', marker='o', color='b');
plt.xlabel('K');
plt.ylabel('Gap Statistic');
plt.title('Gap Statistic vs. K');

### function for dynamic time warping with weights, put more weight in the last observations: NEVER USED
def dtw_WEIGHT(s, t, window,window_weight):
    n, m = len(s), len(t)
    w = np.max([window, abs(n-m)])
    dtw_matrix = np.zeros((n+1, m+1))
    
    for i in range(n+1):
        for j in range(m+1):
            dtw_matrix[i, j] = np.inf
    dtw_matrix[0, 0] = 0
    
    for i in range(1, n+1):
        for j in range(np.max([1, i-w]), np.min([m, i+w])+1):
            dtw_matrix[i, j] = 0
    
    for i in range(1, n+1):
        for j in range(np.max([1, i-w]), np.min([m, i+w])+1):
            cost = distance.euclidean(s[i-1], t[j-1])**2
            last_min = np.min([dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1]]) 
            # take last min from a square box weighing
            if i in range(n+1-window_weight,n+1):
                 dtw_matrix[i, j] = cost + (1+(1/2)**(window_weight))*last_min
                 window_weight=window_weight-1
            else :
                dtw_matrix[i, j] = cost + last_min
    return math.sqrt(dtw_matrix[n, m])


# function to find the day closest to the centroid using soft-DTW : NEVER USED
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

# function to find the day closest to the centroid using DTW: compute the clustering inside the function, return 5 days closer to the target  
def closest_days_target(train,test,starting_time,window_past):
    km_dba = TimeSeriesKMeans(n_clusters=4, metric="softdtw",metric_params={"gamma":gamma_soft_dtw(dataset=train, n_samples=200,random_state=0) }, max_iter=5,max_iter_barycenter=5, random_state=0).fit(train)
    prediction_train=km_dba.predict(train)
    #take the centroid 
    centroid=km_dba.cluster_centers_
    #for each centroid select only the observations available of the day that we would like to predict 
    train_set=centroid[:,starting_time-window_past:starting_time,:]
    #select observations available in the test
    X_test=test[:,starting_time-window_past:starting_time,:]
    columns=["cluster","sim"]
    df=pd.DataFrame(columns=columns)
    #select the centroid closest to test data 
    for i in range (0,4):
      sim = dtw(X_test[0,:,:],train_set[i,:,:])   
      df = df.append({'cluster': i,'sim': sim}, ignore_index=True)
    df["sim"]=df["sim"].abs()
    cluster=df[df.sim==df.sim.min()].cluster
    print(cluster)
    train_set=train[prediction_train==cluster.values]
    #X and Y split select the loop that we would like to predict 
    X_train=train_set[:,starting_time-window_past:starting_time,:]
    columns1=["ts","sim1"]
    df1=pd.DataFrame(columns=columns1)
    print(train_set.shape[0])
    for j in range(0,train_set.shape[0]):
        sim1 = dtw(X_test[0,:,:],X_train[j,:,:],5,10) 
        df1 = df1.append({'ts': j,'sim1': sim1}, ignore_index=True)
    df1["sim1"]=df1["sim1"].abs()
    ts=df1.nsmallest(5, 'sim1', keep='all').index
    return print(ts)

# function to find the day closest to the  target using DTW: no clustering inside the function, return 5 days closer to the target  
def closest_days_target_speed(train,test,starting_time,window_past):
    #X and Y split select the loop that we would like to predict 
    X_test=test[:,starting_time-window_past:starting_time,:]
    X_train=train[:,starting_time-window_past:starting_time,:]
    columns1=["ts","sim1"]
    df1=pd.DataFrame(columns=columns1)
    print(train.shape[0])
    for j in range(0,train.shape[0]):
        sim1 = dtw(X_test[0,:,:],X_train[j,:,:]) 
        df1 = df1.append({'ts': j,'sim1': sim1}, ignore_index=True)
    df1["sim1"]=df1["sim1"].abs()
    ts=df1.nsmallest(5, 'sim1', keep='all').index
    return print(ts)



#PREDICTIONS: FUNCTION FOR one step and multistep predictions

# sets for C and epsilon hyperparameters, used in the functions
p_grid = {"C": [0.1,1,10,100], "epsilon":[0.01,0.1,1,10]}

# function for one step SVR prediction using all loops of the road segment, and the clustering procedure to select closest days to the target;
# The  number of cluster is set equal to the number of cluster used in the clustering of the entire time series. 
def walk_forward_validation(train,test,window_size,starting_time,loop):
    #define a starting split X and Y for test set 
    X_test=test[:,0:starting_time,:]
    Y_test=test[:,starting_time:,:]
    GROUND_TRUTH_test=[]
    PRED_test=[]
    for t in range(0,10): # number of prediction from starting point
        # take an observation forward: len(X_train)=len(X_test)+1 
        train_set=train[:,0:starting_time+1+t,:]
        #select only most window_size+1 recent observations for train 
        XY_train=train_set[:,-window_size-1:,:]
        #select only most window_size recent observations for test
        X_test=X_test[:,-window_size:,:]
        #clustering
        km_dba = TimeSeriesKMeans(n_clusters=4, metric="softdtw",metric_params={"gamma":gamma_soft_dtw(dataset=XY_train, n_samples=200,random_state=0) }, max_iter=5,max_iter_barycenter=5, random_state=0).fit(XY_train)
        #assign the day you want to predict to a cluster
        prediction_train=km_dba.fit_predict(XY_train,y=None)
        prediction_test_cluster= km_dba.predict(X_test)
        print(prediction_test_cluster)
        #select train observations that belongs to the same cluster that we want to predict
        cluster_train=XY_train[prediction_train==prediction_test_cluster]
        #divide X and Y in the train set 
        X_train=cluster_train[:,-window_size-1:-1,:]
        Y_train=cluster_train[:,-1:,:]
        #select the detector we want to predict
        x_train=X_train.reshape(X_train.shape[0],-1)
        x_test=X_test.reshape(1,-1)
        #select the target Train 
        y_train=Y_train[:,0,loop]
        #select the target Test
        y_test=Y_test[:,t,loop]
        #rescale the target Test 
        GROUND_TRUTH_test.append(y_test)
        #Grid search to tune the parameters
        reg =SVR(kernel="rbf", gamma="auto")
        clf = GridSearchCV(estimator=reg, param_grid=p_grid, scoring='neg_mean_squared_error',refit=True,cv=3)
        #fit the model with the best found parameters
        clf.fit(x_train,y_train)
        # prediction for the test 
        y_hat_test=clf.predict(x_test)
        #rescale the prediction of the Test 
        PRED_test.append(y_hat_test)
        #Add the curent observation to the X set 
        obs_test=Y_test[:,t:t+1,:]
        X_test=np.hstack((X_test,obs_test))
    PRED_test=np.concatenate(PRED_test,axis=0)
    GROUND_TRUTH_test=np.concatenate(GROUND_TRUTH_test,axis=0)
    return PRED_test, GROUND_TRUTH_test
 
# function for one step SVR prediction using all loops in the road segment, NO clustering use of the entire train set
def walk_forward_validation_bis(train,test,window_size,starting_time,loop):
    #define a starting split X and Y for test set 
    X_test=test[:,0:starting_time,:]
    Y_test=test[:,starting_time:,:]
    GROUND_TRUTH_test=[]
    PRED_test=[]
    for t in range(0,10): # number of prediction from starting point
        # take an observation forward: len(X_train)=len(X_test)+1 
        train_set=train[:,0:starting_time+1+t,:]
        #select only most window_size+1 recent observations for train 
        XY_train=train_set[:,-window_size-1:,:]
        #select only most window_size recent observations for test
        X_test=X_test[:,-window_size:,:]
        #divide X and Y in the train set 
        X_train=XY_train[:,-window_size-1:-1,:]
        Y_train=XY_train[:,-1:,:]
        #select the detector we want to predict
        x_train=X_train.reshape(X_train.shape[0],-1)
        x_test=X_test.reshape(1,-1)
        #select the target Train 
        y_train=Y_train[:,0,loop]
        #select the target Test
        y_test=Y_test[:,t,loop]
        #rescale the target Test 
        GROUND_TRUTH_test.append(y_test)
        #Grid search to tune the parameters
        reg =SVR(kernel="rbf", gamma="auto")
        clf = GridSearchCV(estimator=reg, param_grid=p_grid, scoring='neg_mean_squared_error',refit=True,cv=3)
        #fit the model with the best found parameters
        clf.fit(x_train,y_train)
        # prediction for the test 
        y_hat_test=clf.predict(x_test)
        #rescale the prediction of the Test 
        PRED_test.append(y_hat_test)
        #Add the curent observation to the X set 
        obs_test=Y_test[:,t:t+1,:]
        X_test=np.hstack((X_test,obs_test))
    PRED_test=np.concatenate(PRED_test,axis=0)
    GROUND_TRUTH_test=np.concatenate(GROUND_TRUTH_test,axis=0)
    return PRED_test, GROUND_TRUTH_test
 
#function for one step SVR prediction using only the  data of the loop that we want to predict, NO clustering use of the entire train set
def walk_forward_validation_tris(train,test,window_size,starting_time,loop):
    #define a starting split X and Y for test set 
    X_test=test[:,0:starting_time,loop]
    Y_test=test[:,starting_time:,loop]
    GROUND_TRUTH_test=[]
    PRED_test=[]
    for t in range(0,10): # number of prediction from starting point
        # take an observation forward: len(X_train)=len(X_test)+1 
        train_set=train[:,0:starting_time+1+t,loop]
        #select only most window_size+1 recent observations for train 
        XY_train=train_set[:,-window_size-1:]
        #select only most window_size recent observations for test
        X_test=X_test[:,-window_size:]
        #divide X and Y in the train set 
        X_train=XY_train[:,-window_size-1:-1]
        Y_train=XY_train[:,-1:]
        #select the detector we want to predict
        x_train=X_train
        x_test=X_test
        #select the target Train 
        y_train=Y_train
        #select the target Test
        y_test=Y_test[:,t].ravel()
        #rescale the target Test 
        GROUND_TRUTH_test.append(y_test)
        #Grid search to tune the parameters
        reg =SVR(kernel="rbf", gamma="auto")
        clf = GridSearchCV(estimator=reg, param_grid=p_grid, scoring='neg_mean_squared_error',refit=True,cv=3)
        #fit the model with the best found parameters
        clf.fit(x_train,y_train)
        # prediction for the test 
        y_hat_test=clf.predict(x_test)
        #rescale the prediction of the Test 
        PRED_test.append(y_hat_test)
        #Add the curent observation to the X set 
        obs_test=Y_test[:,t:t+1]
        X_test=np.hstack((X_test,obs_test))
    PRED_test=np.concatenate(PRED_test,axis=0)
    GROUND_TRUTH_test=np.concatenate(GROUND_TRUTH_test,axis=0)
    return PRED_test, GROUND_TRUTH_test
 


#Function for one step prediction based only on the clustering of all loops of the road segment. The train set moves forward and as a prediction we took the last observation of the centroid. Be aware that the train set 
# has 1 observation more than test set, this is necessary to select the observation in the future.       
def loubes(train,test,window_size,starting_time):
    #define a starting split X and Y for test set 
    X_test=test[:,0:starting_time,:]
    Y_test=test[:,starting_time:,:]  
    #initialize the list
    PRED_test=[]
    ground_truth=[] 
    MSE_test=[] 
    for t in range(0,30): # number of prediction from starting point 
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
        #select the last observation in the centroid as prediction for X_test with respect to the loop that we would like to predict 
        prediction_test=centroid_train[:,-1:,0]
        true_value=Y_test[:,t:t+1,0]
        print(prediction_test)
        print(prediction_test.shape)
        print(true_value.shape)
        PRED_test.append(prediction_test)
        ground_truth.append(true_value)
        mse_test=mean_squared_error(true_value,prediction_test)
        MSE_test.append(mse_test)
        #Add the curent observation to the X set 
        obs_test=Y_test[:,t:t+1,:]
        X_test=np.hstack((X_test,obs_test))
    PRED_test=np.concatenate(PRED_test,axis=0)
    ground_truth=np.concatenate(ground_truth,axis=0)
    return PRED_test, ground_truth, mean(MSE_test)



#function for multistep prediction, clustering on all loops of the road segment and then recognize the closest day of the with respect to all loops based on DTW:
    #1. see in which cluster belong the target
    #2. pick inside the cluster the closest day to the target
def classification_pred_same(train,test,starting_time,window_future,window_past):
    km_dba = TimeSeriesKMeans(n_clusters=2, metric="softdtw",metric_params={"gamma":gamma_soft_dtw(dataset=train, n_samples=200,random_state=0) }, max_iter=5,max_iter_barycenter=5, random_state=0).fit(train)
    prediction_train=km_dba.predict(train)
    #take the centroid 
    centroid=km_dba.cluster_centers_
    #for each centroid select only the observations available of the day that we would like to predict 
    train_set=centroid[:,starting_time-window_past:starting_time,:]
    #select observations available in the test
    X_test=test[:,starting_time-window_past:starting_time,:]
    columns=["cluster","sim"]
    df=pd.DataFrame(columns=columns)
    #select the centroid closest to test data 
    for i in range (0,2):
      sim = dtw(X_test[0,:,:],train_set[i,:,:])   
      df = df.append({'cluster': i,'sim': sim}, ignore_index=True)
    df["sim"]=df["sim"].abs()
    cluster=df[df.sim==df.sim.min()].cluster
    print(cluster)
    train_set=train[prediction_train==cluster.values]
    #X and Y split only for test set, train set XY together 
    X_train=train_set[:,starting_time-window_past:starting_time,:]
    X_test=test[:,starting_time-window_past:starting_time,:]
    Y_test=test[0,starting_time:starting_time+window_future,:]
    columns1=["ts","sim1"]
    df1=pd.DataFrame(columns=columns1)
    for j in range(0,train_set.shape[0]):
        #compare time series of the same lenght 
        sim1 = dtw(X_test[0,:,:],X_train[j,:,:]) 
        df1 = df1.append({'ts': j,'sim1': sim1}, ignore_index=True)
    df1["sim1"]=df1["sim1"].abs()
    ts=df1[df1.sim1==df1.sim1.min()].ts
    print(ts)
    #select the time series closest to the test and return the window_future as prediction
    Y_pred=train_set[ts.index,starting_time:starting_time+window_future,:]
    return cluster,ts, Y_pred, Y_test

# function for multistep prediction, NO clustering on all loops of the road segment, directly pick up the closest day with respect to all loops based on DTW 

def classification_pred_same_bis(train,test,starting_time,window_future,window_past):
    X_train=train[:,starting_time-window_past:starting_time,:]
    X_test=test[:,starting_time-window_past:starting_time,:]
    Y_test=test[0,starting_time:starting_time+window_future,:]
    columns1=["ts","sim1"]
    df1=pd.DataFrame(columns=columns1)
    for j in range(0,X_train.shape[0]):
        #compare time series of the same lenght 
        sim1 = dtw(X_test[0,:,:],X_train[j,:,:]) 
        df1 = df1.append({'ts': j,'sim1': sim1}, ignore_index=True)
    df1["sim1"]=df1["sim1"].abs()
    ts=df1[df1.sim1==df1.sim1.min()].ts
    print(ts)
    #select the time series closest to the test and return the window_future as prediction
    Y_pred=train[ts.index,starting_time:starting_time+window_future,:]
    return ts, Y_pred, Y_test


#function for multistep prediction: NO clustering on all loops of the road segment, pick up the closest day for every loop in the road segment
#selection of the closest day only considering single loop detector 
def classification_pred_same_tris(train,test,starting_time,window_future,window_past,loop):
    X_train=train[:,starting_time-window_past:starting_time,loop]
    X_test=test[:,starting_time-window_past:starting_time,loop]
    Y_test=test[0,starting_time:starting_time+window_future,loop]
    columns1=["ts","sim1"]
    df1=pd.DataFrame(columns=columns1)
    for j in range(0,X_train.shape[0]):
        #compare time series of the same lenght 
        sim1 = dtw(X_test[0,:],X_train[j,:]) 
        df1 = df1.append({'ts': j,'sim1': sim1}, ignore_index=True)
    df1["sim1"]=df1["sim1"].abs()
    ts=df1[df1.sim1==df1.sim1.min()].ts
    print(ts)
    #select the time series closest to the test and return the window_future as prediction
    Y_pred=train[ts.index,starting_time:starting_time+window_future,loop]
    return ts, Y_pred, Y_test

train= random_walks(n_ts=50, sz=32, d=4)
test = X = random_walks(n_ts=1, sz=32, d=4)
train.shape
test.shape

def classification_pred_same_qatris(train,test,starting_time,window_future,window_past,loop):
    X_train=train[:,starting_time-window_past:starting_time,loop]
    X_test=test[:,starting_time-window_past:starting_time,loop]
    Y_test=test[0,starting_time:starting_time+window_future,loop]
    columns1=["ts","sim1"]
    df1=pd.DataFrame(columns=columns1)
    for j in range(0,X_train.shape[0]):
        #compare time series of the same lenght 
        sim1 = dtw(X_test[0,:],X_train[j,:]) 
        df1 = df1.append({'ts': j,'sim1': sim1}, ignore_index=True)
    df1["sim1"]=df1["sim1"].abs()
    ts=df1[df1.sim1==df1.sim1.min()].ts
    print(ts)
    #select the time series closest to the test and return the window_future as prediction
    Y_pred=train[ts.index,starting_time:starting_time+window_future,loop]
    #history of the closest day
    X_train_bis=train[ts.index,starting_time-window_past:starting_time,loop]
    X_train_bis=X_train_bis[0,]
    #history of the target 
    X_test_bis=test[0,starting_time-window_past:starting_time,loop]
    print(X_test_bis.shape)
    print(X_train_bis.shape)
    #apply ordinary least square 
    #X_train_tris = sm.add_constant(X_train_bis)
    #print(X_train_tris.shape)
    model = sm.OLS(X_test,X_train_bis)
    results = model.fit()
    alpha=results.params[0]
    beta=results.params[1]
    print(alpha,beta)
    #Y_pred_bis=np.array(Y_pred)*beta+alpha
    #print(Y_pred_bis)
    #print(Y_pred)
    return ts, #Y_pred_bis, Y_test

classification_pred_same_qatris(train,test,20,10,10,1)

#function for multistep SVR prediction: clustering on all loops of the road segment to select only day closest to the target. Then to predict single loop use all looop of the road segment
def SVR_pred_d(train,test,starting_time,window_past,window_future,loop):
    km_dba = TimeSeriesKMeans(n_clusters=4, metric="softdtw",metric_params={"gamma":gamma_soft_dtw(dataset=train, n_samples=200,random_state=0) }, max_iter=5,max_iter_barycenter=5, random_state=0).fit(train)
    prediction_train=km_dba.predict(train)
    #take the centroid 
    centroid=km_dba.cluster_centers_
    #for each centroid select only the observations available of the day that we would like to predict 
    train_set=centroid[:,starting_time-window_past:starting_time,:]
    #select observations available in the test
    X_test=test[:,starting_time-window_past:starting_time,:]
    columns=["cluster","sim"]
    df=pd.DataFrame(columns=columns)
    #select the centroid closest to test data 
    for i in range (0,4):
      sim = dtw(X_test[0,:,:],train_set[i,:,:],5,10)   
      df = df.append({'cluster': i,'sim': sim}, ignore_index=True)
    df["sim"]=df["sim"].abs()
    cluster=df[df.sim==df.sim.min()].cluster
    print(cluster)
    train_set=train[prediction_train==cluster.values]
    #X and Y split select the loop that we would like to predict 
    X_train=train_set[:,starting_time-window_past:starting_time,:]
    X_train=X_train.reshape(train_set.shape[0],-1)
    Y_train=train_set[:,starting_time:starting_time+window_future,loop]
    X_test=test[:,starting_time-window_past:starting_time,:]
    X_test=X_test.reshape(1,-1)
    Y_test=test[:,starting_time:starting_time+window_future,loop]
    reg = SVR(kernel="rbf", gamma="auto")
    pipe_svr = Pipeline([('reg', MultiOutputRegressor(reg))])
    grid_param_svr = {"reg__estimator__C": [0.1,1,10,100], "reg__estimator__epsilon":[0.01,0.1,1,10]}
    gs_svr = (GridSearchCV(estimator=pipe_svr, param_grid=grid_param_svr, cv=3,scoring = 'neg_mean_squared_error', n_jobs = -1))
    gs_svr = gs_svr.fit(X_train,Y_train)
    print(gs_svr.best_estimator_) 
    Y_pred=gs_svr.predict(X_test)
    return cluster, Y_pred, Y_test


# function for multistep SVR prediction: NO clustering on all loops of the road segment, just use the whole train set. Then to predict single loop use all looop of the road segment 
def SVR_pred_d_speed(train,test,starting_time,window_past,window_future,loop):
    X_train=train[:,starting_time-window_past:starting_time,:]
    X_train=X_train.reshape(train.shape[0],-1)
    Y_train=train[:,starting_time:starting_time+window_future,loop]
    X_test=test[:,starting_time-window_past:starting_time,:]
    X_test=X_test.reshape(1,-1)
    Y_test=test[:,starting_time:starting_time+window_future,loop]
    reg = SVR(kernel="rbf", gamma="auto")
    pipe_svr = Pipeline([('reg', MultiOutputRegressor(reg))])
    grid_param_svr = {"reg__estimator__C": [0.1,1,10,100], "reg__estimator__epsilon":[0.01,0.1,1,10]}
    gs_svr = (GridSearchCV(estimator=pipe_svr, param_grid=grid_param_svr, cv=3,scoring = 'neg_mean_squared_error', n_jobs = -1))
    gs_svr = gs_svr.fit(X_train,Y_train)
    print(gs_svr.best_estimator_) 
    Y_pred=gs_svr.predict(X_test)
    return  Y_pred, Y_test

# function for multistep SVR prediction : just using data from the loop that we want to predict 
def SVR_pred_d_speed_bis(train,test,starting_time,window_past,window_future,loop):
    X_train=train[:,starting_time-window_past:starting_time,loop]
    X_train=X_train
    Y_train=train[:,starting_time:starting_time+window_future,loop]
    X_test=test[:,starting_time-window_past:starting_time,loop]
    X_test=X_test
    Y_test=test[:,starting_time:starting_time+window_future,loop]
    reg = SVR(kernel="rbf", gamma="auto")
    pipe_svr = Pipeline([('reg', MultiOutputRegressor(reg))])
    grid_param_svr = {"reg__estimator__C": [0.1,1,10,100], "reg__estimator__epsilon":[0.01,0.1,1,10]}
    gs_svr = (GridSearchCV(estimator=pipe_svr, param_grid=grid_param_svr, cv=3,scoring = 'neg_mean_squared_error', n_jobs = -1))
    gs_svr = gs_svr.fit(X_train,Y_train)
    print(gs_svr.best_estimator_) 
    Y_pred=gs_svr.predict(X_test)
    return  Y_pred, Y_test







