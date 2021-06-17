#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 11:43:47 2021

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

#IMPORT DATA S54  
S54= pd.read_excel(r"TrafficData_I35W_2013.xlsx",sheet_name=0) 
#speed
S54_speed=data_split(S54)
#flow
S54_flow=data_split_flow(S54)

#IMPORT DATA S1706  
S1706= pd.read_excel(r"TrafficData_I35W_2013.xlsx",sheet_name=1) 
#speed
S1706_speed=data_split(S1706)
#flow
S1706_flow=data_split_flow(S1706)

#IMPORT DATA Off-Ramp 169  
R169= pd.read_excel(r"TrafficData_I35W_2013.xlsx",sheet_name=2) 

R169_flow=data_split_flow(R169)

#IMPORT DATA S56 
S56=pd.read_excel(r"TrafficData_I35W_2013.xlsx",sheet_name=3) 

S56_speed=data_split(S56)

S56_flow=data_split_flow(S56)

#IMPORT DATA On ramp 129
R129= pd.read_excel(r"TrafficData_I35W_2013.xlsx",sheet_name=4) 

R129_flow=data_split_flow(R129)

#IMPORT DATA S57
S57= pd.read_excel(r"TrafficData_I35W_2013.xlsx",sheet_name=5) 

S57_speed=data_split(S57)

S57_flow=data_split_flow(S57)

#IMPORT DATA Off ramp 170
R170= pd.read_excel(r"TrafficData_I35W_2013.xlsx",sheet_name=6) 

R170_flow=data_split_flow(R170)

#IMPORT DATA S1707
S1707= pd.read_excel(r"TrafficData_I35W_2013.xlsx",sheet_name=7) 

S1707_speed=data_split(S1707)

S1707_flow=data_split_flow(S1707)

#IMPORT DATA S59
S59= pd.read_excel(r"TrafficData_I35W_2013.xlsx",sheet_name=8) 

S59_speed=data_split(S59)

S59_flow=data_split_flow(S59)
#IMPORT DATA On ramp 130
R130= pd.read_excel(r"TrafficData_I35W_2013.xlsx",sheet_name=9) 

R130_flow=data_split_flow(R130)

#IMPORT DATA Off ramp 171
R171= pd.read_excel(r"TrafficData_I35W_2013.xlsx",sheet_name=10) 

R171_flow=data_split_flow(R171)

#IMPORT DATA S60
S60= pd.read_excel(r"TrafficData_I35W_2013.xlsx",sheet_name=11) 

S60_speed=data_split(S60)

S60_flow=data_split_flow(S60)
#IMPORT DATA S61
S61= pd.read_excel(r"TrafficData_I35W_2013.xlsx",sheet_name=12) 

S61_speed=data_split(S61)

S61_flow=data_split_flow(S61)


## create daily time series train 

#S54
#speed
series_train_S54_speed=daily_series(S54_speed[0],180)
series_train_S54_speed[0].shape
print('Min: %f, Max: %f' % (series_train_S54_speed[1].data_min_, series_train_S54_speed[1].data_max_))
#flow
series_train_S54_flow=daily_series(S54_flow[0],180)
series_train_S54_flow[0].shape
print('Min: %f, Max: %f' % (series_train_S54_flow[1].data_min_, series_train_S54_flow[1].data_max_))

#S1706
#speed
series_train_S1706_speed=daily_series(S1706_speed[0],180)
series_train_S1706_speed[0].shape
print('Min: %f, Max: %f' % (series_train_S1706_speed[1].data_min_, series_train_S1706_speed[1].data_max_))
#flow
series_train_S1706_flow=daily_series(S1706_flow[0],180)
series_train_S1706_flow[0].shape
print('Min: %f, Max: %f' % (series_train_S1706_flow[1].data_min_, series_train_S1706_flow[1].data_max_))

#R169 
#flow
series_train_R169_flow=daily_series(R169_flow[0],180)
series_train_R169_flow[0].shape
print('Min: %f, Max: %f' % (series_train_R169_flow[1].data_min_, series_train_R169_flow[1].data_max_))

#S56
#speed
series_train_S56_speed=daily_series(S56_speed[0],180)
series_train_S56_speed[0].shape
print('Min: %f, Max: %f' % (series_train_S56_speed[1].data_min_, series_train_S56_speed[1].data_max_))
#flow
series_train_S56_flow=daily_series(S56_flow[0],180)
series_train_S56_flow[0].shape
print('Min: %f, Max: %f' % (series_train_S56_flow[1].data_min_, series_train_S56_flow[1].data_max_))

#R129
#flow
series_train_R129_flow=daily_series(R129_flow[0],180)
series_train_R129_flow[0].shape
print('Min: %f, Max: %f' % (series_train_R129_flow[1].data_min_, series_train_R129_flow[1].data_max_))

#S57
#speed
series_train_S57_speed=daily_series(S57_speed[0],180)
series_train_S57_speed[0].shape
print('Min: %f, Max: %f' % (series_train_S57_speed[1].data_min_, series_train_S57_speed[1].data_max_))
#flow
series_train_S57_flow=daily_series(S57_flow[0],180)
series_train_S57_flow[0].shape
print('Min: %f, Max: %f' % (series_train_S57_flow[1].data_min_, series_train_S57_flow[1].data_max_))

#R170
#flow
series_train_R170_flow=daily_series(R170_flow[0],180)
series_train_R170_flow[0].shape
print('Min: %f, Max: %f' % (series_train_R170_flow[1].data_min_, series_train_R170_flow[1].data_max_))

#S1707
#speed
series_train_S1707_speed=daily_series(S1707_speed[0],180)
series_train_S1707_speed[0].shape
print('Min: %f, Max: %f' % (series_train_S1707_speed[1].data_min_, series_train_S1707_speed[1].data_max_))
#flow
series_train_S1707_flow=daily_series(S1707_flow[0],180)
series_train_S1707_flow[0].shape
print('Min: %f, Max: %f' % (series_train_S1707_flow[1].data_min_, series_train_S1707_flow[1].data_max_))

#S59
#speed
series_train_S59_speed=daily_series(S59_speed[0],180)
series_train_S59_speed[0].shape
print('Min: %f, Max: %f' % (series_train_S59_speed[1].data_min_, series_train_S59_speed[1].data_max_))
#flow
series_train_S59_flow=daily_series(S59_flow[0],180)
series_train_S59_flow[0].shape
print('Min: %f, Max: %f' % (series_train_S59_flow[1].data_min_, series_train_S59_flow[1].data_max_))

#R130
#flow
series_train_R130_flow=daily_series(R130_flow[0],180)
series_train_R130_flow[0].shape
print('Min: %f, Max: %f' % (series_train_R130_flow[1].data_min_, series_train_R130_flow[1].data_max_))

#R171
#flow
series_train_R171_flow=daily_series(R171_flow[0],180)
series_train_R171_flow[0].shape
print('Min: %f, Max: %f' % (series_train_R171_flow[1].data_min_, series_train_R171_flow[1].data_max_))

#S60
#speed
series_train_S60_speed=daily_series(S60_speed[0],180)
series_train_S60_speed[0].shape
print('Min: %f, Max: %f' % (series_train_S60_speed[1].data_min_, series_train_S60_speed[1].data_max_))
#flow
series_train_S60_flow=daily_series(S60_flow[0],180)
series_train_S60_flow[0].shape
print('Min: %f, Max: %f' % (series_train_S60_flow[1].data_min_, series_train_S60_flow[1].data_max_))

#S61
#speed
series_train_S61_speed=daily_series(S61_speed[0],180)
series_train_S61_speed[0].shape
print('Min: %f, Max: %f' % (series_train_S61_speed[1].data_min_, series_train_S61_speed[1].data_max_))
#flow
series_train_S61_flow=daily_series(S61_flow[0],180)
series_train_S61_flow[0].shape
print('Min: %f, Max: %f' % (series_train_S61_flow[1].data_min_, series_train_S61_flow[1].data_max_))


## create daily time series test 
#S54
#speed
series_test_S54_speed=daily_series(S54_speed[1],180)
series_test_S54_speed[0].shape
print('Min: %f, Max: %f' % (series_test_S54_speed[1].data_min_, series_test_S54_speed[1].data_max_))
#flow
series_test_S54_flow=daily_series(S54_flow[1],180)
series_test_S54_flow[0].shape
print('Min: %f, Max: %f' % (series_test_S54_flow[1].data_min_, series_test_S54_flow[1].data_max_))

#S1706
#speed
series_test_S1706_speed=daily_series(S1706_speed[1],180)
series_test_S1706_speed[0].shape
print('Min: %f, Max: %f' % (series_test_S1706_speed[1].data_min_, series_test_S1706_speed[1].data_max_))
#flow
series_test_S1706_flow=daily_series(S1706_flow[1],180)
series_test_S1706_flow[0].shape
print('Min: %f, Max: %f' % (series_test_S1706_flow[1].data_min_, series_test_S1706_flow[1].data_max_))


#R169
#flow
series_test_R169_flow=daily_series(R169_flow[1],180)
series_test_R169_flow[0].shape
print('Min: %f, Max: %f' % (series_test_R169_flow[1].data_min_, series_test_R169_flow[1].data_max_))

#S56
#speed
series_test_S56_speed=daily_series(S56_speed[1],180)
series_test_S56_speed[0].shape
print('Min: %f, Max: %f' % (series_test_S56_speed[1].data_min_, series_test_S56_speed[1].data_max_))
#flow
series_test_S56_flow=daily_series(S56_flow[1],180)
series_test_S56_flow[0].shape
print('Min: %f, Max: %f' % (series_test_S56_flow[1].data_min_, series_test_S56_flow[1].data_max_))

#R129
#flow
series_test_R129_flow=daily_series(R129_flow[1],180)
series_test_R129_flow[0].shape
print('Min: %f, Max: %f' % (series_test_R129_flow[1].data_min_, series_test_R129_flow[1].data_max_))

#S57
#speed
series_test_S57_speed=daily_series(S57_speed[1],180)
series_test_S57_speed[0].shape
print('Min: %f, Max: %f' % (series_test_S57_speed[1].data_min_, series_test_S57_speed[1].data_max_))
#flow
series_test_S57_flow=daily_series(S57_flow[1],180)
series_test_S57_flow[0].shape
print('Min: %f, Max: %f' % (series_test_S57_flow[1].data_min_, series_test_S57_flow[1].data_max_))

#R170
#flow
series_test_R170_flow=daily_series(R170_flow[1],180)
series_test_R170_flow[0].shape
print('Min: %f, Max: %f' % (series_test_R170_flow[1].data_min_, series_test_R170_flow[1].data_max_))

#S1707
#speed
series_test_S1707_speed=daily_series(S1707_speed[1],180)
series_test_S1707_speed[0].shape
print('Min: %f, Max: %f' % (series_test_S1707_speed[1].data_min_, series_test_S1707_speed[1].data_max_))
#flow
series_test_S1707_flow=daily_series(S1707_flow[1],180)
series_test_S1707_flow[0].shape
print('Min: %f, Max: %f' % (series_test_S1707_flow[1].data_min_, series_test_S1707_flow[1].data_max_))

#S59
#speed
series_test_S59_speed=daily_series(S59_speed[1],180)
series_test_S59_speed[0].shape
print('Min: %f, Max: %f' % (series_test_S59_speed[1].data_min_, series_test_S59_speed[1].data_max_))
#flow
series_test_S59_flow=daily_series(S59_flow[1],180)
series_test_S59_flow[0].shape
print('Min: %f, Max: %f' % (series_test_S59_flow[1].data_min_, series_test_S59_flow[1].data_max_))

#R130
#flow
series_test_R130_flow=daily_series(R130_flow[1],180)
series_test_R130_flow[0].shape
print('Min: %f, Max: %f' % (series_test_R130_flow[1].data_min_, series_test_R130_flow[1].data_max_))

#R171
#flow
series_test_R171_flow=daily_series(R171_flow[1],180)
series_test_R171_flow[0].shape
print('Min: %f, Max: %f' % (series_test_R171_flow[1].data_min_, series_test_R171_flow[1].data_max_))

#S60
#speed
series_test_S60_speed=daily_series(S60_speed[1],180)
series_test_S60_speed[0].shape
print('Min: %f, Max: %f' % (series_test_S60_speed[1].data_min_, series_test_S60_speed[1].data_max_))
#flow
series_test_S60_flow=daily_series(S60_flow[1],180)
series_test_S60_flow[0].shape
print('Min: %f, Max: %f' % (series_test_S60_flow[1].data_min_, series_test_S60_flow[1].data_max_))


#S61
#speed
series_test_S61_speed=daily_series(S61_speed[1],180)
series_test_S61_speed[0].shape
print('Min: %f, Max: %f' % (series_test_S61_speed[1].data_min_, series_test_S61_speed[1].data_max_))
#flow
series_test_S61_flow=daily_series(S61_flow[1],180)
series_test_S61_flow[0].shape
print('Min: %f, Max: %f' % (series_test_S61_flow[1].data_min_, series_test_S61_flow[1].data_max_))


#multivariate time series train
multivariate=np.dstack((series_train_S54_speed[0],series_train_S54_flow[0],series_train_S1706_speed[0],series_train_R169_flow[0],series_train_S56_speed[0],series_train_R129_flow[0],series_train_S57_speed[0],series_train_R170_flow[0],series_train_S1707_speed[0],series_train_S59_speed[0],series_train_R130_flow[0],series_train_R171_flow[0],series_train_S60_speed[0],series_train_S61_speed[0],series_train_S61_flow[0]))
multivariate_time_series_train = to_time_series(multivariate)
print(multivariate_time_series_train.shape)

#multivariate time series test
multivariate_test=np.dstack((series_test_S54_speed[0],series_test_S54_flow[0],series_test_S1706_speed[0],series_test_R169_flow[0],series_test_S56_speed[0],series_test_R129_flow[0],series_test_S57_speed[0],series_test_R170_flow[0],series_test_S1707_speed[0],series_test_S59_speed[0],series_test_R130_flow[0],series_test_R171_flow[0],series_test_S60_speed[0],series_test_S61_speed[0],series_test_S61_flow[0]))
multivariate_time_series_test = to_time_series(multivariate_test)
print(multivariate_time_series_test.shape)

#CLUSTERING

from tslearn.clustering import TimeSeriesKMeans, KernelKMeans, silhouette_score
from tslearn.metrics import gamma_soft_dtw

score_g, df = optimalK(multivariate_time_series_test, nrefs=5, maxClusters=7)

plt.plot(df['clusterCount'], df['gap'], linestyle='--', marker='o', color='b');
plt.xlabel('K');
plt.ylabel('Gap Statistic');
plt.title('Gap Statistic vs. number of cluster, test set');

#estimate the gamma hyperparameter 
gamma_soft_dtw(dataset=multivariate_time_series_train, n_samples=200,random_state=0) 

#fit the model on train data 
km_dba = TimeSeriesKMeans(n_clusters=2, metric="softdtw",metric_params={"gamma":gamma_soft_dtw(dataset=multivariate_time_series_train, n_samples=200,random_state=0) }, max_iter=5,max_iter_barycenter=5, random_state=0).fit(multivariate_time_series_train)

#predict train 
prediction_train=km_dba.fit_predict(multivariate_time_series_train,y=None)

#prediction test 
prediction_test=km_dba.predict(multivariate_time_series_test)

#silhouette
#train 
silhouette_score(multivariate_time_series_train, prediction_train, metric="softdtw",metric_params={"gamma":})
#test 
silhouette_score(multivariate_time_series_test, prediction_test, metric="softdtw",metric_params={"gamma":})


#visualization
import calplot
#train
first_mid=pd.date_range('1/1/2013', periods=171, freq='D')
second_mid=pd.date_range('6/25/2013', periods=61, freq='D')
third_mid=pd.date_range('8/27/2013', periods=12, freq='D')
fourth_mid=pd.date_range('9/10/2013', periods=113, freq='D')


first_mid=pd.Series(data=first_mid)
second_mid=pd.Series(data=second_mid)
third_mid=pd.Series(data=third_mid)
fourth_mid=pd.Series(data=fourth_mid)
index_train=pd.concat([first_mid,second_mid,third_mid,fourth_mid],ignore_index=True)


#plot the result 
new=[]
for i in range(0,357):
    if prediction_train[i] == 0:
        y=0.05
    elif prediction_train[i] !=0: 
        y=prediction_train[i]
    new.append(y)
#assign at every day the cluster
events_train = pd.Series(new,index=index_train)
calplot.calplot(events_train,yearlabel_kws={'color': 'black'}, cmap='Accent', suptitle='TrafficData_I35W_2013 (train): Scenario 3, $\gamma$=184', linewidth=2.3,dropzero=True,vmin=0) 


#test
first_week=pd.date_range('2/10/2014', periods=7, freq='D')
second_week=pd.date_range('3/17/2014', periods=7, freq='D')
third_week=pd.date_range('8/11/2014', periods=7, freq='D')
fourth_week=pd.date_range('9/8/2014', periods=7, freq='D')
fifth_week=pd.date_range('11/3/2014', periods=7, freq='D')

first_week=pd.Series(data=first_week)
second_week=pd.Series(data=second_week)
third_week=pd.Series(data=third_week)
fourth_week=pd.Series(data=fourth_week)
fifth_week=pd.Series(data=fifth_week)

index_test=pd.concat([first_week,second_week,third_week,fourth_week,fifth_week],ignore_index=True)
new=[]
for i in range(0,35):
    if prediction_test[i] == 0:
        y=0.05
    elif prediction_test[i] !=0: 
        y=prediction_test[i]
    new.append(y)
events_test = pd.Series(new,index=index_test)
calplot.calplot(events_test,yearlabel_kws={'color': 'black'}, cmap='Accent', suptitle='TrafficData_I35W_2014 (test): Scenario 3, $\gamma$=184', linewidth=2.3,dropzero=True,vmin=0) 


#centroids 
centroids=km_dba.cluster_centers_

centroids.shape

# Scenario 1
#first cluster k=0
#S54
S54_speed_centroid=centroids[0][:,0]
S54_speed_centroid_0= series_train_S54_speed[1].inverse_transform(S54_speed_centroid.reshape((len(S54_speed_centroid), 1)))
#S1706
S1706_speed_centroid=centroids[0][:,1]
S1706_speed_centroid_0= series_train_S1706_speed[1].inverse_transform(S1706_speed_centroid.reshape((len(S1706_speed_centroid), 1)))
#R169
R169_flow_centroid=centroids[0][:,2]
R169_flow_centroid_0= series_train_R169_flow[1].inverse_transform(R169_flow_centroid.reshape((len(R169_flow_centroid), 1)))
#S56
S56_speed_centroid=centroids[0][:,3]
S56_speed_centroid_0= series_train_S56_speed[1].inverse_transform(S56_speed_centroid.reshape((len(S56_speed_centroid), 1)))
#R129
R129_flow_centroid=centroids[0][:,4]
R129_flow_centroid_0= series_train_R129_flow[1].inverse_transform(R129_flow_centroid.reshape((len(R129_flow_centroid), 1)))
#S57
S57_speed_centroid=centroids[0][:,5]
S57_speed_centroid_0= series_train_S57_speed[1].inverse_transform(S57_speed_centroid.reshape((len(S57_speed_centroid), 1)))
#R170
R170_flow_centroid=centroids[0][:,6]
R170_flow_centroid_0= series_train_R170_flow[1].inverse_transform(R170_flow_centroid.reshape((len(R170_flow_centroid), 1)))
#S1707
S1707_speed_centroid=centroids[0][:,7]
S1707_speed_centroid_0= series_train_S1707_speed[1].inverse_transform(S1707_speed_centroid.reshape((len(S1707_speed_centroid), 1)))
#S59
S59_speed_centroid=centroids[0][:,8]
S59_speed_centroid_0= series_train_S59_speed[1].inverse_transform(S59_speed_centroid.reshape((len(S59_speed_centroid), 1)))
#R130
R130_flow_centroid=centroids[0][:,9]
R130_flow_centroid_0= series_train_R130_flow[1].inverse_transform(R130_flow_centroid.reshape((len(R130_flow_centroid), 1)))
#R171
R171_flow_centroid=centroids[0][:,10]
R171_flow_centroid_0= series_train_R171_flow[1].inverse_transform(R171_flow_centroid.reshape((len(R171_flow_centroid), 1)))
#S60
S60_speed_centroid=centroids[0][:,11]
S60_speed_centroid_0= series_train_S60_speed[1].inverse_transform(S60_speed_centroid.reshape((len(S60_speed_centroid), 1)))
#S61
S61_speed_centroid=centroids[0][:,12]
S61_speed_centroid_0= series_train_S61_speed[1].inverse_transform(S61_speed_centroid.reshape((len(S61_speed_centroid), 1)))

#second cluster k=1
#S54
S54_speed_centroid=centroids[1][:,0]
S54_speed_centroid_1= series_train_S54_speed[1].inverse_transform(S54_speed_centroid.reshape((len(S54_speed_centroid), 1)))
#S1706
S1706_speed_centroid=centroids[1][:,1]
S1706_speed_centroid_1= series_train_S1706_speed[1].inverse_transform(S1706_speed_centroid.reshape((len(S1706_speed_centroid), 1)))
#R169
R169_flow_centroid=centroids[1][:,2]
R169_flow_centroid_1= series_train_R169_flow[1].inverse_transform(R169_flow_centroid.reshape((len(R169_flow_centroid), 1)))
#S56
S56_speed_centroid=centroids[1][:,3]
S56_speed_centroid_1= series_train_S56_speed[1].inverse_transform(S56_speed_centroid.reshape((len(S56_speed_centroid), 1)))
#R129
R129_flow_centroid=centroids[1][:,4]
R129_flow_centroid_1= series_train_R129_flow[1].inverse_transform(R129_flow_centroid.reshape((len(R129_flow_centroid), 1)))
#S57
S57_speed_centroid=centroids[1][:,5]
S57_speed_centroid_1= series_train_S57_speed[1].inverse_transform(S57_speed_centroid.reshape((len(S57_speed_centroid), 1)))
#R170
R170_flow_centroid=centroids[1][:,6]
R170_flow_centroid_1= series_train_R170_flow[1].inverse_transform(R170_flow_centroid.reshape((len(R170_flow_centroid), 1)))
#S1707
S1707_speed_centroid=centroids[1][:,7]
S1707_speed_centroid_1= series_train_S1707_speed[1].inverse_transform(S1707_speed_centroid.reshape((len(S1707_speed_centroid), 1)))
#S59
S59_speed_centroid=centroids[1][:,8]
S59_speed_centroid_1= series_train_S59_speed[1].inverse_transform(S59_speed_centroid.reshape((len(S59_speed_centroid), 1)))
#R130
R130_flow_centroid=centroids[1][:,9]
R130_flow_centroid_1= series_train_R130_flow[1].inverse_transform(R130_flow_centroid.reshape((len(R130_flow_centroid), 1)))
#R171
R171_flow_centroid=centroids[1][:,10]
R171_flow_centroid_1= series_train_R171_flow[1].inverse_transform(R171_flow_centroid.reshape((len(R171_flow_centroid), 1)))
#S60
S60_speed_centroid=centroids[1][:,11]
S60_speed_centroid_1= series_train_S60_speed[1].inverse_transform(S60_speed_centroid.reshape((len(S60_speed_centroid), 1)))
#S61
S61_speed_centroid=centroids[1][:,12]
S61_speed_centroid_1= series_train_S61_speed[1].inverse_transform(S61_speed_centroid.reshape((len(S61_speed_centroid), 1)))

#save centroids of the cluster 
columns = ['S54 speed (km/h)','S1706 speed (km/h)', 'R169 flow (veh/h)','S56 speed (km/h)','R129 flow (veh/h)', 'S57 speed (km/h)','R170 flow (veh/h)','S1707 speed (km/h)', 'S59 speed (km/h)','R130 flow (veh/h)','R171 flow (veh/h)', 'S60 speed (km/h)','S61 speed (km/h)']
index=pd.date_range("5:00", periods=180, freq="6min")
index
df_0 = pd.DataFrame(index=index.time, columns=columns)
df_0['S54 speed (km/h)']=S54_speed_centroid_0
df_0['S1706 speed (km/h)']=S1706_speed_centroid_0
df_0['R169 flow (veh/h)']=R169_flow_centroid_0
df_0['S56 speed (km/h)']=S56_speed_centroid_0
df_0['R129 flow (veh/h)']=R129_flow_centroid_0
df_0['S57 speed (km/h)']=S57_speed_centroid_0
df_0['R170 flow (veh/h)']=R170_flow_centroid_0
df_0['S1707 speed (km/h)']=S1707_speed_centroid_0
df_0['S59 speed (km/h)']=S59_speed_centroid_0
df_0['R130 flow (veh/h)']=R130_flow_centroid_0
df_0['R171 flow (veh/h)']=R171_flow_centroid_0
df_0['S60 speed (km/h)']=S60_speed_centroid_0
df_0['S61 speed (km/h)']=S61_speed_centroid_0

df_0

df_1 = pd.DataFrame(index=index.time, columns=columns)
df_1['S54 speed (km/h)']=S54_speed_centroid_1
df_1['S1706 speed (km/h)']=S1706_speed_centroid_1
df_1['R169 flow (veh/h)']=R169_flow_centroid_1
df_1['S56 speed (km/h)']=S56_speed_centroid_1
df_1['R129 flow (veh/h)']=R129_flow_centroid_1
df_1['S57 speed (km/h)']=S57_speed_centroid_1
df_1['R170 flow (veh/h)']=R170_flow_centroid_1
df_1['S1707 speed (km/h)']=S1707_speed_centroid_1
df_1['S59 speed (km/h)']=S59_speed_centroid_1
df_1['R130 flow (veh/h)']=R130_flow_centroid_1
df_1['R171 flow (veh/h)']=R171_flow_centroid_1
df_1['S60 speed (km/h)']=S60_speed_centroid_1
df_1['S61 speed (km/h)']=S61_speed_centroid_1

df_1

# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter('/Users/nronzoni/Desktop/TrafficData Minnesota/Scenario1.xlsx', engine='xlsxwriter')

# Write each dataframe to a different worksheet.
df_0.to_excel(writer, sheet_name='k=0')
df_1.to_excel(writer, sheet_name='k=1')
# Close the Pandas Excel writer and output the Excel file.
writer.save()


#prediction
result=walk_forward_validation(multivariate_time_series_train,multivariate_time_series_test[0:1,:,:],10,60)      
result[0]
result[1]

# plot forecasts against actual outcomes
x= np.arange(6,24,0.1)
len(x)
plt.plot(x,np.concatenate(result[3], axis=0 ),color='blue', label="prediction test") 
plt.plot(x,np.concatenate(result[5], axis=0 ), color='red', label="ground truth") 
plt.legend()
plt.title('')
plt.show()

#Scenario 3
#first cluster k=0
#S54
#speed
S54_speed_centroid=centroids[0][:,0]
S54_speed_centroid_0= series_train_S54_speed[1].inverse_transform(S54_speed_centroid.reshape((len(S54_speed_centroid), 1)))
#flow
S54_flow_centroid=centroids[0][:,1]
S54_flow_centroid_0= series_train_S54_flow[1].inverse_transform(S54_flow_centroid.reshape((len(S54_flow_centroid), 1)))
#S1706
S1706_speed_centroid=centroids[0][:,2]
S1706_speed_centroid_0= series_train_S1706_speed[1].inverse_transform(S1706_speed_centroid.reshape((len(S1706_speed_centroid), 1)))
#R169
R169_flow_centroid=centroids[0][:,3]
R169_flow_centroid_0= series_train_R169_flow[1].inverse_transform(R169_flow_centroid.reshape((len(R169_flow_centroid), 1)))
#S56
S56_speed_centroid=centroids[0][:,4]
S56_speed_centroid_0= series_train_S56_speed[1].inverse_transform(S56_speed_centroid.reshape((len(S56_speed_centroid), 1)))
#R129
R129_flow_centroid=centroids[0][:,5]
R129_flow_centroid_0= series_train_R129_flow[1].inverse_transform(R129_flow_centroid.reshape((len(R129_flow_centroid), 1)))
#S57
S57_speed_centroid=centroids[0][:,6]
S57_speed_centroid_0= series_train_S57_speed[1].inverse_transform(S57_speed_centroid.reshape((len(S57_speed_centroid), 1)))
#R170
R170_flow_centroid=centroids[0][:,7]
R170_flow_centroid_0= series_train_R170_flow[1].inverse_transform(R170_flow_centroid.reshape((len(R170_flow_centroid), 1)))
#S1707
S1707_speed_centroid=centroids[0][:,8]
S1707_speed_centroid_0= series_train_S1707_speed[1].inverse_transform(S1707_speed_centroid.reshape((len(S1707_speed_centroid), 1)))
#S59
S59_speed_centroid=centroids[0][:,9]
S59_speed_centroid_0= series_train_S59_speed[1].inverse_transform(S59_speed_centroid.reshape((len(S59_speed_centroid), 1)))
#R130
R130_flow_centroid=centroids[0][:,10]
R130_flow_centroid_0= series_train_R130_flow[1].inverse_transform(R130_flow_centroid.reshape((len(R130_flow_centroid), 1)))
#R171
R171_flow_centroid=centroids[0][:,11]
R171_flow_centroid_0= series_train_R171_flow[1].inverse_transform(R171_flow_centroid.reshape((len(R171_flow_centroid), 1)))
#S60
S60_speed_centroid=centroids[0][:,12]
S60_speed_centroid_0= series_train_S60_speed[1].inverse_transform(S60_speed_centroid.reshape((len(S60_speed_centroid), 1)))
#S61
#speed
S61_speed_centroid=centroids[0][:,13]
S61_speed_centroid_0= series_train_S61_speed[1].inverse_transform(S61_speed_centroid.reshape((len(S61_speed_centroid), 1)))
#flow
S61_flow_centroid=centroids[0][:,14]
S61_flow_centroid_0= series_train_S61_flow[1].inverse_transform(S61_flow_centroid.reshape((len(S61_flow_centroid), 1)))

#second cluster k=1
#S54
#speed
S54_speed_centroid=centroids[1][:,0]
S54_speed_centroid_1= series_train_S54_speed[1].inverse_transform(S54_speed_centroid.reshape((len(S54_speed_centroid), 1)))
#flow
S54_flow_centroid=centroids[1][:,1]
S54_flow_centroid_1= series_train_S54_flow[1].inverse_transform(S54_flow_centroid.reshape((len(S54_flow_centroid), 1)))
#S1706
S1706_speed_centroid=centroids[1][:,2]
S1706_speed_centroid_1= series_train_S1706_speed[1].inverse_transform(S1706_speed_centroid.reshape((len(S1706_speed_centroid), 1)))
#R169
R169_flow_centroid=centroids[1][:,3]
R169_flow_centroid_1= series_train_R169_flow[1].inverse_transform(R169_flow_centroid.reshape((len(R169_flow_centroid), 1)))
#S56
S56_speed_centroid=centroids[1][:,4]
S56_speed_centroid_1= series_train_S56_speed[1].inverse_transform(S56_speed_centroid.reshape((len(S56_speed_centroid), 1)))
#R129
R129_flow_centroid=centroids[1][:,5]
R129_flow_centroid_1= series_train_R129_flow[1].inverse_transform(R129_flow_centroid.reshape((len(R129_flow_centroid), 1)))
#S57
S57_speed_centroid=centroids[1][:,6]
S57_speed_centroid_1= series_train_S57_speed[1].inverse_transform(S57_speed_centroid.reshape((len(S57_speed_centroid), 1)))
#R170
R170_flow_centroid=centroids[1][:,7]
R170_flow_centroid_1= series_train_R170_flow[1].inverse_transform(R170_flow_centroid.reshape((len(R170_flow_centroid), 1)))
#S1707
S1707_speed_centroid=centroids[1][:,8]
S1707_speed_centroid_1= series_train_S1707_speed[1].inverse_transform(S1707_speed_centroid.reshape((len(S1707_speed_centroid), 1)))
#S59
S59_speed_centroid=centroids[1][:,9]
S59_speed_centroid_1= series_train_S59_speed[1].inverse_transform(S59_speed_centroid.reshape((len(S59_speed_centroid), 1)))
#R130
R130_flow_centroid=centroids[1][:,10]
R130_flow_centroid_1= series_train_R130_flow[1].inverse_transform(R130_flow_centroid.reshape((len(R130_flow_centroid), 1)))
#R171
R171_flow_centroid=centroids[1][:,11]
R171_flow_centroid_1= series_train_R171_flow[1].inverse_transform(R171_flow_centroid.reshape((len(R171_flow_centroid), 1)))
#S60
S60_speed_centroid=centroids[1][:,12]
S60_speed_centroid_1= series_train_S60_speed[1].inverse_transform(S60_speed_centroid.reshape((len(S60_speed_centroid), 1)))
#S61
#speed
S61_speed_centroid=centroids[1][:,13]
S61_speed_centroid_1= series_train_S61_speed[1].inverse_transform(S61_speed_centroid.reshape((len(S61_speed_centroid), 1)))
#flow
S61_flow_centroid=centroids[1][:,14]
S61_flow_centroid_1= series_train_S61_flow[1].inverse_transform(S61_flow_centroid.reshape((len(S61_flow_centroid), 1)))

#save centroids of the cluster 
columns = ['S54 speed (km/h)','S54 flow (veh/h)','S1706 speed (km/h)', 'R169 flow (veh/h)','S56 speed (km/h)','R129 flow (veh/h)', 'S57 speed (km/h)','R170 flow (veh/h)','S1707 speed (km/h)', 'S59 speed (km/h)','R130 flow (veh/h)','R171 flow (veh/h)', 'S60 speed (km/h)','S61 speed (km/h)','S61 flow (veh/h)']
index=pd.date_range("5:00", periods=180, freq="6min")
index
df_0 = pd.DataFrame(index=index.time, columns=columns)
df_0['S54 speed (km/h)']=S54_speed_centroid_0
df_0['S54 flow (veh/h)']=S54_flow_centroid_0
df_0['S1706 speed (km/h)']=S1706_speed_centroid_0
df_0['R169 flow (veh/h)']=R169_flow_centroid_0
df_0['S56 speed (km/h)']=S56_speed_centroid_0
df_0['R129 flow (veh/h)']=R129_flow_centroid_0
df_0['S57 speed (km/h)']=S57_speed_centroid_0
df_0['R170 flow (veh/h)']=R170_flow_centroid_0
df_0['S1707 speed (km/h)']=S1707_speed_centroid_0
df_0['S59 speed (km/h)']=S59_speed_centroid_0
df_0['R130 flow (veh/h)']=R130_flow_centroid_0
df_0['R171 flow (veh/h)']=R171_flow_centroid_0
df_0['S60 speed (km/h)']=S60_speed_centroid_0
df_0['S61 speed (km/h)']=S61_speed_centroid_0
df_0['S61 flow (veh/h)']=S61_flow_centroid_0
df_0


df_1 = pd.DataFrame(index=index.time, columns=columns)
df_1['S54 speed (km/h)']=S54_speed_centroid_1
df_1['S54 flow (veh/h)']=S54_flow_centroid_1
df_1['S1706 speed (km/h)']=S1706_speed_centroid_1
df_1['R169 flow (veh/h)']=R169_flow_centroid_1
df_1['S56 speed (km/h)']=S56_speed_centroid_1
df_1['R129 flow (veh/h)']=R129_flow_centroid_1
df_1['S57 speed (km/h)']=S57_speed_centroid_1
df_1['R170 flow (veh/h)']=R170_flow_centroid_1
df_1['S1707 speed (km/h)']=S1707_speed_centroid_1
df_1['S59 speed (km/h)']=S59_speed_centroid_1
df_1['R130 flow (veh/h)']=R130_flow_centroid_1
df_1['R171 flow (veh/h)']=R171_flow_centroid_1
df_1['S60 speed (km/h)']=S60_speed_centroid_1
df_1['S61 speed (km/h)']=S61_speed_centroid_1
df_1['S61 flow (veh/h)']=S61_flow_centroid_1
df_1

# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter('/Users/nronzoni/Desktop/TrafficData Minnesota/Scenario3.xlsx', engine='xlsxwriter')

# Write each dataframe to a different worksheet.
df_0.to_excel(writer, sheet_name='k=0')
df_1.to_excel(writer, sheet_name='k=1')
# Close the Pandas Excel writer and output the Excel file.
writer.save()


