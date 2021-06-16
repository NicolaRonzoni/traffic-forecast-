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

S54_speed=data_split(S54)
#IMPORT DATA S1706  
S1706= pd.read_excel(r"TrafficData_I35W_2013.xlsx",sheet_name=1) 

S1706_speed=data_split(S1706)

#IMPORT DATA Off-Ramp 169  
R169= pd.read_excel(r"TrafficData_I35W_2013.xlsx",sheet_name=2) 

R169_speed=data_split(R169)
#IMPORT DATA S56 
S56= pd.read_excel(r"TrafficData_I35W_2013.xlsx",sheet_name=3) 

S56_speed=data_split(S56)
#IMPORT DATA On ramp 129
R129= pd.read_excel(r"TrafficData_I35W_2013.xlsx",sheet_name=4) 

R129_speed=data_split(R129)

#IMPORT DATA S57
S57= pd.read_excel(r"TrafficData_I35W_2013.xlsx",sheet_name=5) 

S57_speed=data_split(S57)
#IMPORT DATA Off ramp 170
R170= pd.read_excel(r"TrafficData_I35W_2013.xlsx",sheet_name=6) 

R170_speed=data_split(R170)
#IMPORT DATA S1707
S1707= pd.read_excel(r"TrafficData_I35W_2013.xlsx",sheet_name=7) 

S1707_speed=data_split(S1707)
#IMPORT DATA S59
S59= pd.read_excel(r"TrafficData_I35W_2013.xlsx",sheet_name=8) 

S59_speed=data_split(S59)
#IMPORT DATA On ramp 130
R130= pd.read_excel(r"TrafficData_I35W_2013.xlsx",sheet_name=9) 

R130_speed=data_split(R130)
#IMPORT DATA Off ramp 171
R171= pd.read_excel(r"TrafficData_I35W_2013.xlsx",sheet_name=10) 

R171_speed=data_split(R171)
#IMPORT DATA S60
S60= pd.read_excel(r"TrafficData_I35W_2013.xlsx",sheet_name=11) 

S60_speed=data_split(S60)
#IMPORT DATA S61
S61= pd.read_excel(r"TrafficData_I35W_2013.xlsx",sheet_name=12) 

S61_speed=data_split(S61)
#IMPORT DATA S62
S62= pd.read_excel(r"TrafficData_I35W_2013.xlsx",sheet_name=13) 

S62_speed=data_split(S62)


## create daily time series train 
series_train_S1706=daily_series(S1706_speed[0],240)
series_train_S1706[0].shape
print('Min: %f, Max: %f' % (ù.data_min_, series_train_S1706[1].data_max_))

series_train_R169=daily_series(R169_speed[0],240)
series_train_R169[0].shape
print('Min: %f, Max: %f' % (series_train_R169[1].data_min_, series_train_R169[1].data_max_))

series_train_S56=daily_series(S56_speed[0],240)
series_train_S56[0].shape
print('Min: %f, Max: %f' % (series_train_S56[1].data_min_, series_train_S56[1].data_max_))

series_train_R129=daily_series(R129_speed[0],240)
series_train_R129[0].shape
print('Min: %f, Max: %f' % (series_train_R129[1].data_min_, series_train_R129[1].data_max_))

series_train_S57=daily_series(S57_speed[0],240)
series_train_S57[0].shape
print('Min: %f, Max: %f' % (series_train_S57[1].data_min_, series_train_S57[1].data_max_))

series_train_R170=daily_series(R170_speed[0],240)
series_train_R170[0].shape
print('Min: %f, Max: %f' % (series_train_R170[1].data_min_, series_train_R170[1].data_max_))

series_train_S1707=daily_series(S1707_speed[0],240)
series_train_S1707[0].shape
print('Min: %f, Max: %f' % (series_train_S1707[1].data_min_, series_train_S1707[1].data_max_))

series_train_S59=daily_series(S59_speed[0],240)
series_train_S59[0].shape
print('Min: %f, Max: %f' % (series_train_S59[1].data_min_, series_train_S59[1].data_max_))

series_train_R130=daily_series(R130_speed[0],240)
series_train_R130[0].shape
print('Min: %f, Max: %f' % (series_train_R130[1].data_min_, series_train_R130[1].data_max_))

series_train_R171=daily_series(R171_speed[0],240)
series_train_R171[0].shape
print('Min: %f, Max: %f' % (series_train_R171[1].data_min_, series_train_R171[1].data_max_))

series_train_S60=daily_series(S60_speed[0],240)
series_train_S60[0].shape
print('Min: %f, Max: %f' % (series_train_S60[1].data_min_, series_train_S60[1].data_max_))

series_train_S61=daily_series(S61_speed[0],240)
series_train_S61[0].shape
print('Min: %f, Max: %f' % (series_train_S61[1].data_min_, series_train_S61[1].data_max_))

series_train_S62=daily_series(S62_speed[0],240)
series_train_S62[0].shape
print('Min: %f, Max: %f' % (series_train_S62[1].data_min_, series_train_S62[1].data_max_))

multivariate=np.dstack((series_train_S1706[0],series_train_R169[0],series_train_S56[0],series_train_R129[0],series_train_S57[0],series_train_R170[0],series_train_S1707[0],series_train_S59[0],series_train_R130[0],series_train_R171[0],series_train_S60[0],series_train_S61[0]))
multivariate_time_series_train = to_time_series(multivariate)
print(multivariate_time_series_train.shape)

## create daily time series test 
series_test_S1706=daily_series(S1706_speed[1],240)
series_test_S1706[0].shape
print('Min: %f, Max: %f' % (series_test_S1706[1].data_min_, series_test_S1706[1].data_max_))

series_test_R169=daily_series(R169_speed[1],240)
series_test_R169[0].shape
print('Min: %f, Max: %f' % (series_test_R169[1].data_min_, series_test_R169[1].data_max_))

series_test_S56=daily_series(S56_speed[1],240)
series_test_S56[0].shape
print('Min: %f, Max: %f' % (series_test_S56[1].data_min_, series_test_S56[1].data_max_))

series_test_R129=daily_series(R129_speed[1],240)
series_test_R129[0].shape
print('Min: %f, Max: %f' % (series_test_R129[1].data_min_, series_test_R129[1].data_max_))

series_test_S57=daily_series(S57_speed[1],240)
series_test_S57[0].shape
print('Min: %f, Max: %f' % (series_test_S57[1].data_min_, series_test_S57[1].data_max_))

series_test_R170=daily_series(R170_speed[1],240)
series_test_R170[0].shape
print('Min: %f, Max: %f' % (series_test_R170[1].data_min_, series_test_R170[1].data_max_))

series_test_S1707=daily_series(S1707_speed[1],240)
series_test_S1707[0].shape
print('Min: %f, Max: %f' % (series_test_S1707[1].data_min_, series_test_S1707[1].data_max_))

series_test_S59=daily_series(S59_speed[1],240)
series_test_S59[0].shape
print('Min: %f, Max: %f' % (series_test_S59[1].data_min_, series_test_S59[1].data_max_))

series_test_R130=daily_series(R130_speed[1],240)
series_test_R130[0].shape
print('Min: %f, Max: %f' % (series_test_R130[1].data_min_, series_test_R130[1].data_max_))

series_test_R171=daily_series(R171_speed[1],240)
series_test_R171[0].shape
print('Min: %f, Max: %f' % (series_test_R171[1].data_min_, series_test_R171[1].data_max_))

series_test_S60=daily_series(S60_speed[1],240)
series_test_S60[0].shape
print('Min: %f, Max: %f' % (series_test_S60[1].data_min_, series_test_S60[1].data_max_))

series_test_S61=daily_series(S61_speed[1],240)
series_test_S61[0].shape
print('Min: %f, Max: %f' % (series_test_S61[1].data_min_, series_test_S61[1].data_max_))

series_test_S62=daily_series(S62_speed[1],240)
series_test_S62[0].shape
print('Min: %f, Max: %f' % (series_test_S62[1].data_min_, series_test_S62[1].data_max_))

multivariate_test=np.dstack((series_test_S1706[0],series_test_R169[0],series_test_S56[0],series_test_R129[0],series_test_S57[0],series_test_R170[0],series_test_S1707[0],series_test_S59[0],series_test_R130[0],series_test_R171[0],series_test_S60[0],series_test_S61[0]))
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

#prediction test 
prediction_test=km_dba.predict(multivariate_time_series_test[0:1,:,:],y=None)

#silhouette
#train 
silhouette_score(multivariate_time_series_train, prediction_train, metric="softdtw",metric_params={"gamma":22.44767991014613})
#test 
silhouette_score(multivariate_time_series_test, prediction_test, metric="softdtw",metric_params={"gamma":22.44767991014613})


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
calplot.calplot(events_train,yearlabel_kws={'color': 'black'}, cmap='Accent', suptitle='', linewidth=2.3,dropzero=True,vmin=0) 


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
    if prediction_train[i] == 0:
        y=0.05
    elif prediction_train[i] !=0: 
        y=prediction_train[i]
    new.append(y)
events_test = pd.Series(new,index=index_test)
calplot.calplot(events_test,yearlabel_kws={'color': 'black'}, cmap='Accent', suptitle='', linewidth=2.3,dropzero=True,vmin=0) 


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


