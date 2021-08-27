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
from tslearn.metrics import soft_dtw, gamma_soft_dtw,dtw
from datetime import datetime
from sklearn.multioutput import MultiOutputRegressor
from tslearn.generators import random_walks
from sklearn.pipeline import Pipeline
from tslearn.clustering import TimeSeriesKMeans, KernelKMeans, silhouette_score
from tslearn.metrics import gamma_soft_dtw
import calplot
#IMPORT DATA S54  
S54= pd.read_excel(r"TrafficData_I35W_2013.xlsx",sheet_name=0) 
#speed
S54_speed=data_split(S54)
#flow
S54_flow=data_split_flow(S54)
#density
S54_density=data_split_density(S54)

#IMPORT DATA S1706  
S1706= pd.read_excel(r"TrafficData_I35W_2013.xlsx",sheet_name=1) 
#speed
S1706_speed=data_split(S1706)
#flow
S1706_flow=data_split_flow(S1706)
#density
S1706_density=data_split_density(S1706)

#IMPORT DATA Off-Ramp 169  
R169= pd.read_excel(r"TrafficData_I35W_2013.xlsx",sheet_name=2) 
#speed
R169_speed=data_split(R169)
#flow
R169_flow=data_split_flow(R169)
#density
R169_density=data_split_density(R169)

#IMPORT DATA S56 
S56=pd.read_excel(r"TrafficData_I35W_2013.xlsx",sheet_name=3) 

S56_speed=data_split(S56)

S56_flow=data_split_flow(S56)

S56_density=data_split_density(S56)

#IMPORT DATA On ramp 129
R129= pd.read_excel(r"TrafficData_I35W_2013.xlsx",sheet_name=4) 

R129_speed=data_split(R129)

R129_flow=data_split_flow(R129)

R129_density=data_split_density(R129)

#IMPORT DATA S57
S57= pd.read_excel(r"TrafficData_I35W_2013.xlsx",sheet_name=5) 

S57_speed=data_split(S57)

S57_flow=data_split_flow(S57)

S57_density=data_split_density(S57)

#IMPORT DATA Off ramp 170
R170= pd.read_excel(r"TrafficData_I35W_2013.xlsx",sheet_name=6) 

R170_speed=data_split(R170)

R170_flow=data_split_flow(R170)

R170_density=data_split_density(R170)

#IMPORT DATA S1707
S1707= pd.read_excel(r"TrafficData_I35W_2013.xlsx",sheet_name=7) 

S1707_speed=data_split(S1707)

S1707_flow=data_split_flow(S1707)

S1707_density=data_split_density(S1707)

#IMPORT DATA S59
S59= pd.read_excel(r"TrafficData_I35W_2013.xlsx",sheet_name=8) 

S59_speed=data_split(S59)

S59_flow=data_split_flow(S59)

S59_density=data_split_density(S59)
#IMPORT DATA On ramp 130
R130= pd.read_excel(r"TrafficData_I35W_2013.xlsx",sheet_name=9) 

R130_speed=data_split(R130)
R130_flow=data_split_flow(R130)
R130_density=data_split_density(R130)

#IMPORT DATA Off ramp 171
R171= pd.read_excel(r"TrafficData_I35W_2013.xlsx",sheet_name=10) 

R171_speed=data_split(R171)

R171_flow=data_split_flow(R171)
R171_density=data_split_density(R171)
#IMPORT DATA S60
S60= pd.read_excel(r"TrafficData_I35W_2013.xlsx",sheet_name=11) 

S60_speed=data_split(S60)

S60_flow=data_split_flow(S60)

S60_density=data_split_density(S60)
#IMPORT DATA S61
S61= pd.read_excel(r"TrafficData_I35W_2013.xlsx",sheet_name=12) 

S61_speed=data_split(S61)

S61_flow=data_split_flow(S61)

S61_density=data_split_density(S61)


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
#density
series_train_S54_density=daily_series(S54_density[0],180)
series_train_S54_density[0].shape
print('Min: %f, Max: %f' % (series_train_S54_density[1].data_min_, series_train_S54_density[1].data_max_))


#S1706
#speed
series_train_S1706_speed=daily_series(S1706_speed[0],180)
series_train_S1706_speed[0].shape
print('Min: %f, Max: %f' % (series_train_S1706_speed[1].data_min_, series_train_S1706_speed[1].data_max_))
#flow
series_train_S1706_flow=daily_series(S1706_flow[0],180)
series_train_S1706_flow[0].shape
print('Min: %f, Max: %f' % (series_train_S1706_flow[1].data_min_, series_train_S1706_flow[1].data_max_))
#density
series_train_S1706_density=daily_series(S1706_density[0],180)
series_train_S1706_density[0].shape
print('Min: %f, Max: %f' % (series_train_S1706_density[1].data_min_, series_train_S1706_density[1].data_max_))


#R169 
#speed
series_train_R169_speed=daily_series(R169_speed[0],180)
series_train_R169_speed[0].shape
print('Min: %f, Max: %f' % (series_train_R169_speed[1].data_min_, series_train_R169_speed[1].data_max_))

#flow
series_train_R169_flow=daily_series(R169_flow[0],180)
series_train_R169_flow[0].shape
print('Min: %f, Max: %f' % (series_train_R169_flow[1].data_min_, series_train_R169_flow[1].data_max_))
#density
series_train_R169_density=daily_series(R169_density[0],180)
series_train_R169_density[0].shape
print('Min: %f, Max: %f' % (series_train_R169_density[1].data_min_, series_train_R169_density[1].data_max_))


#S56
#speed
series_train_S56_speed=daily_series(S56_speed[0],180)
series_train_S56_speed[0].shape
print('Min: %f, Max: %f' % (series_train_S56_speed[1].data_min_, series_train_S56_speed[1].data_max_))
#flow
series_train_S56_flow=daily_series(S56_flow[0],180)
series_train_S56_flow[0].shape
print('Min: %f, Max: %f' % (series_train_S56_flow[1].data_min_, series_train_S56_flow[1].data_max_))
#density
series_train_S56_density=daily_series(S56_density[0],180)
series_train_S56_density[0].shape
print('Min: %f, Max: %f' % (series_train_S56_density[1].data_min_, series_train_S56_density[1].data_max_))


#R129
#speed
series_train_R129_speed=daily_series(R129_speed[0],180)
series_train_R129_speed[0].shape
print('Min: %f, Max: %f' % (series_train_R129_speed[1].data_min_, series_train_R129_speed[1].data_max_))

#flow
series_train_R129_flow=daily_series(R129_flow[0],180)
series_train_R129_flow[0].shape
print('Min: %f, Max: %f' % (series_train_R129_flow[1].data_min_, series_train_R129_flow[1].data_max_))

#density
series_train_R129_density=daily_series(R129_density[0],180)
series_train_R129_density[0].shape
print('Min: %f, Max: %f' % (series_train_R129_density[1].data_min_, series_train_R129_density[1].data_max_))

#S57
#speed
series_train_S57_speed=daily_series(S57_speed[0],180)
series_train_S57_speed[0].shape
print('Min: %f, Max: %f' % (series_train_S57_speed[1].data_min_, series_train_S57_speed[1].data_max_))
#flow
series_train_S57_flow=daily_series(S57_flow[0],180)
series_train_S57_flow[0].shape
print('Min: %f, Max: %f' % (series_train_S57_flow[1].data_min_, series_train_S57_flow[1].data_max_))
#density
series_train_S57_density=daily_series(S57_density[0],180)
series_train_S57_density[0].shape
print('Min: %f, Max: %f' % (series_train_S57_density[1].data_min_, series_train_S57_density[1].data_max_))


#R170
#speed
series_train_R170_speed=daily_series(R170_speed[0],180)
series_train_R170_speed[0].shape
print('Min: %f, Max: %f' % (series_train_R170_speed[1].data_min_, series_train_R170_speed[1].data_max_))

#flow
series_train_R170_flow=daily_series(R170_flow[0],180)
series_train_R170_flow[0].shape
print('Min: %f, Max: %f' % (series_train_R170_flow[1].data_min_, series_train_R170_flow[1].data_max_))

#density
series_train_R170_density=daily_series(R170_density[0],180)
series_train_R170_density[0].shape
print('Min: %f, Max: %f' % (series_train_R170_density[1].data_min_, series_train_R170_density[1].data_max_))


#S1707
#speed
series_train_S1707_speed=daily_series(S1707_speed[0],180)
series_train_S1707_speed[0].shape
print('Min: %f, Max: %f' % (series_train_S1707_speed[1].data_min_, series_train_S1707_speed[1].data_max_))
#flow
series_train_S1707_flow=daily_series(S1707_flow[0],180)
series_train_S1707_flow[0].shape
print('Min: %f, Max: %f' % (series_train_S1707_flow[1].data_min_, series_train_S1707_flow[1].data_max_))
#density
series_train_S1707_density=daily_series(S1707_density[0],180)
series_train_S1707_density[0].shape
print('Min: %f, Max: %f' % (series_train_S1707_density[1].data_min_, series_train_S1707_density[1].data_max_))

#S59
#speed
series_train_S59_speed=daily_series(S59_speed[0],180)
series_train_S59_speed[0].shape
print('Min: %f, Max: %f' % (series_train_S59_speed[1].data_min_, series_train_S59_speed[1].data_max_))
#flow
series_train_S59_flow=daily_series(S59_flow[0],180)
series_train_S59_flow[0].shape
print('Min: %f, Max: %f' % (series_train_S59_flow[1].data_min_, series_train_S59_flow[1].data_max_))
#density
series_train_S59_density=daily_series(S59_density[0],180)
series_train_S59_density[0].shape
print('Min: %f, Max: %f' % (series_train_S59_density[1].data_min_, series_train_S59_density[1].data_max_))


#R130
#speed
series_train_R130_speed=daily_series(R130_speed[0],180)
series_train_R130_speed[0].shape
print('Min: %f, Max: %f' % (series_train_R130_speed[1].data_min_, series_train_R130_speed[1].data_max_))

#flow
series_train_R130_flow=daily_series(R130_flow[0],180)
series_train_R130_flow[0].shape
print('Min: %f, Max: %f' % (series_train_R130_flow[1].data_min_, series_train_R130_flow[1].data_max_))

#density
series_train_R130_density=daily_series(R130_density[0],180)
series_train_R130_density[0].shape
print('Min: %f, Max: %f' % (series_train_R130_density[1].data_min_, series_train_R130_density[1].data_max_))

#R171
#speed
series_train_R171_speed=daily_series(R171_speed[0],180)
series_train_R171_speed[0].shape
print('Min: %f, Max: %f' % (series_train_R171_speed[1].data_min_, series_train_R171_speed[1].data_max_))

#flow
series_train_R171_flow=daily_series(R171_flow[0],180)
series_train_R171_flow[0].shape
print('Min: %f, Max: %f' % (series_train_R171_flow[1].data_min_, series_train_R171_flow[1].data_max_))

#density
series_train_R171_density=daily_series(R171_density[0],180)
series_train_R171_density[0].shape
print('Min: %f, Max: %f' % (series_train_R171_density[1].data_min_, series_train_R171_density[1].data_max_))

#S60
#speed
series_train_S60_speed=daily_series(S60_speed[0],180)
series_train_S60_speed[0].shape
print('Min: %f, Max: %f' % (series_train_S60_speed[1].data_min_, series_train_S60_speed[1].data_max_))
#flow
series_train_S60_flow=daily_series(S60_flow[0],180)
series_train_S60_flow[0].shape
print('Min: %f, Max: %f' % (series_train_S60_flow[1].data_min_, series_train_S60_flow[1].data_max_))
#density
series_train_S60_density=daily_series(S60_density[0],180)
series_train_S60_density[0].shape
print('Min: %f, Max: %f' % (series_train_S60_density[1].data_min_, series_train_S60_density[1].data_max_))

#S61
#speed
series_train_S61_speed=daily_series(S61_speed[0],180)
series_train_S61_speed[0].shape
print('Min: %f, Max: %f' % (series_train_S61_speed[1].data_min_, series_train_S61_speed[1].data_max_))
#flow
series_train_S61_flow=daily_series(S61_flow[0],180)
series_train_S61_flow[0].shape
print('Min: %f, Max: %f' % (series_train_S61_flow[1].data_min_, series_train_S61_flow[1].data_max_))
#density
series_train_S61_density=daily_series(S61_density[0],180)
series_train_S61_density[0].shape
print('Min: %f, Max: %f' % (series_train_S61_density[1].data_min_, series_train_S61_density[1].data_max_))


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
#density
series_test_S54_density=daily_series(S54_density[1],180)
series_test_S54_density[0].shape
print('Min: %f, Max: %f' % (series_test_S54_density[1].data_min_, series_test_S54_density[1].data_max_))


#S1706
#speed
series_test_S1706_speed=daily_series(S1706_speed[1],180)
series_test_S1706_speed[0].shape
print('Min: %f, Max: %f' % (series_test_S1706_speed[1].data_min_, series_test_S1706_speed[1].data_max_))
#flow
series_test_S1706_flow=daily_series(S1706_flow[1],180)
series_test_S1706_flow[0].shape
print('Min: %f, Max: %f' % (series_test_S1706_flow[1].data_min_, series_test_S1706_flow[1].data_max_))
#density
series_test_S1706_density=daily_series(S1706_density[1],180)
series_test_S1706_density[0].shape
print('Min: %f, Max: %f' % (series_test_S1706_density[1].data_min_, series_test_S1706_density[1].data_max_))


#R169
#speed
series_test_R169_speed=daily_series(R169_speed[1],180)
series_test_R169_speed[0].shape
print('Min: %f, Max: %f' % (series_test_R169_speed[1].data_min_, series_test_R169_speed[1].data_max_))

#flow
series_test_R169_flow=daily_series(R169_flow[1],180)
series_test_R169_flow[0].shape
print('Min: %f, Max: %f' % (series_test_R169_flow[1].data_min_, series_test_R169_flow[1].data_max_))

#density
series_test_R169_density=daily_series(R169_density[1],180)
series_test_R169_density[0].shape
print('Min: %f, Max: %f' % (series_test_R169_density[1].data_min_, series_test_R169_density[1].data_max_))

#S56
#speed
series_test_S56_speed=daily_series(S56_speed[1],180)
series_test_S56_speed[0].shape
print('Min: %f, Max: %f' % (series_test_S56_speed[1].data_min_, series_test_S56_speed[1].data_max_))
#flow
series_test_S56_flow=daily_series(S56_flow[1],180)
series_test_S56_flow[0].shape
print('Min: %f, Max: %f' % (series_test_S56_flow[1].data_min_, series_test_S56_flow[1].data_max_))

#density
series_test_S56_density=daily_series(S56_density[1],180)
series_test_S56_density[0].shape
print('Min: %f, Max: %f' % (series_test_S56_density[1].data_min_, series_test_S56_density[1].data_max_))

#R129
#speed
series_test_R129_speed=daily_series(R129_speed[1],180)
series_test_R129_speed[0].shape
print('Min: %f, Max: %f' % (series_test_R129_speed[1].data_min_, series_test_R129_speed[1].data_max_))

#flow
series_test_R129_flow=daily_series(R129_flow[1],180)
series_test_R129_flow[0].shape
print('Min: %f, Max: %f' % (series_test_R129_flow[1].data_min_, series_test_R129_flow[1].data_max_))

#density
series_test_R129_density=daily_series(R129_density[1],180)
series_test_R129_density[0].shape
print('Min: %f, Max: %f' % (series_test_R129_density[1].data_min_, series_test_R129_density[1].data_max_))

#S57
#speed
series_test_S57_speed=daily_series(S57_speed[1],180)
series_test_S57_speed[0].shape
print('Min: %f, Max: %f' % (series_test_S57_speed[1].data_min_, series_test_S57_speed[1].data_max_))
#flow
series_test_S57_flow=daily_series(S57_flow[1],180)
series_test_S57_flow[0].shape
print('Min: %f, Max: %f' % (series_test_S57_flow[1].data_min_, series_test_S57_flow[1].data_max_))
#density
series_test_S57_density=daily_series(S57_density[1],180)
series_test_S57_density[0].shape
print('Min: %f, Max: %f' % (series_test_S57_density[1].data_min_, series_test_S57_density[1].data_max_))

#R170
#speed
series_test_R170_speed=daily_series(R170_speed[1],180)
series_test_R170_speed[0].shape
print('Min: %f, Max: %f' % (series_test_R170_speed[1].data_min_, series_test_R170_speed[1].data_max_))

#flow
series_test_R170_flow=daily_series(R170_flow[1],180)
series_test_R170_flow[0].shape
print('Min: %f, Max: %f' % (series_test_R170_flow[1].data_min_, series_test_R170_flow[1].data_max_))
#density
series_test_R170_density=daily_series(R170_density[1],180)
series_test_R170_density[0].shape
print('Min: %f, Max: %f' % (series_test_R170_density[1].data_min_, series_test_R170_density[1].data_max_))

#S1707
#speed
series_test_S1707_speed=daily_series(S1707_speed[1],180)
series_test_S1707_speed[0].shape
print('Min: %f, Max: %f' % (series_test_S1707_speed[1].data_min_, series_test_S1707_speed[1].data_max_))
#flow
series_test_S1707_flow=daily_series(S1707_flow[1],180)
series_test_S1707_flow[0].shape
print('Min: %f, Max: %f' % (series_test_S1707_flow[1].data_min_, series_test_S1707_flow[1].data_max_))
#density
series_test_S1707_density=daily_series(S1707_density[1],180)
series_test_S1707_density[0].shape
print('Min: %f, Max: %f' % (series_test_S1707_density[1].data_min_, series_test_S1707_density[1].data_max_))

#S59
#speed
series_test_S59_speed=daily_series(S59_speed[1],180)
series_test_S59_speed[0].shape
print('Min: %f, Max: %f' % (series_test_S59_speed[1].data_min_, series_test_S59_speed[1].data_max_))
#flow
series_test_S59_flow=daily_series(S59_flow[1],180)
series_test_S59_flow[0].shape
print('Min: %f, Max: %f' % (series_test_S59_flow[1].data_min_, series_test_S59_flow[1].data_max_))
#density
series_test_S59_density=daily_series(S59_density[1],180)
series_test_S59_density[0].shape
print('Min: %f, Max: %f' % (series_test_S59_density[1].data_min_, series_test_S59_density[1].data_max_))

#R130
#speed
series_test_R130_speed=daily_series(R130_speed[1],180)
series_test_R130_speed[0].shape
print('Min: %f, Max: %f' % (series_test_R130_speed[1].data_min_, series_test_R130_speed[1].data_max_))

#flow
series_test_R130_flow=daily_series(R130_flow[1],180)
series_test_R130_flow[0].shape
print('Min: %f, Max: %f' % (series_test_R130_flow[1].data_min_, series_test_R130_flow[1].data_max_))

#density
series_test_R130_density=daily_series(R130_density[1],180)
series_test_R130_density[0].shape
print('Min: %f, Max: %f' % (series_test_R130_density[1].data_min_, series_test_R130_density[1].data_max_))

#R171
#speed
series_test_R171_speed=daily_series(R171_speed[1],180)
series_test_R171_speed[0].shape
print('Min: %f, Max: %f' % (series_test_R171_speed[1].data_min_, series_test_R171_speed[1].data_max_))

#flow
series_test_R171_flow=daily_series(R171_flow[1],180)
series_test_R171_flow[0].shape
print('Min: %f, Max: %f' % (series_test_R171_flow[1].data_min_, series_test_R171_flow[1].data_max_))

#density
series_test_R171_density=daily_series(R171_density[1],180)
series_test_R171_density[0].shape
print('Min: %f, Max: %f' % (series_test_R171_density[1].data_min_, series_test_R171_density[1].data_max_))

#S60
#speed
series_test_S60_speed=daily_series(S60_speed[1],180)
series_test_S60_speed[0].shape
print('Min: %f, Max: %f' % (series_test_S60_speed[1].data_min_, series_test_S60_speed[1].data_max_))
#flow
series_test_S60_flow=daily_series(S60_flow[1],180)
series_test_S60_flow[0].shape
print('Min: %f, Max: %f' % (series_test_S60_flow[1].data_min_, series_test_S60_flow[1].data_max_))
#density
series_test_S60_density=daily_series(S60_density[1],180)
series_test_S60_density[0].shape
print('Min: %f, Max: %f' % (series_test_S60_density[1].data_min_, series_test_S60_density[1].data_max_))


#S61
#speed
series_test_S61_speed=daily_series(S61_speed[1],180)
series_test_S61_speed[0].shape
print('Min: %f, Max: %f' % (series_test_S61_speed[1].data_min_, series_test_S61_speed[1].data_max_))
#flow
series_test_S61_flow=daily_series(S61_flow[1],180)
series_test_S61_flow[0].shape
print('Min: %f, Max: %f' % (series_test_S61_flow[1].data_min_, series_test_S61_flow[1].data_max_))
#density
series_test_S61_density=daily_series(S61_density[1],180)
series_test_S61_density[0].shape
print('Min: %f, Max: %f' % (series_test_S61_density[1].data_min_, series_test_S61_density[1].data_max_))



#multivariate time series train flow 
multivariate=np.dstack((series_train_S54_flow[0],series_train_S1706_flow[0],series_train_R169_flow[0],series_train_S56_flow[0],series_train_R129_flow[0],series_train_S57_flow[0],series_train_R170_flow[0],series_train_S1707_flow[0],series_train_S59_flow[0],series_train_R130_flow[0],series_train_R171_flow[0],series_train_S60_flow[0],series_train_S61_flow[0]))
multivariate_time_series_train = to_time_series(multivariate)
print(multivariate_time_series_train.shape)

#multivariate time series test flow 
multivariate_test=np.dstack((series_test_S54_flow[0],series_test_S1706_flow[0],series_test_R169_flow[0],series_test_S56_flow[0],series_test_R129_flow[0],series_test_S57_flow[0],series_test_R170_flow[0],series_test_S1707_flow[0],series_test_S59_flow[0],series_test_R130_flow[0],series_test_R171_flow[0],series_test_S60_flow[0],series_test_S61_flow[0]))
multivariate_time_series_test = to_time_series(multivariate_test)
print(multivariate_time_series_test.shape)

#multivariate time series train speed
#check if multivariate_speed already exist
multivariate_speed=np.dstack((series_train_S54_speed[0],series_train_S1706_speed[0],series_train_R169_speed[0],series_train_S56_speed[0],series_train_R129_speed[0],series_train_S57_speed[0],series_train_R170_speed[0],series_train_S1707_speed[0],series_train_S59_speed[0],series_train_R130_speed[0],series_train_R171_speed[0],series_train_S60_speed[0],series_train_S61_speed[0]))
multivariate_time_series_train_speed = to_time_series(multivariate_speed)
print(multivariate_time_series_train_speed.shape)


#multivariate time series test speed
multivariate_test_speed=np.dstack((series_test_S54_speed[0],series_test_S1706_speed[0],series_test_R169_speed[0],series_test_S56_speed[0],series_test_R129_speed[0],series_test_S57_speed[0],series_test_R170_speed[0],series_test_S1707_speed[0],series_test_S59_speed[0],series_test_R130_speed[0],series_test_R171_speed[0],series_test_S60_speed[0],series_test_S61_speed[0]))
multivariate_time_series_test_speed = to_time_series(multivariate_test_speed)
print(multivariate_time_series_test_speed.shape)




#CLUSTERING
#estimate the gamma hyperparameter 
gamma_soft_dtw(dataset=multivariate_time_series_train_speed, n_samples=200,random_state=0) 

#fit the model on train data 
km_dba = TimeSeriesKMeans(n_clusters=2, metric="softdtw",metric_params={"gamma":gamma_soft_dtw(dataset=multivariate_time_series_train_speed, n_samples=200,random_state=0) }, max_iter=5,max_iter_barycenter=5, random_state=0).fit(multivariate_time_series_train_speed)

#predict train 
prediction_train=km_dba.fit_predict(multivariate_time_series_train_speed,y=None)

#prediction test 
prediction_test=km_dba.predict(multivariate_time_series_test_speed)


#VISUALIZATION 

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


#plot the result change 0 to 0.05 otherwise the calplot get confused  
new=[]
for i in range(0,):
    if prediction_train[i] == 0:
        y=0.05
    elif prediction_train[i] !=0: 
        y=prediction_train[i]
    new.append(y)
#### use these  cycles  if you want to change colors between cluster   
for i in range(0,357):
    if new[i] == 0.05:
        new[i] =4
        
for i in range(0,357):
    if new[i] == 2:
        new[i] =0.05

for i in range(0,357):
    if new[i] ==4:
        new[i] =2

#assign at every day the cluster
events_train = pd.Series(new,index=index_train)
calplot.calplot(events_train,yearlabel_kws={'color': 'black'}, cmap='Accent', suptitle='TrafficData_I35W_2013 (train): loops', linewidth=2.3,dropzero=True,vmin=0) 


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

#plot the result change 0 to 0.05 otherwise the calplot get confused 
for i in range(0,35):
    if prediction_test[i] == 0:
        y=0.05
    elif prediction_test[i] !=0: 
        y=prediction_test[i]
    new.append(y)
#### use these  cycles  if you want to change colors between cluster   
for i in range(0,35):
    if new[i] == 0.05:
        new[i] =4
        
for i in range(0,35):
    if new[i] == 2:
        new[i] =0.05

for i in range(0,35):
    if new[i] ==4:
        new[i] =2


events_test = pd.Series(new,index=index_test)
calplot.calplot(events_test,yearlabel_kws={'color': 'black'}, cmap='Accent', suptitle='TrafficData_I35W_2014 (test):loops', linewidth=2.3,dropzero=True,vmin=0) 


#dataframe to mark days with the correspondent cluster.
len(index_train)
columns=['days','k']
index=np.arange(357)
len(index)
dataframe_train=pd.DataFrame(columns=columns,index=index)
dataframe_train['days']=index_train
dataframe_train['k']=prediction_train
dataframe_train
dataframe_train['day'] = dataframe_train['days'].dt.day
dataframe_train['month'] =dataframe_train['days'].dt.month
dataframe_train['year'] = dataframe_train['days'].dt.year
dataframe_train.drop(['days'], axis=1)
dataframe_train = dataframe_train[['year', 'month', 'day', 'k']]
dataframe_train
dataframe_train.to_excel('/Users/nronzoni/Desktop/TrafficData Minnesota/Prediction without ramps clustering on the speed/Classification of the days no ramps clusteringspeed.xlsx')

# if the clustering is done with respect to the flow and we wanto to select the days inside a specific cluster with respec to another variable (speed) run the following rows
days_cluster=dataframe_train[dataframe_train['k']==1].index
len(days_cluster)
days_cluster[100]

pd.set_option('display.max_seq_items', 300)
print(days_cluster)
multivariate_time_series_train_speed_subset=multivariate_time_series_train_speed[(  0,   2,   4,   5,   6,  11,  12,  13,  14,  18,  19,  20,  21,
             25,  27,  32,  33,  36,  39,  40,  41,  46,  47,  48,  53,  54,
             55,  60,  61,  67,  68,  69,  70,  74,  75,  81,  82,  83,  84,
             87,  88,  89,  90,  91,  92,  93,  95,  96,  99, 102, 103, 104,
            106, 108, 109, 110, 116, 117, 123, 124, 137, 138, 144, 145, 146,
            151, 152, 158, 159, 165, 166, 175, 176, 180, 181, 182, 183, 189,
            190, 196, 197, 203, 204, 210, 211, 212, 217, 218, 224, 225, 236,
            237, 238, 243, 248, 249, 255, 256, 262, 263, 264, 269, 270, 276,
            277, 283, 284, 285, 287, 290, 291, 292, 297, 298, 301, 304, 305,
            306, 311, 312, 318, 319, 320, 322, 323, 324, 325, 326, 332, 340,
            346, 347, 349, 350, 351, 352, 353, 354, 356),:,:]

multivariate_time_series_train_speed_subset.shape

multivariate_time_series_train_subset=multivariate_time_series_train[( 0,   2,   4,   5,   6,  11,  12,  13,  14,  18,  19,  20,  21,
             25,  27,  32,  33,  36,  39,  40,  41,  46,  47,  48,  53,  54,
             55,  60,  61,  67,  68,  69,  70,  74,  75,  81,  82,  83,  84,
             87,  88,  89,  90,  91,  92,  93,  95,  96,  99, 102, 103, 104,
            106, 108, 109, 110, 116, 117, 123, 124, 137, 138, 144, 145, 146,
            151, 152, 158, 159, 165, 166, 175, 176, 180, 181, 182, 183, 189,
            190, 196, 197, 203, 204, 210, 211, 212, 217, 218, 224, 225, 236,
            237, 238, 243, 248, 249, 255, 256, 262, 263, 264, 269, 270, 276,
            277, 283, 284, 285, 287, 290, 291, 292, 297, 298, 301, 304, 305,
            306, 311, 312, 318, 319, 320, 322, 323, 324, 325, 326, 332, 340,
            346, 347, 349, 350, 351, 352, 353, 354, 356),:,:]

multivariate_time_series_train_subset.shape






#centroids 
centroids=km_dba.cluster_centers_

centroids.shape


############# plot of the centroids 
# centroid of each cluster with a random sample of time series inside the same cluster 
########################################### k=4 ################################

##### first cluster #######
cluster1=multivariate_time_series_test_speed[prediction_train==0]

random.shuffle(cluster1)

sample1=cluster1[0:20]

sample1.shape
# inverse of the normalization for each time series in the sample
S54_sample1=sample1[:,:,0]
S54_sample1=series_test_S54_speed[1].inverse_transform(S54_sample1)
S54_sample1.shape

S1706_sample1=sample1[:,:,1]
S1706_sample1=series_test_S1706_speed[1].inverse_transform(S1706_sample1)
S1706_sample1.shape

S56_sample1=sample1[:,:,2]
S56_sample1=series_test_S56_speed[1].inverse_transform(S56_sample1)
S56_sample1.shape

S57_sample1=sample1[:,:,3]
S57_sample1=series_test_S57_speed[1].inverse_transform(S57_sample1)
S57_sample1.shape

S1707_sample1=sample1[:,:,4]
S1707_sample1=series_test_S1707_speed[1].inverse_transform(S1707_sample1)
S1707_sample1.shape

S59_sample1=sample1[:,:,5]
S59_sample1=series_test_S59_speed[1].inverse_transform(S59_sample1)
S59_sample1.shape

S60_sample1=sample1[:,:,6]
S60_sample1=series_test_S60_speed[1].inverse_transform(S60_sample1)
S60_sample1.shape

S61_sample1=sample1[:,:,7]
S61_sample1=series_test_S61_speed[1].inverse_transform(S61_sample1)
S61_sample1.shape


####second cluster #######

cluster2=multivariate_time_series_test_speed[prediction_train==1]


random.shuffle(cluster2)

sample2=cluster2[0:20]

sample2.shape
# inverse of the normalization for each time series in the sample
S54_sample2=sample2[:,:,0]
S54_sample2=series_test_S54_speed[1].inverse_transform(S54_sample2)
S54_sample2.shape

S1706_sample2=sample2[:,:,1]
S1706_sample2=series_test_S1706_speed[1].inverse_transform(S1706_sample2)
S1706_sample2.shape

S56_sample2=sample2[:,:,2]
S56_sample2=series_test_S56_speed[1].inverse_transform(S56_sample2)
S56_sample2.shape

S57_sample2=sample2[:,:,3]
S57_sample2=series_test_S57_speed[1].inverse_transform(S57_sample2)
S57_sample2.shape

S1707_sample2=sample2[:,:,4]
S1707_sample2=series_test_S1707_speed[1].inverse_transform(S1707_sample2)
S1707_sample2.shape

S59_sample2=sample2[:,:,5]
S59_sample2=series_test_S59_speed[1].inverse_transform(S59_sample2)
S59_sample2.shape

S60_sample2=sample2[:,:,6]
S60_sample2=series_test_S60_speed[1].inverse_transform(S60_sample2)
S60_sample2.shape

S61_sample2=sample2[:,:,7]
S61_sample2=series_test_S61_speed[1].inverse_transform(S61_sample2)
S61_sample2.shape


#select randomly time series from third cluster 
cluster3=multivariate_time_series_train[prediction_train==2]

random.shuffle(cluster3)

sample3=cluster3[0:20]

sample3.shape

# inverse of the normalization for each time series in the sample
S54_sample3=sample3[:,:,0]
S54_sample3=series_train_S54_flow[1].inverse_transform(S54_sample3)
S54_sample3.shape

S1706_sample3=sample3[:,:,1]
S1706_sample3=series_train_S1706_flow[1].inverse_transform(S1706_sample3)
S1706_sample3.shape

S56_sample3=sample3[:,:,2]
S56_sample3=series_train_S56_flow[1].inverse_transform(S56_sample3)
S56_sample3.shape

S57_sample3=sample3[:,:,3]
S57_sample3=series_train_S57_flow[1].inverse_transform(S57_sample3)
S57_sample3.shape

S1707_sample3=sample3[:,:,4]
S1707_sample3=series_train_S1707_flow[1].inverse_transform(S1707_sample3)
S1707_sample3.shape

S59_sample3=sample3[:,:,5]
S59_sample3=series_train_S59_flow[1].inverse_transform(S59_sample3)
S59_sample3.shape

S60_sample3=sample3[:,:,6]
S60_sample3=series_train_S60_flow[1].inverse_transform(S60_sample3)
S60_sample3.shape

S61_sample3=sample3[:,:,7]
S61_sample3=series_train_S61_flow[1].inverse_transform(S61_sample3)
S61_sample3.shape

#select randomly time series from fourth cluster 
cluster4=multivariate_time_series_train[prediction_train==3]

random.shuffle(cluster4)

sample4=cluster4[0:20]

sample4.shape
# inverse of the normalization for each time series in the sample
S54_sample4=sample4[:,:,0]
S54_sample4=series_train_S54_flow[1].inverse_transform(S54_sample4)
S54_sample4.shape

S1706_sample4=sample4[:,:,1]
S1706_sample4=series_train_S1706_flow[1].inverse_transform(S1706_sample4)
S1706_sample4.shape

S56_sample4=sample4[:,:,2]
S56_sample4=series_train_S56_flow[1].inverse_transform(S56_sample4)
S56_sample4.shape

S57_sample4=sample4[:,:,3]
S57_sample4=series_train_S57_flow[1].inverse_transform(S57_sample4)
S57_sample4.shape

S1707_sample4=sample4[:,:,4]
S1707_sample4=series_train_S57_flow[1].inverse_transform(S1707_sample4)
S1707_sample4.shape

S59_sample4=sample4[:,:,5]
S59_sample4=series_train_S59_flow[1].inverse_transform(S59_sample4)
S59_sample4.shape

S60_sample4=sample4[:,:,6]
S60_sample4=series_train_S60_flow[1].inverse_transform(S60_sample4)
S60_sample4.shape

S61_sample4=sample4[:,:,7]
S61_sample4=series_train_S61_flow[1].inverse_transform(S61_sample4)
S61_sample4.shape


# centroids 


#k=0#
S54_1=centroids[0][:,0]
S54_1=S54_1.reshape((len(S54_1), 1))

S1706_1=centroids[0][:,1]
S1706_1=S1706_1.reshape((len(S1706_1), 1))

S56_1=centroids[0][:,2]
S56_1=S56_1.reshape((len(S56_1), 1))

S57_1=centroids[0][:,3]
S57_1=S57_1.reshape((len(S57_1), 1))

S1707_1=centroids[0][:,4]
S1707_1=S1707_1.reshape((len(S1707_1), 1))

S59_1=centroids[0][:,5]
S59_1=S59_1.reshape((len(S59_1), 1))

S60_1=centroids[0][:,6]
S60_1=S60_1.reshape((len(S60_1), 1))

S61_1=centroids[0][:,7]
S61_1=S61_1.reshape((len(S61_1), 1))

#k=1#
S54_2=centroids[1][:,0]
S54_2=S54_2.reshape((len(S54_2), 1))

S1706_2=centroids[1][:,1]
S1706_2=S1706_2.reshape((len(S1706_2), 1))

S56_2=centroids[1][:,2]
S56_2=S56_2.reshape((len(S56_2), 1))

S57_2=centroids[1][:,3]
S57_2=S57_2.reshape((len(S57_2), 1))

S1707_2=centroids[1][:,4]
S1707_2=S1707_2.reshape((len(S1707_2), 1))

S59_2=centroids[1][:,5]
S59_2=S59_2.reshape((len(S59_2), 1))

S60_2=centroids[1][:,6]
S60_2=S60_2.reshape((len(S60_2), 1))

S61_2=centroids[1][:,7]
S61_2=S61_2.reshape((len(S61_2), 1))

#k=2#
S54_3=centroids[2][:,0]
S54_3=S54_3.reshape((len(S54_3), 1))

S1706_3=centroids[2][:,1]
S1706_3=S1706_3.reshape((len(S1706_3), 1))

S56_3=centroids[2][:,2]
S56_3=S56_3.reshape((len(S56_3), 1))

S57_3=centroids[2][:,3]
S57_3=S57_3.reshape((len(S57_3), 1))

S1707_3=centroids[2][:,4]
S1707_3=S1707_3.reshape((len(S1707_3), 1))

S59_3=centroids[2][:,5]
S59_3=S59_3.reshape((len(S59_3), 1))

S60_3=centroids[2][:,6]
S60_3=S60_3.reshape((len(S60_3), 1))

S61_3=centroids[2][:,7]
S61_3=S61_3.reshape((len(S61_3), 1))

#k=3#
S54_4=centroids[3][:,0]
S54_4=S54_4.reshape((len(S54_4), 1))

S1706_4=centroids[3][:,1]
S1706_4=S1706_4.reshape((len(S1706_4), 1))

S56_4=centroids[3][:,2]
S56_4=S56_4.reshape((len(S56_4), 1))

S57_4=centroids[3][:,3]
S57_4=S57_4.reshape((len(S57_4), 1))

S1707_4=centroids[3][:,4]
S1707_4=S1707_4.reshape((len(S1707_4), 1))

S59_4=centroids[3][:,5]
S59_4=S59_4.reshape((len(S59_4), 1))

S60_4=centroids[3][:,6]
S60_4=S60_4.reshape((len(S60_4), 1))

S61_4=centroids[3][:,7]
S61_4=S61_4.reshape((len(S61_4), 1))

import matplotlib.pyplot as plt
fig = plt.gcf()

x=np.arange(5,23,0.1)
len(x)


plt.figure(figsize=(35,30))
plt.subplot(2,8,1)
for i in range(0,20):
    plt.plot(x,S54_sample1[i],'k-', alpha=.8)
plt.plot(x,series_train_S54_flow[1].inverse_transform(S54_1),'#33cc33', label = 'S54',linewidth=3)
plt.xlabel('hours of the day',fontsize=18)
plt.ylabel('km/h',labelpad=0,fontsize=18)
plt.ylim((0,200))
plt.title('k=0',fontsize=18)
plt.legend(loc='upper right',fontsize=18)
plt.subplot(2,8,2)
for i in range(0,20):
    plt.plot(x,S1706_sample1[i],'k-', alpha=.8)
plt.plot(x,series_train_S1706_flow[1].inverse_transform(S1706_1),'#33cc33', label = 'S1706',linewidth=3)
plt.xlabel('hours of the day',fontsize=18)
plt.ylabel('km/h',labelpad=0,fontsize=18)
plt.ylim((0,200))
plt.title('k=0',fontsize=18)
plt.legend(loc='upper right',fontsize=18)
plt.subplot(2,8,3)
for i in range(0,20):
    plt.plot(x,S56_sample1[i],'k-', alpha=.8)
plt.plot(x,series_train_S56_flow[1].inverse_transform(S56_1),'#33cc33', label = 'S56',linewidth=3)
plt.xlabel('hours of the day',fontsize=18)
plt.ylabel('km/h',labelpad=0,fontsize=18)
plt.ylim((0,200))
plt.title('k=0',fontsize=18)
plt.legend(loc='upper right',fontsize=18)
plt.subplot(2,8,4)
for i in range(0,20):
    plt.plot(x,S57_sample1[i],'k-', alpha=.8)
plt.plot(x,series_train_S57_flow[1].inverse_transform(S57_1),'#33cc33', label = 'S57',linewidth=3)
plt.xlabel('hours of the day',fontsize=18)
plt.ylabel('km/h',labelpad=0,fontsize=18)
plt.ylim((0,200))
plt.title('k=0',fontsize=18)
plt.legend(loc='upper right',fontsize=18)
plt.subplot(2,8,5)
for i in range(0,20):
    plt.plot(x,S1707_sample1[i],'k-', alpha=.8)
plt.plot(x,series_train_S1707_flow[1].inverse_transform(S1707_1),'#33cc33', label = 'S1707',linewidth=3)
plt.xlabel('hours of the day',fontsize=18)
plt.ylabel('km/h',labelpad=0,fontsize=18)
plt.ylim((0,200))
plt.title('k=0',fontsize=18)
plt.legend(loc='upper right',fontsize=18)
plt.subplot(2,8,6)
for i in range(0,20):
    plt.plot(x,S59_sample1[i],'k-', alpha=.8)
plt.plot(x,series_train_S59_flow[1].inverse_transform(S59_1),'#33cc33', label = 'S59',linewidth=3)
plt.xlabel('hours of the day',fontsize=18)
plt.ylabel('km/h',labelpad=0,fontsize=18)
plt.ylim((0,200))
plt.title('k=0',fontsize=18)
plt.legend(loc='upper right',fontsize=18)
plt.subplot(2,8,7)
for i in range(0,20):
    plt.plot(x,S60_sample1[i],'k-', alpha=.8)
plt.plot(x,series_train_S60_flow[1].inverse_transform(S60_1),'#33cc33', label = 'S60',linewidth=3)
plt.xlabel('hours of the day',fontsize=18)
plt.ylabel('km/h',labelpad=0,fontsize=18)
plt.ylim((0,200))
plt.title('k=0',fontsize=18)
plt.legend(loc='upper right',fontsize=18)
plt.subplot(2,8,8)
for i in range(0,20):
    plt.plot(x,S61_sample1[i],'k-', alpha=.8)
plt.plot(x,series_train_S61_flow[1].inverse_transform(S61_1),'#33cc33', label = 'S61',linewidth=3)
plt.xlabel('hours of the day',fontsize=18)
plt.ylabel('km/h',labelpad=0,fontsize=18)
plt.ylim((0,200))
plt.title('k=0',fontsize=18)
plt.legend(loc='upper right',fontsize=18)
plt.subplot(2,8,9)
for i in range(0,20):
    plt.plot(x,S54_sample2[i],'k-', alpha=.1)
plt.plot(x,series_train_S54_flow[1].inverse_transform(S54_2),'#666699', label = 'S54',linewidth=3)
plt.xlabel('hours of the day',fontsize=18)
plt.ylabel('km/h',labelpad=0,fontsize=18)
plt.ylim((0,200))
plt.title('k=1',fontsize=18)
plt.legend(loc='upper right',fontsize=18)
plt.subplot(2,8,10)
for i in range(0,20):
    plt.plot(x,S1706_sample2[i],'k-', alpha=.1)
plt.plot(x,series_train_S1706_flow[1].inverse_transform(S1706_2),'#666699', label = 'S1706',linewidth=3)
plt.xlabel('hours of the day',fontsize=18)
plt.ylabel('km/h',labelpad=0,fontsize=18)
plt.ylim((0,200))
plt.title('k=1',fontsize=18)
plt.legend(loc='upper right',fontsize=18)
plt.subplot(2,8,11)
for i in range(0,20):
    plt.plot(x,S56_sample2[i],'k-', alpha=.1)
plt.plot(x,series_train_S56_flow[1].inverse_transform(S56_2),'#666699', label = 'S56',linewidth=3)
plt.xlabel('hours of the day',fontsize=18)
plt.ylabel('km/h',labelpad=0,fontsize=18)
plt.ylim((0,200))
plt.title('k=1',fontsize=18)
plt.legend(loc='upper right',fontsize=18)
plt.subplot(2,8,12)
for i in range(0,20):
    plt.plot(x,S57_sample2[i],'k-', alpha=.1)
plt.plot(x,series_train_S57_flow[1].inverse_transform(S57_2),'#666699', label = 'S57',linewidth=3)
plt.xlabel('hours of the day',fontsize=18)
plt.ylabel('veh/h',labelpad=0,fontsize=18)
plt.ylim((0,200))
plt.title('k=1',fontsize=18)
plt.legend(loc='upper right',fontsize=18)
plt.subplot(2,8,13)
for i in range(0,20):
    plt.plot(x,S1707_sample2[i],'k-', alpha=.1)
plt.plot(x,series_train_S1707_flow[1].inverse_transform(S1707_2),'#666699', label = 'S1707',linewidth=3)
plt.xlabel('hours of the day',fontsize=18)
plt.ylabel('km/h',labelpad=0,fontsize=18)
plt.ylim((0,200))
plt.title('k=1',fontsize=18)
plt.legend(loc='upper right',fontsize=18)
plt.subplot(2,8,14)
for i in range(0,20):
    plt.plot(x,S59_sample2[i],'k-', alpha=.1)
plt.plot(x,series_train_S59_flow[1].inverse_transform(S59_2),'#666699', label = 'S59',linewidth=3)
plt.xlabel('hours of the day',fontsize=18)
plt.ylabel('veh/h',labelpad=0,fontsize=18)
plt.ylim((0,200))
plt.title('k=1',fontsize=18)
plt.legend(loc='upper right',fontsize=18)
plt.subplot(2,8,15)
for i in range(0,20):
    plt.plot(x,S60_sample2[i],'k-', alpha=.1)
plt.plot(x,series_train_S60_flow[1].inverse_transform(S60_2),'#666699', label = 'S60',linewidth=3)
plt.xlabel('hours of the day',fontsize=18)
plt.ylabel('veh/h',labelpad=0,fontsize=18)
plt.ylim((0,200))
plt.title('k=1',fontsize=18)
plt.legend(loc='upper right',fontsize=18)
plt.subplot(2,8,16)
for i in range(0,20):
    plt.plot(x,S61_sample2[i],'k-', alpha=.1)
plt.plot(x,series_train_S61_flow[1].inverse_transform(S61_2),'#666699', label = 'S61',linewidth=3)
plt.xlabel('hours of the day',fontsize=18)
plt.ylabel('veh/h',labelpad=0,fontsize=18)
plt.ylim((0,200))
plt.title('k=1',fontsize=18)
plt.legend(loc='upper right',fontsize=18)
plt.show()
plt.subplot(4,8,17)
for i in range(0,20):
    plt.plot(x,S54_sample1[i],'k-', alpha=.1)
plt.plot(x,series_train_S54_flow[1].inverse_transform(S54_1),'#ff0066', label = 'S54',linewidth=3)
plt.xlabel('hours of the day')
plt.ylabel('veh/h',labelpad=0)
plt.ylim((0,9800))
plt.title('k=2')
plt.legend(loc='upper right')
plt.subplot(4,8,18)
for i in range(0,20):
    plt.plot(x,S1706_sample1[i],'k-', alpha=.1)
plt.plot(x,series_train_S1706_flow[1].inverse_transform(S1706_1),'#ff0066', label = 'S1706',linewidth=3)
plt.xlabel('hours of the day')
plt.ylabel('veh/h',labelpad=0)
plt.ylim((0,9800))
plt.title('k=2')
plt.legend(loc='upper right')
plt.subplot(4,8,19)
for i in range(0,20):
    plt.plot(x,S56_sample1[i],'k-', alpha=.1)
plt.plot(x,series_train_S56_flow[1].inverse_transform(S56_1),'#ff0066', label = 'S56',linewidth=3)
plt.xlabel('hours of the day')
plt.ylabel('veh/h',labelpad=0)
plt.ylim((0,9800))
plt.title('k=2')
plt.legend(loc='upper right')
plt.subplot(4,8,20)
for i in range(0,20):
    plt.plot(x,S57_sample1[i],'k-', alpha=.1)
plt.plot(x,series_train_S57_flow[1].inverse_transform(S57_1),'#ff0066', label = 'S57',linewidth=3)
plt.xlabel('hours of the day')
plt.ylabel('veh/h',labelpad=0)
plt.ylim((0,9800))
plt.title('k=2')
plt.legend(loc='upper right')
plt.subplot(4,8,21)
for i in range(0,20):
    plt.plot(x,S1707_sample1[i],'k-', alpha=.1)
plt.plot(x,series_train_S1707_flow[1].inverse_transform(S1707_1),'#ff0066', label = 'S1707',linewidth=3)
plt.xlabel('hours of the day')
plt.ylabel('veh/h',labelpad=0)
plt.ylim((0,9800))
plt.title('k=2')
plt.legend(loc='upper right')
plt.subplot(4,8,22)
for i in range(0,20):
    plt.plot(x,S59_sample1[i],'k-', alpha=.1)
plt.plot(x,series_train_S59_flow[1].inverse_transform(S59_1),'#ff0066', label = 'S59',linewidth=3)
plt.xlabel('hours of the day')
plt.ylabel('veh/h',labelpad=0)
plt.ylim((0,9800))
plt.title('k=2')
plt.legend(loc='upper right')
plt.subplot(4,8,23)
for i in range(0,20):
    plt.plot(x,S60_sample1[i],'k-', alpha=.1)
plt.plot(x,series_train_S60_flow[1].inverse_transform(S60_1),'#ff0066', label = 'S60',linewidth=3)
plt.xlabel('hours of the day')
plt.ylabel('veh/h',labelpad=0)
plt.ylim((0,9800))
plt.title('k=2')
plt.legend(loc='upper right')
plt.subplot(4,8,24)
for i in range(0,20):
    plt.plot(x,S61_sample1[i],'k-', alpha=.1)
plt.plot(x,series_train_S61_flow[1].inverse_transform(S61_1),'#ff0066', label = 'S61',linewidth=3)
plt.xlabel('hours of the day')
plt.ylabel('veh/h',labelpad=0)
plt.ylim((0,9800))
plt.title('k=2')
plt.legend(loc='upper right')
plt.subplot(4,8,25)
for i in range(0,20):
    plt.plot(x,S54_sample4[i],'k-', alpha=.1)
plt.plot(x,series_train_S54_flow[1].inverse_transform(S54_4),'#476b6b', label = 'S54',linewidth=3)
plt.xlabel('hours of the day')
plt.ylabel('veh/h',labelpad=0)
plt.ylim((0,9800))
plt.title('k=3')
plt.legend(loc='upper right')
plt.subplot(4,8,26)
for i in range(0,20):
    plt.plot(x,S1706_sample4[i],'k-', alpha=.1)
plt.plot(x,series_train_S1706_flow[1].inverse_transform(S1706_4),'#476b6b', label = 'S1706',linewidth=3)
plt.xlabel('hours of the day')
plt.ylabel('veh/h',labelpad=0)
plt.ylim((0,9800))
plt.title('k=3')
plt.legend(loc='upper right')
plt.subplot(4,8,27)
for i in range(0,20):
    plt.plot(x,S56_sample4[i],'k-', alpha=.1)
plt.plot(x,series_train_S56_flow[1].inverse_transform(S56_4),'#476b6b', label = 'S56',linewidth=3)
plt.xlabel('hours of the day')
plt.ylabel('veh/h',labelpad=0)
plt.ylim((0,9800))
plt.title('k=3')
plt.legend(loc='upper right')
plt.subplot(4,8,28)
for i in range(0,20):
    plt.plot(x,S57_sample4[i],'k-', alpha=.1)
plt.plot(x,series_train_S57_flow[1].inverse_transform(S57_4),'#476b6b', label = 'S57',linewidth=3)
plt.xlabel('hours of the day')
plt.ylabel('veh/h',labelpad=0)
plt.ylim((0,9800))
plt.title('k=3')
plt.legend(loc='upper right')
plt.subplot(4,8,29)
for i in range(0,20):
    plt.plot(x,S1707_sample4[i],'k-', alpha=.1)
plt.plot(x,series_train_S1707_flow[1].inverse_transform(S1707_4),'#476b6b', label = 'S1707',linewidth=3)
plt.xlabel('hours of the day')
plt.ylabel('veh/h',labelpad=0)
plt.ylim((0,9800))
plt.title('k=3')
plt.legend(loc='upper right')
plt.subplot(4,8,30)
for i in range(0,20):
    plt.plot(x,S59_sample4[i],'k-', alpha=.1)
plt.plot(x,series_train_S59_flow[1].inverse_transform(S59_4),'#476b6b', label = 'S59',linewidth=3)
plt.xlabel('hours of the day')
plt.ylabel('veh/h',labelpad=0)
plt.ylim((0,9800))
plt.title('k=3')
plt.legend(loc='upper right')
plt.subplot(4,8,31)
for i in range(0,20):
    plt.plot(x,S60_sample4[i],'k-', alpha=.1)
plt.plot(x,series_train_S60_flow[1].inverse_transform(S60_4),'#476b6b', label = 'S60',linewidth=3)
plt.xlabel('hours of the day')
plt.ylabel('veh/h',labelpad=0)
plt.ylim((0,9800))
plt.title('k=3')
plt.legend(loc='upper right')
plt.subplot(4,8,32)
for i in range(0,20):
    plt.plot(x,S61_sample4[i],'k-', alpha=.1)
plt.plot(x,series_train_S61_flow[1].inverse_transform(S61_4),'#476b6b', label = 'S61',linewidth=3)
plt.xlabel('hours of the day')
plt.ylabel('veh/h',labelpad=0)
plt.ylim((0,9800))
plt.title('k=3')
plt.legend(loc='upper right')
plt.figtext(0.5,0.30, "January,February and December", ha="center", va="top", fontsize=14, color="r")
plt.figtext(0.5,0.50, "Fridays, Wednesdays and Thursdays in June, July and August", ha="center", va="top", fontsize=14, color="r")
plt.figtext(0.5,0.70, "No working days", ha="center", va="top", fontsize=14, color="r")
plt.figtext(0.5,0.90, "First part of the week from May to November", ha="center", va="top", fontsize=14, color="r")
plt.show()


