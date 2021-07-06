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


#multivariate time series train
multivariate=np.dstack((series_train_S54_flow[0],series_train_S1706_flow[0],series_train_R169_flow[0],series_train_S56_flow[0],series_train_R129_flow[0],series_train_S57_flow[0],series_train_R170_flow[0],series_train_S1707_flow[0],series_train_S59_flow[0],series_train_R130_flow[0],series_train_R171_flow[0],series_train_S60_flow[0],series_train_S61_flow[0]))
multivariate_time_series_train = to_time_series(multivariate)
print(multivariate_time_series_train.shape)

#multivariate time series test
multivariate_test=np.dstack((series_test_S54_flow[0],series_test_S1706_flow[0],series_test_R169_flow[0],series_test_S56_flow[0],series_test_R129_flow[0],series_test_S57_flow[0],series_test_R170_flow[0],series_test_S1707_flow[0],series_test_S59_flow[0],series_test_R130_flow[0],series_test_R171_flow[0],series_test_S60_flow[0],series_test_S61_flow[0]))
multivariate_time_series_test = to_time_series(multivariate_test)
print(multivariate_time_series_test.shape)

#CLUSTERING

from tslearn.clustering import TimeSeriesKMeans, KernelKMeans, silhouette_score
from tslearn.metrics import gamma_soft_dtw

score_g, df = optimalK(multivariate_time_series_train, nrefs=5, maxClusters=7)

plt.plot(df['clusterCount'], df['gap'], linestyle='--', marker='o', color='b');
plt.xlabel('K');
plt.ylabel('Gap Statistic');
plt.title('Gap Statistic vs. number of cluster, test set');

#estimate the gamma hyperparameter 
gamma_soft_dtw(dataset=multivariate_time_series_train, n_samples=200,random_state=0) 

#fit the model on train data 
km_dba = TimeSeriesKMeans(n_clusters=4, metric="softdtw",metric_params={"gamma":gamma_soft_dtw(dataset=multivariate_time_series_train, n_samples=200,random_state=0) }, max_iter=5,max_iter_barycenter=5, random_state=0).fit(multivariate_time_series_train)

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
calplot.calplot(events_train,yearlabel_kws={'color': 'black'}, cmap='Accent', suptitle='TrafficData_I35W_2013 (train): Scenario 11, $\gamma$=86', linewidth=2.3,dropzero=True,vmin=0) 


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
calplot.calplot(events_test,yearlabel_kws={'color': 'black'}, cmap='Accent', suptitle='TrafficData_I35W_2014 (test): Scenario 11, $\gamma$=86', linewidth=2.3,dropzero=True,vmin=0) 


#centroids 
centroids=km_dba.cluster_centers_

centroids.shape

#dataframe to select days which belogns to the closest cluster 
len(index_train)
columns=['days','k']
index=np.arange(357)
len(index)
dataframe_train=pd.DataFrame(columns=columns,index=index)
dataframe_train['days']=index_train
dataframe_train['k']=prediction_train
dataframe_train
#if k=0 
days_cluster=dataframe_train[dataframe_train['k']==0].index
len(days_cluster)
#multivariate time series train speed
#check if multivariate_speed already exist
multivariate_speed=np.dstack((series_train_S54_speed[0],series_train_S1706_speed[0],series_train_R169_speed[0],series_train_S56_speed[0],series_train_R129_speed[0],series_train_S57_speed[0],series_train_R170_speed[0],series_train_S1707_speed[0],series_train_S59_speed[0],series_train_R130_speed[0],series_train_R171_speed[0],series_train_S60_speed[0],series_train_S61_speed[0]))
multivariate_time_series_train_speed = to_time_series(multivariate_speed)
print(multivariate_time_series_train_speed.shape)

#multivariate time series test speed
multivariate_test_speed=np.dstack((series_test_S54_speed[0],series_test_S1706_speed[0],series_test_R169_speed[0],series_test_S56_speed[0],series_test_R129_speed[0],series_test_S57_speed[0],series_test_R170_speed[0],series_test_S1707_speed[0],series_test_S59_speed[0],series_test_R130_speed[0],series_test_R171_speed[0],series_test_S60_speed[0],series_test_S61_speed[0]))
multivariate_time_series_test_speed = to_time_series(multivariate_test_speed)
print(multivariate_time_series_test_speed.shape)

pd.set_option('display.max_seq_items', 200)
print(days_cluster)
multivariate_time_series_train_speed_subset=multivariate_time_series_train_speed[(10,  17,  24,  37,  38,  43,  51,  57,  58,  65,  66,  70,  71,
             72,  73,  78,  79,  85,  86,  92,  93, 101, 105, 106, 113, 114,
            119, 121, 122, 125, 126, 127, 128, 132, 133, 134, 135, 140, 141,
            142, 143, 147, 148, 149, 153, 154, 155, 156, 160, 161, 162, 167,
            168, 171, 177, 178, 184, 185, 191, 198, 199, 200, 205, 206, 212,
            213, 219, 220, 226, 227, 228, 232, 233, 240, 241, 244, 245, 246,
            251, 252, 253, 257, 258, 259, 265, 266, 267, 271, 272, 273, 274,
            279, 280, 281, 286, 287, 288, 292, 293, 294, 295, 301, 302, 307,
            308, 309, 314, 315, 321, 322, 331, 337, 338, 343, 344),:,:]

multivariate_time_series_train_speed_subset.shape
#day nearest to the cluster centroid 
closest(multivariate_time_series_train,prediction_train,centroids,3,events_train)



