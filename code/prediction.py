#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 11:48:13 2021

@author: nronzoni
"""

## create daily time series train 
#S54
#speed
series_train_S54_speed_pred=daily_series_pred(S54_speed[0],180)
series_train_S54_speed_pred.shape
#flow
series_train_S54_flow_pred=daily_series_pred(S54_flow[0],180)
series_train_S54_flow_pred.shape
#density
series_train_S54_density_pred=daily_series_pred(S54_density[0],180)
series_train_S54_density_pred.shape
#S1706
#speed
series_train_S1706_speed_pred=daily_series_pred(S1706_speed[0],180)
series_train_S1706_speed_pred.shape
#flow
series_train_S1706_flow_pred=daily_series_pred(S1706_flow[0],180)
series_train_S1706_flow_pred.shape
#density
series_train_S1706_density_pred=daily_series_pred(S1706_density[0],180)
series_train_S1706_density_pred.shape

#R169 
#flow
series_train_R169_flow_pred=daily_series_pred(R169_flow[0],180)
series_train_R169_flow_pred.shape
#speed
series_train_R169_speed_pred=daily_series_pred(R169_speed[0],180)
series_train_R169_speed_pred.shape
#density
series_train_R169_density_pred=daily_series_pred(R169_density[0],180)
series_train_R169_density_pred.shape

#S56
#speed
series_train_S56_speed_pred=daily_series_pred(S56_speed[0],180)
series_train_S56_speed_pred.shape

#flow
series_train_S56_flow_pred=daily_series_pred(S56_flow[0],180)
series_train_S56_flow_pred.shape

#density
series_train_S56_density_pred=daily_series_pred(S56_density[0],180)
series_train_S56_density_pred.shape

#R129
#flow
series_train_R129_flow_pred=daily_series_pred(R129_flow[0],180)
series_train_R129_flow_pred.shape
#speed
series_train_R129_speed_pred=daily_series_pred(R129_speed[0],180)
series_train_R129_speed_pred.shape
#density
series_train_R129_density_pred=daily_series_pred(R129_density[0],180)
series_train_R129_density_pred.shape

#S57
#speed
series_train_S57_speed_pred=daily_series_pred(S57_speed[0],180)
series_train_S57_speed_pred.shape

#flow
series_train_S57_flow_pred=daily_series_pred(S57_flow[0],180)
series_train_S57_flow_pred.shape

#density
series_train_S57_density_pred=daily_series_pred(S57_density[0],180)
series_train_S57_density_pred.shape

#R170
#flow
series_train_R170_flow_pred=daily_series_pred(R170_flow[0],180)
series_train_R170_flow_pred.shape
#speed
series_train_R170_speed_pred=daily_series_pred(R170_speed[0],180)
series_train_R170_speed_pred.shape
#density
series_train_R170_density_pred=daily_series_pred(R170_density[0],180)
series_train_R170_density_pred.shape

#S1707
#speed
series_train_S1707_speed_pred=daily_series_pred(S1707_speed[0],180)
series_train_S1707_speed_pred.shape

#flow
series_train_S1707_flow_pred=daily_series_pred(S1707_flow[0],180)
series_train_S1707_flow_pred.shape
#density
series_train_S1707_density_pred=daily_series_pred(S1707_density[0],180)
series_train_S1707_density_pred.shape

#S59
#speed
series_train_S59_speed_pred=daily_series_pred(S59_speed[0],180)
series_train_S59_speed_pred.shape

#flow
series_train_S59_flow_pred=daily_series_pred(S59_flow[0],180)
series_train_S59_flow_pred.shape

#density
series_train_S59_density_pred=daily_series_pred(S59_density[0],180)
series_train_S59_density_pred.shape

#R130
#flow
series_train_R130_flow_pred=daily_series_pred(R130_flow[0],180)
series_train_R130_flow_pred.shape
#speed
series_train_R130_speed_pred=daily_series_pred(R130_speed[0],180)
series_train_R130_speed_pred.shape
#density
series_train_R130_density_pred=daily_series_pred(R130_density[0],180)
series_train_R130_density_pred.shape
#R171
#flow
series_train_R171_flow_pred=daily_series_pred(R171_flow[0],180)
series_train_R171_flow_pred.shape
#speed
series_train_R171_speed_pred=daily_series_pred(R171_speed[0],180)
series_train_R171_speed_pred.shape
#density
series_train_R171_density_pred=daily_series_pred(R171_density[0],180)
series_train_R171_density_pred.shape
#S60
#speed
series_train_S60_speed_pred=daily_series_pred(S60_speed[0],180)
series_train_S60_speed_pred.shape
#flow
series_train_S60_flow_pred=daily_series_pred(S60_flow[0],180)
series_train_S60_flow_pred.shape
#density
series_train_S60_density_pred=daily_series_pred(S60_density[0],180)
series_train_S60_density_pred.shape

#S61
#speed
series_train_S61_speed_pred=daily_series_pred(S61_speed[0],180)
series_train_S61_speed_pred.shape

#flow
series_train_S61_flow_pred=daily_series_pred(S61_flow[0],180)
series_train_S61_flow_pred.shape
#density
series_train_S61_density_pred=daily_series_pred(S61_density[0],180)
series_train_S61_density_pred.shape

## create daily time series test 
#S54
#speed
series_test_S54_speed_pred=daily_series_pred(S54_speed[1],180)
series_test_S54_speed_pred.shape
#flow
series_test_S54_flow_pred=daily_series_pred(S54_flow[1],180)
series_test_S54_flow_pred.shape
#density
series_test_S54_density_pred=daily_series_pred(S54_density[1],180)
series_test_S54_density_pred.shape
#S1706
#speed
series_test_S1706_speed_pred=daily_series_pred(S1706_speed[1],180)
series_test_S1706_speed_pred.shape
#flow
series_test_S1706_flow_pred=daily_series_pred(S1706_flow[1],180)
series_test_S1706_flow_pred.shape
#density
series_test_S1706_density_pred=daily_series_pred(S1706_density[1],180)
series_test_S1706_density_pred.shape

#R169 
#flow
series_test_R169_flow_pred=daily_series_pred(R169_flow[1],180)
series_test_R169_flow_pred.shape
#speed
series_test_R169_speed_pred=daily_series_pred(R169_speed[1],180)
series_test_R169_speed_pred.shape
#density
series_test_R169_density_pred=daily_series_pred(R169_density[1],180)
series_test_R169_density_pred.shape

#S56
#speed
series_test_S56_speed_pred=daily_series_pred(S56_speed[1],180)
series_test_S56_speed_pred.shape

#flow
series_test_S56_flow_pred=daily_series_pred(S56_flow[1],180)
series_test_S56_flow_pred.shape

#density
series_test_S56_density_pred=daily_series_pred(S56_density[1],180)
series_test_S56_density_pred.shape

#R129
#flow
series_test_R129_flow_pred=daily_series_pred(R129_flow[1],180)
series_test_R129_flow_pred.shape
#speed
series_test_R129_speed_pred=daily_series_pred(R129_speed[1],180)
series_test_R129_speed_pred.shape
#density
series_test_R129_density_pred=daily_series_pred(R129_density[1],180)
series_test_R129_density_pred.shape

#S57
#speed
series_test_S57_speed_pred=daily_series_pred(S57_speed[1],180)
series_test_S57_speed_pred.shape

#flow
series_test_S57_flow_pred=daily_series_pred(S57_flow[1],180)
series_test_S57_flow_pred.shape

#density
series_test_S57_density_pred=daily_series_pred(S57_density[1],180)
series_test_S57_density_pred.shape

#R170
#flow
series_test_R170_flow_pred=daily_series_pred(R170_flow[1],180)
series_test_R170_flow_pred.shape
#speed
series_test_R170_speed_pred=daily_series_pred(R170_speed[1],180)
series_test_R170_speed_pred.shape
#density
series_test_R170_density_pred=daily_series_pred(R170_density[1],180)
series_test_R170_density_pred.shape

#S1707
#speed
series_test_S1707_speed_pred=daily_series_pred(S1707_speed[1],180)
series_test_S1707_speed_pred.shape

#flow
series_test_S1707_flow_pred=daily_series_pred(S1707_flow[1],180)
series_test_S1707_flow_pred.shape
#density
series_test_S1707_density_pred=daily_series_pred(S1707_density[1],180)
series_test_S1707_density_pred.shape

#S59
#speed
series_test_S59_speed_pred=daily_series_pred(S59_speed[1],180)
series_test_S59_speed_pred.shape

#flow
series_test_S59_flow_pred=daily_series_pred(S59_flow[1],180)
series_test_S59_flow_pred.shape

#density
series_test_S59_density_pred=daily_series_pred(S59_density[1],180)
series_test_S59_density_pred.shape

#R130
#flow
series_test_R130_flow_pred=daily_series_pred(R130_flow[1],180)
series_test_R130_flow_pred.shape
#speed
series_test_R130_speed_pred=daily_series_pred(R130_speed[1],180)
series_test_R130_speed_pred.shape
#density
series_test_R130_density_pred=daily_series_pred(R130_density[1],180)
series_test_R130_density_pred.shape
#R171
#flow
series_test_R171_flow_pred=daily_series_pred(R171_flow[1],180)
series_test_R171_flow_pred.shape
#speed
series_test_R171_speed_pred=daily_series_pred(R171_speed[1],180)
series_test_R171_speed_pred.shape
#density
series_test_R171_density_pred=daily_series_pred(R171_density[1],180)
series_test_R171_density_pred.shape
#S60
#speed
series_test_S60_speed_pred=daily_series_pred(S60_speed[1],180)
series_test_S60_speed_pred.shape
#flow
series_test_S60_flow_pred=daily_series_pred(S60_flow[1],180)
series_test_S60_flow_pred.shape
#density
series_test_S60_density_pred=daily_series_pred(S60_density[1],180)
series_test_S60_density_pred.shape

#S61
#speed
series_test_S61_speed_pred=daily_series_pred(S61_speed[1],180)
series_test_S61_speed_pred.shape

#flow
series_test_S61_flow_pred=daily_series_pred(S61_flow[1],180)
series_test_S61_flow_pred.shape
#density
series_test_S61_density_pred=daily_series_pred(S61_density[1],180)
series_test_S61_density_pred.shape

#multivariate time series train
multivariate_pred=np.dstack((series_train_S54_speed_pred,series_train_S1706_speed_pred,series_train_S56_speed_pred,series_train_S57_speed_pred,series_train_S1707_speed_pred,series_train_S59_speed_pred,series_train_S60_speed_pred,series_train_S61_speed_pred))
multivariate_time_series_train_pred = to_time_series(multivariate_pred)
print(multivariate_time_series_train_pred.shape)

#multivariate time series test
multivariate_test_pred=np.dstack((series_test_S54_speed_pred,series_test_S1706_speed_pred,series_test_S56_speed_pred,series_test_S57_speed_pred,series_test_S1707_speed_pred,series_test_S59_speed_pred,series_test_S60_speed_pred,series_test_S61_speed_pred))
multivariate_time_series_test_pred = to_time_series(multivariate_test_pred)
print(multivariate_time_series_test_pred.shape)

first_day=walk_forward_validation(multivariate_time_series_train_pred,multivariate_time_series_test_pred[0:1,:,:],10,20)
second_day=walk_forward_validation(multivariate_time_series_train_pred,multivariate_time_series_test_pred[0:1,:,:],5,20)
third_day=walk_forward_validation(multivariate_time_series_train_pred,multivariate_time_series_test_pred[3:4,:,:],5,110)

third_day[0]

third_day[1]

third_day[3]

third_day[5]


second_day[5]

import matplotlib.pyplot as plt
fig = plt.gcf()

x= np.arange(16,19,0.1)

len(x)

plt.plot(x,third_day[3],'r-',label='prediction')
plt.plot(x,third_day[5],'b-',label='ground truth')
plt.ylabel(ylabel='km/h')
plt.xlabel(xlabel='hours of the day')
plt.title(label='13/2/2014 S56 window size of 30 minutes, 3 clusters')
plt.legend()
plt.show()

fourth_day=loubes(multivariate_time_series_train_pred,multivariate_time_series_test_pred[3:4,:,:],5,110)
plt.plot(x,fourth_day[0],'r-',label='prediction')
plt.plot(x,fourth_day[1],'b-',label='ground truth')
plt.ylabel(ylabel='km/h')
plt.xlabel(xlabel='hours of the day')
plt.title(label='13/2/2014 S54 window size of 30 minutes, 3 clusters')
plt.legend()
plt.show()
