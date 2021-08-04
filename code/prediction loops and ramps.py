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
multivariate_pred=np.dstack((series_train_S54_flow_pred,series_train_S1706_flow_pred,series_train_R169_flow_pred,series_train_S56_flow_pred,series_train_R129_flow_pred,series_train_S57_flow_pred,series_train_R170_flow_pred,series_train_S1707_flow_pred,series_train_S59_flow_pred,series_train_R130_flow_pred,series_train_R171_flow_pred,series_train_S60_flow_pred,series_train_S61_flow_pred))
multivariate_time_series_train_pred = to_time_series(multivariate_pred)
print(multivariate_time_series_train_pred.shape)

#multivariate time series test
multivariate_test_pred=np.dstack((series_test_S54_flow_pred,series_test_S1706_flow_pred,series_test_R169_flow_pred,series_test_S56_flow_pred,series_test_R129_flow_pred,series_test_S57_flow_pred,series_test_R170_flow_pred,series_test_S1707_flow_pred,series_test_S59_flow_pred,series_test_R130_flow_pred,series_test_R171_flow_pred,series_test_S60_flow_pred,series_test_S61_flow_pred))
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

first_day=SVR_pred(multivariate_time_series_train,multivariate_time_series_test[0:1,:,:],30,10,10)
#cluster
first_day[0]
#prediction
first_day[1]
#ground_truth
first_day[2]

#rescale with respect to the loop analyzed
Y_pred=series_test_S54_flow[1].inverse_transform(first_day[1])
Y_test=series_test_S54_flow[1].inverse_transform(first_day[2])
#plot
x=index_third_period=pd.date_range('2014-02-10 08:00:00',periods=10, freq='6min')
len(x)
x=x.strftime("%H:%M")
plt.plot(x,np.concatenate(Y_pred,axis=0),'r-',label='prediction')
plt.plot(x,np.concatenate(Y_test,axis=0),'b-',label='ground truth')
plt.xticks(rotation=30,size=8)
plt.ylabel(ylabel='veh/h')
plt.xlabel(xlabel='hours of the day')
plt.title(label='10/02/2014 S54 loops, window size of 1 hour')
plt.legend()
plt.show()

first_day=SVR_pred(multivariate_time_series_train,multivariate_time_series_test[0:1,:,:],30,5,10)
#cluster
first_day[0]
#prediction
first_day[1]
#ground_truth
first_day[2]

#rescale with respect to the loop analyzed
Y_pred=series_test_S54_flow[1].inverse_transform(first_day[1])
Y_test=series_test_S54_flow[1].inverse_transform(first_day[2])
#plot
x=index_third_period=pd.date_range('2014-02-10 08:00:00',periods=10, freq='6min')
len(x)
x=x.strftime("%H:%M")
plt.plot(x,np.concatenate(Y_pred,axis=0),'r-',label='prediction')
plt.plot(x,np.concatenate(Y_test,axis=0),'b-',label='ground truth')
plt.xticks(rotation=30,size=8)
plt.ylabel(ylabel='veh/h')
plt.xlabel(xlabel='hours of the day')
plt.title(label='10/02/2014 S54 loops, window size of 30 minutes')
plt.legend()
plt.show()

first_day=SVR_pred(multivariate_time_series_train,multivariate_time_series_test[0:1,:,:],30,15,10)
#cluster
first_day[0]
#prediction
first_day[1]
#ground_truth
first_day[2]

#rescale with respect to the loop analyzed
Y_pred=series_test_S54_flow[1].inverse_transform(first_day[1])
Y_test=series_test_S54_flow[1].inverse_transform(first_day[2])
#plot
x=index_third_period=pd.date_range('2014-02-10 08:00:00',periods=10, freq='6min')
len(x)
x=x.strftime("%H:%M")
plt.plot(x,np.concatenate(Y_pred,axis=0),'r-',label='prediction')
plt.plot(x,np.concatenate(Y_test,axis=0),'b-',label='ground truth')
plt.xticks(rotation=30,size=8)
plt.ylabel(ylabel='veh/h')
plt.xlabel(xlabel='hours of the day')
plt.title(label='10/02/2014 S54 loops, window size of 1 hour and 30 minutes')
plt.legend()
plt.show()

first_day=SVR_pred(multivariate_time_series_train,multivariate_time_series_test[0:1,:,:],115,10,10)
#cluster
first_day[0]
#prediction
first_day[1]
#ground_truth
first_day[2]

#rescale with respect to the loop analyzed
Y_pred=series_test_S54_flow[1].inverse_transform(first_day[1])
Y_test=series_test_S54_flow[1].inverse_transform(first_day[2])
#plot
x=index_third_period=pd.date_range('2014-02-10 16:30:00',periods=10, freq='6min')
len(x)
x=x.strftime("%H:%M")
plt.plot(x,np.concatenate(Y_pred,axis=0),'r-',label='prediction')
plt.plot(x,np.concatenate(Y_test,axis=0),'b-',label='ground truth')
plt.xticks(rotation=30,size=8)
plt.ylabel(ylabel='veh/h')
plt.xlabel(xlabel='hours of the day')
plt.title(label='10/02/2014 S54 loops, window size of 1 hour')
plt.legend()
plt.show()

first_day=SVR_pred(multivariate_time_series_train,multivariate_time_series_test[0:1,:,:],115,10,10)
#cluster
first_day[0]
#prediction
first_day[1]
#ground_truth
first_day[2]

#rescale with respect to the loop analyzed
Y_pred=series_test_S54_flow[1].inverse_transform(first_day[1])
Y_test=series_test_S54_flow[1].inverse_transform(first_day[2])
#plot
x=index_third_period=pd.date_range('2014-02-10 16:30:00',periods=10, freq='6min')
len(x)
x=x.strftime("%H:%M")
plt.plot(x,np.concatenate(Y_pred,axis=0),'r-',label='prediction')
plt.plot(x,np.concatenate(Y_test,axis=0),'b-',label='ground truth')
plt.xticks(rotation=30,size=8)
plt.ylabel(ylabel='veh/h')
plt.xlabel(xlabel='hours of the day')
plt.title(label='10/02/2014 S54 loops, window size of 1 hour')
plt.legend()
plt.show()

first_day=SVR_pred(multivariate_time_series_train,multivariate_time_series_test[0:1,:,:],30,10,10,8)
#cluster
first_day[0]
#prediction
first_day[1]
#ground_truth
first_day[2]

#rescale with respect to the loop analyzed
Y_pred=series_test_S1707_flow[1].inverse_transform(first_day[1])
Y_test=series_test_S1707_flow[1].inverse_transform(first_day[2])
#plot
x=index_third_period=pd.date_range('2014-02-10 08:00:00',periods=10, freq='6min')
len(x)
x=x.strftime("%H:%M")
plt.plot(x,np.concatenate(Y_pred,axis=0),'r-',label='prediction')
plt.plot(x,np.concatenate(Y_test,axis=0),'b-',label='ground truth')
plt.xticks(rotation=30,size=8)
plt.ylabel(ylabel='veh/h')
plt.xlabel(xlabel='hours of the day')
plt.title(label='10/02/2014 S1707, window size of 1 hour ')
plt.legend()
plt.show()


first_day=SVR_pred_d(multivariate_time_series_train,multivariate_time_series_test[1:2,:,:],30,10,10,12)
#rescale with respect to the loop analyzed
Y_pred=series_test_S61_flow[1].inverse_transform(first_day[1])
Y_test=series_test_S61_flow[1].inverse_transform(first_day[2])
#plot
x=index_third_period=pd.date_range('2014-02-10 08:00',periods=10, freq='6min')
len(x)
x=x.strftime("%H:%M")
plt.plot(x,np.concatenate(Y_pred,axis=0),'r-',label='prediction')
plt.plot(x,np.concatenate(Y_test,axis=0),'b-',label='ground truth')
plt.xticks(rotation=30,size=8)
plt.ylabel(ylabel='veh/h')
plt.xlabel(xlabel='hours of the day')
plt.title(label='10/02/2014 S61, window size of 1 hour')
plt.legend()
plt.show()


first_day=classification_pred_same_bis(multivariate_time_series_train,multivariate_time_series_test[23:24,:,:],30,10,5)
#prediction
first_day[1].shape
#ground_truth
first_day[2].shape
columns = ['S54 flow (veh/h)','S54 flow (veh/h) ground truth','S1706 flow (veh/h)','S1706 flow (veh/h) ground truth', 'R169 flow (veh/h)','R169 flow (veh/h) ground truth','S56 flow (veh/h)','S56 flow (veh/h) ground truth','R129 flow (veh/h)','R129 flow (veh/h) ground truth', 'S57 flow (veh/h)','S57 flow (veh/h) ground truth','R170 flow (veh/h)','R170 flow (veh/h) ground truth','S1707 flow (veh/h)','S1707 flow (veh/h) ground truth', 'S59 flow (veh/h)','S59 flow (veh/h) ground truth','R130 flow (veh/h)','R130 flow (veh/h) ground truth','R171 flow (veh/h)','R171 flow (veh/h) ground truth', 'S60 flow (veh/h)','S60 flow (veh/h) ground truth','S61 flow (veh/h)','S61 flow (veh/h) ground truth']
index=pd.date_range("08:00", periods=10, freq="6min")
df_5= pd.DataFrame(index=index.time, columns=columns)
df_5


Y_pred_S54=series_train_S54_flow[1].inverse_transform(first_day[1][:,:,0])
Y_test_S54=series_test_S54_flow[1].inverse_transform(first_day[2][:,0:1])
error_S54=math.sqrt(mean_squared_error(Y_test_S54,Y_pred_S54.reshape(-1,1)))
Y_pred_S1706=series_train_S1706_flow[1].inverse_transform(first_day[1][:,:,1])
Y_test_S1706=series_test_S1706_flow[1].inverse_transform(first_day[2][:,1:2])
error_S1706=math.sqrt(mean_squared_error(Y_test_S1706,Y_pred_S1706.reshape(-1,1)))
Y_pred_R169=series_train_R169_flow[1].inverse_transform(first_day[1][:,:,2])
Y_test_R169=series_test_R169_flow[1].inverse_transform(first_day[2][:,2:3])
error_R169=math.sqrt(mean_squared_error(Y_test_R169,Y_pred_R169.reshape(-1,1)))
Y_pred_S56=series_train_S56_flow[1].inverse_transform(first_day[1][:,:,3])
Y_test_S56=series_test_S56_flow[1].inverse_transform(first_day[2][:,3:4])
error_S56=math.sqrt(mean_squared_error(Y_test_S56,Y_pred_S56.reshape(-1,1)))
Y_pred_R129=series_train_R129_flow[1].inverse_transform(first_day[1][:,:,4])
Y_test_R129=series_test_R129_flow[1].inverse_transform(first_day[2][:,4:5])
error_R129=math.sqrt(mean_squared_error(Y_test_R129,Y_pred_R129.reshape(-1,1)))
Y_pred_S57=series_train_S57_flow[1].inverse_transform(first_day[1][:,:,5])
Y_test_S57=series_test_S57_flow[1].inverse_transform(first_day[2][:,5:6])
error_S57=math.sqrt(mean_squared_error(Y_test_S57,Y_pred_S57.reshape(-1,1)))
Y_pred_R170=series_train_R170_flow[1].inverse_transform(first_day[1][:,:,6])
Y_test_R170=series_test_R170_flow[1].inverse_transform(first_day[2][:,6:7])
error_R170=math.sqrt(mean_squared_error(Y_test_R170,Y_pred_R170.reshape(-1,1)))
Y_pred_S1707=series_train_S1707_flow[1].inverse_transform(first_day[1][:,:,7])
Y_test_S1707=series_test_S1707_flow[1].inverse_transform(first_day[2][:,7:8])
error_S1707=math.sqrt(mean_squared_error(Y_test_S1707,Y_pred_S1707.reshape(-1,1)))
Y_pred_S59=series_train_S59_flow[1].inverse_transform(first_day[1][:,:,8])
Y_test_S59=series_test_S59_flow[1].inverse_transform(first_day[2][:,8:9])
error_S59=math.sqrt(mean_squared_error(Y_test_S59,Y_pred_S59.reshape(-1,1)))
Y_pred_R130=series_train_R130_flow[1].inverse_transform(first_day[1][:,:,9])
Y_test_R130=series_test_R130_flow[1].inverse_transform(first_day[2][:,9:10])
error_R130=math.sqrt(mean_squared_error(Y_test_R130,Y_pred_R130.reshape(-1,1)))
Y_pred_R171=series_train_R171_flow[1].inverse_transform(first_day[1][:,:,10])
Y_test_R171=series_test_R171_flow[1].inverse_transform(first_day[2][:,10:11])
error_R171=math.sqrt(mean_squared_error(Y_test_R171,Y_pred_R171.reshape(-1,1)))
Y_pred_S60=series_train_S60_flow[1].inverse_transform(first_day[1][:,:,11])
Y_test_S60=series_test_S60_flow[1].inverse_transform(first_day[2][:,11:12])
error_S60=math.sqrt(mean_squared_error(Y_test_S60,Y_pred_S60.reshape(-1,1)))
Y_pred_S61=series_train_S61_flow[1].inverse_transform(first_day[1][:,:,12])
Y_test_S61=series_test_S61_flow[1].inverse_transform(first_day[2][:,12:13])
error_S61=math.sqrt(mean_squared_error(Y_test_S61,Y_pred_S61.reshape(-1,1)))

error=mean([error_S54,error_S1706,error_R169,error_S56,error_R129,error_S57,error_R170,error_S1707,error_S59,error_R130,error_R171,error_S60,error_S61])

df_5['S54 flow (veh/h)']=Y_pred_S54.reshape(-1,1)
df_5['S54 flow (veh/h) ground truth']=Y_test_S54
df_5['S1706 flow (veh/h)']=Y_pred_S1706.reshape(-1,1)
df_5['S1706 flow (veh/h) ground truth']=Y_test_S1706
df_5['R169 flow (veh/h)']=Y_pred_R169.reshape(-1,1)
df_5['R169 flow (veh/h) ground truth']=Y_test_R169
df_5['S56 flow (veh/h)']=Y_pred_S56.reshape(-1,1)
df_5['S56 flow (veh/h) ground truth']=Y_test_S56
df_5['R129 flow (veh/h)']=Y_pred_R129.reshape(-1,1)
df_5['R129 flow (veh/h) ground truth']=Y_test_R129
df_5['S57 flow (veh/h)']=Y_pred_S57.reshape(-1,1)
df_5['S57 flow (veh/h) ground truth']=Y_test_S57
df_5['R170 flow (veh/h)']=Y_pred_R170.reshape(-1,1)
df_5['R170 flow (veh/h) ground truth']=Y_test_R170
df_5['S1707 flow (veh/h)']=Y_pred_S1707.reshape(-1,1)
df_5['S1707 flow (veh/h) ground truth']=Y_test_S1707
df_5['S59 flow (veh/h)']=Y_pred_S59.reshape(-1,1)
df_5['S59 flow (veh/h) ground truth']=Y_test_S59
df_5['R130 flow (veh/h)']=Y_pred_R130.reshape(-1,1)
df_5['R130 flow (veh/h) ground truth']=Y_test_R130
df_5['R171 flow (veh/h)']=Y_pred_R171.reshape(-1,1)
df_5['R171 flow (veh/h) ground truth']=Y_test_R171
df_5['S60 flow (veh/h)']=Y_pred_S60.reshape(-1,1)
df_5['S60 flow (veh/h) ground truth']=Y_test_S60
df_5['S61 flow (veh/h)']=Y_pred_S61.reshape(-1,1)
df_5['S61 flow (veh/h) ground truth']=Y_test_S61
df_5




df_0
df_1
df_2
df_3
df_4



# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter('/Users/nronzoni/Desktop/TrafficData Minnesota/Prediction with ramps clustering on the speed/Classification_prediction_flow_30MIN.xlsx', engine='xlsxwriter')

# Write each dataframe to a different worksheet.
df_1.to_excel(writer, sheet_name='10-2-2014 morning')
df_2.to_excel(writer, sheet_name='12-2-2014 morning')
df_3.to_excel(writer, sheet_name='22-3-2014 afternoon')
df_4.to_excel(writer, sheet_name='15-8-2014 afternoon')
df_5.to_excel(writer, sheet_name='10-9-2014 afternoon')

# Close the Pandas Excel writer and output the Excel file.
writer.save()


first_day_S54=SVR_pred_d_speed(multivariate_time_series_train_subset,multivariate_time_series_test[12:13,:,:],115,5,10,0)
first_day_S1706=SVR_pred_d_speed(multivariate_time_series_train_subset,multivariate_time_series_test[12:13,:,:],115,5,10,1)
first_day_R169=SVR_pred_d_speed(multivariate_time_series_train_subset,multivariate_time_series_test[12:13,:,:],115,5,10,2)
first_day_S56=SVR_pred_d_speed(multivariate_time_series_train_subset,multivariate_time_series_test[12:13,:,:],115,5,10,3)
first_day_R129=SVR_pred_d_speed(multivariate_time_series_train_subset,multivariate_time_series_test[12:13,:,:],115,5,10,4)
first_day_S57=SVR_pred_d_speed(multivariate_time_series_train_subset,multivariate_time_series_test[12:13,:,:],115,5,10,5)
first_day_R170=SVR_pred_d_speed(multivariate_time_series_train_subset,multivariate_time_series_test[12:13,:,:],115,5,10,6)
first_day_S1707=SVR_pred_d_speed(multivariate_time_series_train_subset,multivariate_time_series_test[12:13,:,:],115,5,10,7)
first_day_S59=SVR_pred_d_speed(multivariate_time_series_train_subset,multivariate_time_series_test[12:13,:,:],115,5,10,8)
first_day_R130=SVR_pred_d_speed(multivariate_time_series_train_subset,multivariate_time_series_test[12:13,:,:],115,5,10,9)
first_day_R171=SVR_pred_d_speed(multivariate_time_series_train_subset,multivariate_time_series_test[12:13,:,:],115,5,10,10)
first_day_S60=SVR_pred_d_speed(multivariate_time_series_train_subset,multivariate_time_series_test[12:13,:,:],115,5,10,11)
first_day_S61=SVR_pred_d_speed(multivariate_time_series_train_subset,multivariate_time_series_test[12:13,:,:],115,5,10,12)


Y_pred_S54=series_test_S54_flow[1].inverse_transform(first_day_S54[0])
Y_test_S54=series_test_S54_flow[1].inverse_transform(first_day_S54[1])
error_S54=math.sqrt(mean_squared_error(Y_test_S54.reshape(-1,1),Y_pred_S54.reshape(-1,1)))
Y_pred_S1706=series_test_S1706_flow[1].inverse_transform(first_day_S1706[0])
Y_test_S1706=series_test_S1706_flow[1].inverse_transform(first_day_S1706[1])
error_S1706=math.sqrt(mean_squared_error(Y_test_S1706.reshape(-1,1),Y_pred_S1706.reshape(-1,1)))
Y_pred_R169=series_test_R169_flow[1].inverse_transform(first_day_R169[0])
Y_test_R169=series_test_R169_flow[1].inverse_transform(first_day_R169[1])
error_R169=math.sqrt(mean_squared_error(Y_test_R169.reshape(-1,1),Y_pred_R169.reshape(-1,1)))
Y_pred_S56=series_test_S56_flow[1].inverse_transform(first_day_S56[0])
Y_test_S56=series_test_S56_flow[1].inverse_transform(first_day_S56[1])
error_S56=math.sqrt(mean_squared_error(Y_test_S56.reshape(-1,1),Y_pred_S56.reshape(-1,1)))
Y_pred_R129=series_test_R129_flow[1].inverse_transform(first_day_R129[0])
Y_test_R129=series_test_R129_flow[1].inverse_transform(first_day_R129[1])
error_R129=math.sqrt(mean_squared_error(Y_test_R129.reshape(-1,1),Y_pred_R129.reshape(-1,1)))
Y_pred_S57=series_test_S57_flow[1].inverse_transform(first_day_S57[0])
Y_test_S57=series_test_S57_flow[1].inverse_transform(first_day_S57[1])
error_S57=math.sqrt(mean_squared_error(Y_test_S57.reshape(-1,1),Y_pred_S57.reshape(-1,1)))
Y_pred_R170=series_test_R170_flow[1].inverse_transform(first_day_R170[0])
Y_test_R170=series_test_R170_flow[1].inverse_transform(first_day_R170[1])
error_R170=math.sqrt(mean_squared_error(Y_test_R170.reshape(-1,1),Y_pred_R170.reshape(-1,1)))
Y_pred_S1707=series_test_S1707_flow[1].inverse_transform(first_day_S1707[0])
Y_test_S1707=series_test_S1707_flow[1].inverse_transform(first_day_S1707[1])
error_S1707=math.sqrt(mean_squared_error(Y_test_S1707.reshape(-1,1),Y_pred_S1707.reshape(-1,1)))
Y_pred_S59=series_test_S59_flow[1].inverse_transform(first_day_S59[0])
Y_test_S59=series_test_S59_flow[1].inverse_transform(first_day_S59[1])
error_S59=math.sqrt(mean_squared_error(Y_test_S59.reshape(-1,1),Y_pred_S59.reshape(-1,1)))
Y_pred_R130=series_test_R130_flow[1].inverse_transform(first_day_R130[0])
Y_test_R130=series_test_R130_flow[1].inverse_transform(first_day_R130[1])
error_R130=math.sqrt(mean_squared_error(Y_test_R130.reshape(-1,1),Y_pred_R130.reshape(-1,1)))
Y_pred_R171=series_test_R171_flow[1].inverse_transform(first_day_R171[0])
Y_test_R171=series_test_R171_flow[1].inverse_transform(first_day_R171[1])
error_R171=math.sqrt(mean_squared_error(Y_test_R171.reshape(-1,1),Y_pred_R171.reshape(-1,1)))
Y_pred_S60=series_test_S60_flow[1].inverse_transform(first_day_S60[0])
Y_test_S60=series_test_S60_flow[1].inverse_transform(first_day_S60[1])
error_S60=math.sqrt(mean_squared_error(Y_test_S60.reshape(-1,1),Y_pred_S60.reshape(-1,1)))
Y_pred_S61=series_test_S61_flow[1].inverse_transform(first_day_S61[0])
Y_test_S61=series_test_S61_flow[1].inverse_transform(first_day_S61[1])
error_S61=math.sqrt(mean_squared_error(Y_test_S61.reshape(-1,1),Y_pred_S61.reshape(-1,1)))

error=mean([error_S54,error_S1706,error_R169,error_S56,error_R129,error_S57,error_R170,error_S1707,error_S59,error_R130,error_R171,error_S60,error_S61])


columns = ['S54 flow (veh/h)','S54 flow (veh/h) ground truth','S1706 flow (veh/h)','S1706 flow (veh/h) ground truth', 'R169 flow (veh/h)','R169 flow (veh/h) ground truth','S56 flow (veh/h)','S56 flow (veh/h) ground truth','R129 flow (veh/h)','R129 flow (veh/h) ground truth', 'S57 flow (veh/h)','S57 flow (veh/h) ground truth','R170 flow (veh/h)','R170 flow (veh/h) ground truth','S1707 flow (veh/h)','S1707 flow (veh/h) ground truth', 'S59 flow (veh/h)','S59 flow (veh/h) ground truth','R130 flow (veh/h)','R130 flow (veh/h) ground truth','R171 flow (veh/h)','R171 flow (veh/h) ground truth', 'S60 flow (veh/h)','S60 flow (veh/h) ground truth','S61 flow (veh/h)','S61 flow (veh/h) ground truth']
index=pd.date_range("16:30", periods=10, freq="6min")
df_3 = pd.DataFrame(index=index.time, columns=columns)
df_3
df_3['S54 flow (veh/h)']=Y_pred_S54.reshape(-1,1)
df_3['S54 flow (veh/h) ground truth']=Y_test_S54.reshape(-1,1)
df_3['S1706 flow (veh/h)']=Y_pred_S1706.reshape(-1,1)
df_3['S1706 flow (veh/h) ground truth']=Y_test_S1706.reshape(-1,1)
df_3['R169 flow (veh/h)']=Y_pred_R169.reshape(-1,1)
df_3['R169 flow (veh/h) ground truth']=Y_test_R169.reshape(-1,1)
df_3['S56 flow (veh/h)']=Y_pred_S56.reshape(-1,1)
df_3['S56 flow (veh/h) ground truth']=Y_test_S56.reshape(-1,1)
df_3['R129 flow (veh/h)']=Y_pred_R129.reshape(-1,1)
df_3['R129 flow (veh/h) ground truth']=Y_test_R129.reshape(-1,1)
df_3['S57 flow (veh/h)']=Y_pred_S57.reshape(-1,1)
df_3['S57 flow (veh/h) ground truth']=Y_test_S57.reshape(-1,1)
df_3['R170 flow (veh/h)']=Y_pred_R170.reshape(-1,1)
df_3['R170 flow (veh/h) ground truth']=Y_test_R170.reshape(-1,1)
df_3['S1707 flow (veh/h)']=Y_pred_S1707.reshape(-1,1)
df_3['S1707 flow (veh/h) ground truth']=Y_test_S1707.reshape(-1,1)
df_3['S59 flow (veh/h)']=Y_pred_S59.reshape(-1,1)
df_3['S59 flow (veh/h) ground truth']=Y_test_S59.reshape(-1,1)
df_3['R130 flow (veh/h)']=Y_pred_R130.reshape(-1,1)
df_3['R130 flow (veh/h) ground truth']=Y_test_R130.reshape(-1,1)
df_3['R171 flow (veh/h)']=Y_pred_R171.reshape(-1,1)
df_3['R171 flow (veh/h) ground truth']=Y_test_R171.reshape(-1,1)
df_3['S60 flow (veh/h)']=Y_pred_S60.reshape(-1,1)
df_3['S60 flow (veh/h) ground truth']=Y_test_S60.reshape(-1,1)
df_3['S61 flow (veh/h)']=Y_pred_S61.reshape(-1,1)
df_3['S61 flow (veh/h) ground truth']=Y_test_S61.reshape(-1,1)
df_3


#10/2
df_1
#12/2
df_11
#22/3
df_12
#15/08
df_13
#10/09
df_14
#

# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter('/Users/nronzoni/Desktop/TrafficData Minnesota/Prediction with ramps clustering on the speed/SupportVectorRegression_prediction_flow_clusteringspeed.xlsx', engine='xlsxwriter')

# Write each dataframe to a different worksheet.
df_1.to_excel(writer, sheet_name='10-2-2014 morning')
df_2.to_excel(writer, sheet_name='12-2-2014 morning')
df_3.to_excel(writer, sheet_name='22-3-2014 afternoon')
df_4.to_excel(writer, sheet_name='15-8-2014 afternoon')
df_5.to_excel(writer, sheet_name='10-9-2014 morning')

# Close the Pandas Excel writer and output the Excel file.
writer.save()






#################################################### speed ##########################
first_day=classification_pred_same_bis(multivariate_time_series_train_speed,multivariate_time_series_test_speed[23:24,:,:],30,10,5)
#cluster
#prediction
first_day[1].shape
#ground_truth
first_day[2].shape
columns = ['S54 speed (km/h)','S54 speed (km/h) ground truth','S1706 speed (km/h)','S1706 speed (km/h) ground truth', 'R169 speed (km/h)','R169 speed (km/h) ground truth','S56 speed (km/h)','S56 speed (km/h) ground truth','R129 speed (km/h)','R129 speed (km/h) ground truth', 'S57 speed (km/h)','S57 speed (km/h) ground truth','R170 speed (km/h)','R170 speed (km/h) ground truth','S1707 speed (km/h)','S1707 speed (km/h) ground truth', 'S59 speed (km/h)','S59 speed (km/h) ground truth','R130 speed (km/h)','R130 speed (km/h) ground truth','R171 speed (km/h)','R171 speed (km/h) ground truth', 'S60 speed (km/h)','S60 speed (km/h) ground truth','S61 speed (km/h)','S61 speed (km/h) ground truth']
index=pd.date_range("08:00", periods=10, freq="6min")
df_5= pd.DataFrame(index=index.time, columns=columns)
df_5

Y_pred_S54=series_train_S54_speed[1].inverse_transform(first_day[1][:,:,0])
Y_test_S54=series_test_S54_speed[1].inverse_transform(first_day[2][:,0:1])
error_S54=math.sqrt(mean_squared_error(Y_test_S54,Y_pred_S54.reshape(-1,1)))
Y_pred_S1706=series_train_S1706_speed[1].inverse_transform(first_day[1][:,:,1])
Y_test_S1706=series_test_S1706_speed[1].inverse_transform(first_day[2][:,1:2])
error_S1706=math.sqrt(mean_squared_error(Y_test_S1706,Y_pred_S1706.reshape(-1,1)))
Y_pred_R169=series_train_R169_speed[1].inverse_transform(first_day[1][:,:,2])
Y_test_R169=series_test_R169_speed[1].inverse_transform(first_day[2][:,2:3])
error_R169=math.sqrt(mean_squared_error(Y_test_R169,Y_pred_R169.reshape(-1,1)))
Y_pred_S56=series_train_S56_speed[1].inverse_transform(first_day[1][:,:,3])
Y_test_S56=series_test_S56_speed[1].inverse_transform(first_day[2][:,3:4])
error_S56=math.sqrt(mean_squared_error(Y_test_S56,Y_pred_S56.reshape(-1,1)))
Y_pred_R129=series_train_R129_speed[1].inverse_transform(first_day[1][:,:,4])
Y_test_R129=series_test_R129_speed[1].inverse_transform(first_day[2][:,4:5])
error_R129=math.sqrt(mean_squared_error(Y_test_R129,Y_pred_R129.reshape(-1,1)))
Y_pred_S57=series_train_S57_speed[1].inverse_transform(first_day[1][:,:,5])
Y_test_S57=series_test_S57_speed[1].inverse_transform(first_day[2][:,5:6])
error_S57=math.sqrt(mean_squared_error(Y_test_S57,Y_pred_S57.reshape(-1,1)))
Y_pred_R170=series_train_R170_speed[1].inverse_transform(first_day[1][:,:,6])
Y_test_R170=series_test_R170_speed[1].inverse_transform(first_day[2][:,6:7])
error_R170=math.sqrt(mean_squared_error(Y_test_R170,Y_pred_R170.reshape(-1,1)))
Y_pred_S1707=series_train_S1707_speed[1].inverse_transform(first_day[1][:,:,7])
Y_test_S1707=series_test_S1707_speed[1].inverse_transform(first_day[2][:,7:8])
error_S1707=math.sqrt(mean_squared_error(Y_test_S1707,Y_pred_S1707.reshape(-1,1)))
Y_pred_S59=series_train_S59_speed[1].inverse_transform(first_day[1][:,:,8])
Y_test_S59=series_test_S59_speed[1].inverse_transform(first_day[2][:,8:9])
error_S59=math.sqrt(mean_squared_error(Y_test_S59,Y_pred_S59.reshape(-1,1)))
Y_pred_R130=series_train_R130_speed[1].inverse_transform(first_day[1][:,:,9])
Y_test_R130=series_test_R130_speed[1].inverse_transform(first_day[2][:,9:10])
error_R130=math.sqrt(mean_squared_error(Y_test_R130,Y_pred_R130.reshape(-1,1)))
Y_pred_R171=series_train_R171_speed[1].inverse_transform(first_day[1][:,:,10])
Y_test_R171=series_test_R171_speed[1].inverse_transform(first_day[2][:,10:11])
error_R171=math.sqrt(mean_squared_error(Y_test_R171,Y_pred_R171.reshape(-1,1)))
Y_pred_S60=series_train_S60_speed[1].inverse_transform(first_day[1][:,:,11])
Y_test_S60=series_test_S60_speed[1].inverse_transform(first_day[2][:,11:12])
error_S60=math.sqrt(mean_squared_error(Y_test_S60,Y_pred_S60.reshape(-1,1)))
Y_pred_S61=series_train_S61_speed[1].inverse_transform(first_day[1][:,:,12])
Y_test_S61=series_test_S61_speed[1].inverse_transform(first_day[2][:,12:13])
error_S61=math.sqrt(mean_squared_error(Y_test_S61,Y_pred_S61.reshape(-1,1)))

error=mean([error_S54,error_S1706,error_R169,error_S56,error_R129,error_S57,error_R170,error_S1707,error_S59,error_R130,error_R171,error_S60,error_S61])




df_5['S54 speed (km/h)']=Y_pred_S54.reshape(-1,1)
df_5['S54 speed (km/h) ground truth']=Y_test_S54
df_5['S1706 speed (km/h)']=Y_pred_S1706.reshape(-1,1)
df_5['S1706 speed (km/h) ground truth']=Y_test_S1706
df_5['R169 speed (km/h)']=Y_pred_R169.reshape(-1,1)
df_5['R169 speed (km/h) ground truth']=Y_test_R169
df_5['S56 speed (km/h)']=Y_pred_S56.reshape(-1,1)
df_5['S56 speed (km/h) ground truth']=Y_test_S56
df_5['R129 speed (km/h)']=Y_pred_R129.reshape(-1,1)
df_5['R129 speed (km/h) ground truth']=Y_test_R129
df_5['S57 speed (km/h)']=Y_pred_S57.reshape(-1,1)
df_5['S57 speed (km/h) ground truth']=Y_test_S57
df_5['R170 speed (km/h)']=Y_pred_R170.reshape(-1,1)
df_5['R170 speed (km/h) ground truth']=Y_test_R170
df_5['S1707 speed (km/h)']=Y_pred_S1707.reshape(-1,1)
df_5['S1707 speed (km/h) ground truth']=Y_test_S1707
df_5['S59 speed (km/h)']=Y_pred_S59.reshape(-1,1)
df_5['S59 speed (km/h) ground truth']=Y_test_S59
df_5['R130 speed (km/h)']=Y_pred_R130.reshape(-1,1)
df_5['R130 speed (km/h) ground truth']=Y_test_R130
df_5['R171 speed (km/h)']=Y_pred_R171.reshape(-1,1)
df_5['R171 speed (km/h) ground truth']=Y_test_R171
df_5['S60 speed (km/h)']=Y_pred_S60.reshape(-1,1)
df_5['S60 speed (km/h) ground truth']=Y_test_S60
df_5['S61 speed (km/h)']=Y_pred_S61.reshape(-1,1)
df_5['S61 speed (km/h) ground truth']=Y_test_S61
df_5



#10/2
df_1
#12/2
df_2
#22/03
df_3
#15/08
df_4
#10/9
df_5



# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter('/Users/nronzoni/Desktop/TrafficData Minnesota/Prediction with ramps clustering on the speed/Classification_prediction_speed_ramps_30MIN.xlsx', engine='xlsxwriter')

# Write each dataframe to a different worksheet.
df_1.to_excel(writer, sheet_name='10-2-2014 morning')
df_2.to_excel(writer, sheet_name='12-2-2014 morning')
df_3.to_excel(writer, sheet_name='22-3-2014 afternoon')
df_4.to_excel(writer, sheet_name='15-8-2014 afternoon')
df_5.to_excel(writer, sheet_name='10-9-2014 morning')

# Close the Pandas Excel writer and output the Excel file.
writer.save()


############################################ SVR prediction speed

first_day_S54=SVR_pred_d_speed(multivariate_time_series_train_speed_subset,multivariate_time_series_test_speed[23:24,:,:],30,5,10,0)
first_day_S1706=SVR_pred_d_speed(multivariate_time_series_train_speed_subset,multivariate_time_series_test_speed[23:24,:,:],30,5,10,1)
first_day_R169=SVR_pred_d_speed(multivariate_time_series_train_speed_subset,multivariate_time_series_test_speed[23:24,:,:],30,5,10,2)
first_day_S56=SVR_pred_d_speed(multivariate_time_series_train_speed_subset,multivariate_time_series_test_speed[23:24,:,:],30,5,10,3)
first_day_R129=SVR_pred_d_speed(multivariate_time_series_train_speed_subset,multivariate_time_series_test_speed[23:24,:,:],30,5,10,4)
first_day_S57=SVR_pred_d_speed(multivariate_time_series_train_speed_subset,multivariate_time_series_test_speed[23:24,:,:],30,5,10,5)
first_day_R170=SVR_pred_d_speed(multivariate_time_series_train_speed_subset,multivariate_time_series_test_speed[23:24,:,:],30,5,10,6)
first_day_S1707=SVR_pred_d_speed(multivariate_time_series_train_speed_subset,multivariate_time_series_test_speed[23:24,:,:],30,5,10,7)
first_day_S59=SVR_pred_d_speed(multivariate_time_series_train_speed_subset,multivariate_time_series_test_speed[23:24,:,:],30,5,10,8)
first_day_R130=SVR_pred_d_speed(multivariate_time_series_train_speed_subset,multivariate_time_series_test_speed[23:24,:,:],30,5,10,9)
first_day_R171=SVR_pred_d_speed(multivariate_time_series_train_speed_subset,multivariate_time_series_test_speed[23:24,:,:],30,5,10,10)
first_day_S60=SVR_pred_d_speed(multivariate_time_series_train_speed_subset,multivariate_time_series_test_speed[23:24,:,:],30,5,10,11)
first_day_S61=SVR_pred_d_speed(multivariate_time_series_train_speed_subset,multivariate_time_series_test_speed[23:24,:,:],30,5,10,12)


Y_pred_S54=series_test_S54_speed[1].inverse_transform(first_day_S54[0])
Y_test_S54=series_test_S54_speed[1].inverse_transform(first_day_S54[1])
error_S54=math.sqrt(mean_squared_error(Y_test_S54.reshape(-1,1),Y_pred_S54.reshape(-1,1)))
Y_pred_S1706=series_test_S1706_speed[1].inverse_transform(first_day_S1706[0])
Y_test_S1706=series_test_S1706_speed[1].inverse_transform(first_day_S1706[1])
error_S1706=math.sqrt(mean_squared_error(Y_test_S1706.reshape(-1,1),Y_pred_S1706.reshape(-1,1)))
Y_pred_R169=series_test_R169_speed[1].inverse_transform(first_day_R169[0])
Y_test_R169=series_test_R169_speed[1].inverse_transform(first_day_R169[1])
error_R169=math.sqrt(mean_squared_error(Y_test_R169.reshape(-1,1),Y_pred_R169.reshape(-1,1)))
Y_pred_S56=series_test_S56_speed[1].inverse_transform(first_day_S56[0])
Y_test_S56=series_test_S56_speed[1].inverse_transform(first_day_S56[1])
error_S56=math.sqrt(mean_squared_error(Y_test_S56.reshape(-1,1),Y_pred_S56.reshape(-1,1)))
Y_pred_R129=series_test_R129_speed[1].inverse_transform(first_day_R129[0])
Y_test_R129=series_test_R129_speed[1].inverse_transform(first_day_R129[1])
error_R129=math.sqrt(mean_squared_error(Y_test_R129.reshape(-1,1),Y_pred_R129.reshape(-1,1)))
Y_pred_S57=series_test_S57_speed[1].inverse_transform(first_day_S57[0])
Y_test_S57=series_test_S57_speed[1].inverse_transform(first_day_S57[1])
error_S57=math.sqrt(mean_squared_error(Y_test_S57.reshape(-1,1),Y_pred_S57.reshape(-1,1)))
Y_pred_R170=series_test_R170_speed[1].inverse_transform(first_day_R170[0])
Y_test_R170=series_test_R170_speed[1].inverse_transform(first_day_R170[1])
error_R170=math.sqrt(mean_squared_error(Y_test_R170.reshape(-1,1),Y_pred_R170.reshape(-1,1)))
Y_pred_S1707=series_test_S1707_speed[1].inverse_transform(first_day_S1707[0])
Y_test_S1707=series_test_S1707_speed[1].inverse_transform(first_day_S1707[1])
error_S1707=math.sqrt(mean_squared_error(Y_test_S1707.reshape(-1,1),Y_pred_S1707.reshape(-1,1)))
Y_pred_S59=series_test_S59_speed[1].inverse_transform(first_day_S59[0])
Y_test_S59=series_test_S59_speed[1].inverse_transform(first_day_S59[1])
error_S59=math.sqrt(mean_squared_error(Y_test_S59.reshape(-1,1),Y_pred_S59.reshape(-1,1)))
Y_pred_R130=series_test_R130_speed[1].inverse_transform(first_day_R130[0])
Y_test_R130=series_test_R130_speed[1].inverse_transform(first_day_R130[1])
error_R130=math.sqrt(mean_squared_error(Y_test_R130.reshape(-1,1),Y_pred_R130.reshape(-1,1)))
Y_pred_R171=series_test_R171_speed[1].inverse_transform(first_day_R171[0])
Y_test_R171=series_test_R171_speed[1].inverse_transform(first_day_R171[1])
error_R171=math.sqrt(mean_squared_error(Y_test_R171.reshape(-1,1),Y_pred_R171.reshape(-1,1)))
Y_pred_S60=series_test_S60_speed[1].inverse_transform(first_day_S60[0])
Y_test_S60=series_test_S60_speed[1].inverse_transform(first_day_S60[1])
error_S60=math.sqrt(mean_squared_error(Y_test_S60.reshape(-1,1),Y_pred_S60.reshape(-1,1)))
Y_pred_S61=series_test_S61_speed[1].inverse_transform(first_day_S61[0])
Y_test_S61=series_test_S61_speed[1].inverse_transform(first_day_S61[1])
error_S61=math.sqrt(mean_squared_error(Y_test_S61.reshape(-1,1),Y_pred_S61.reshape(-1,1)))

error=mean([error_S54,error_S1706,error_R169,error_S56,error_R129,error_S57,error_R170,error_S1707,error_S59,error_R130,error_R171,error_S60,error_S61])


columns = ['S54 speed (km/h)','S54 speed (km/h) ground truth','S1706 speed (km/h)','S1706 speed (km/h) ground truth', 'R169 speed (km/h)','R169 speed (km/h) ground truth','S56 speed (km/h)','S56 speed (km/h) ground truth','R129 speed (km/h)','R129 speed (km/h) ground truth', 'S57 speed (km/h)','S57 speed (km/h) ground truth','R170 speed (km/h)','R170 speed (km/h) ground truth','S1707 speed (km/h)','S1707 speed (km/h) ground truth', 'S59 speed (km/h)','S59 speed (km/h) ground truth','R130 speed (km/h)','R130 speed (km/h) ground truth','R171 speed (km/h)','R171 speed (km/h) ground truth', 'S60 speed (km/h)','S60 speed (km/h) ground truth','S61 speed (km/h)','S61 speed (km/h) ground truth']
index=pd.date_range("08:00", periods=10, freq="6min")
df_5= pd.DataFrame(index=index.time, columns=columns)
df_5
df_5['S54 speed (km/h)']=Y_pred_S54.reshape(-1,1)
df_5['S54 speed (km/h) ground truth']=Y_test_S54.reshape(-1,1)
df_5['S1706 speed (km/h)']=Y_pred_S1706.reshape(-1,1)
df_5['S1706 speed (km/h) ground truth']=Y_test_S1706.reshape(-1,1)
df_5['R169 speed (km/h)']=Y_pred_R169.reshape(-1,1)
df_5['R169 speed (km/h) ground truth']=Y_test_R169.reshape(-1,1)
df_5['S56 speed (km/h)']=Y_pred_S56.reshape(-1,1)
df_5['S56 speed (km/h) ground truth']=Y_test_S56.reshape(-1,1)
df_5['R129 speed (km/h)']=Y_pred_R129.reshape(-1,1)
df_5['R129 speed (km/h) ground truth']=Y_test_R129.reshape(-1,1)
df_5['S57 speed (km/h)']=Y_pred_S57.reshape(-1,1)
df_5['S57 speed (km/h) ground truth']=Y_test_S57.reshape(-1,1)
df_5['R170 speed (km/h)']=Y_pred_R170.reshape(-1,1)
df_5['R170 speed (km/h) ground truth']=Y_test_R170.reshape(-1,1)
df_5['S1707 speed (km/h)']=Y_pred_S1707.reshape(-1,1)
df_5['S1707 speed (km/h) ground truth']=Y_test_S1707.reshape(-1,1)
df_5['S59 speed (km/h)']=Y_pred_S59.reshape(-1,1)
df_5['S59 speed (km/h) ground truth']=Y_test_S59.reshape(-1,1)
df_5['R130 speed (km/h)']=Y_pred_R130.reshape(-1,1)
df_5['R130 speed (km/h) ground truth']=Y_test_R130.reshape(-1,1)
df_5['R171 speed (km/h)']=Y_pred_R171.reshape(-1,1)
df_5['R171 speed (km/h) ground truth']=Y_test_R171.reshape(-1,1)
df_5['S60 speed (km/h)']=Y_pred_S60.reshape(-1,1)
df_5['S60 speed (km/h) ground truth']=Y_test_S60.reshape(-1,1)
df_5['S61 speed (km/h)']=Y_pred_S61.reshape(-1,1)
df_5['S61 speed (km/h) ground truth']=Y_test_S61.reshape(-1,1)
df_5



#10/2
df_15
#10/9
df_16
#12/2
df_17
#22/3
df_18
#15/8
df_19

writer = pd.ExcelWriter('/Users/nronzoni/Desktop/TrafficData Minnesota/Prediction with ramps clustering on the speed/SupportVectorRegression_prediction_speed_clusteringspeed.xlsx', engine='xlsxwriter')

# Write each dataframe to a different worksheet.
df_1.to_excel(writer, sheet_name='10-2-2014 morning')
df_2.to_excel(writer, sheet_name='12-2-2014 morning')
df_3.to_excel(writer, sheet_name='22-3-2014 afternoon')
df_4.to_excel(writer, sheet_name='15-8-2014 afternoon')
df_5.to_excel(writer, sheet_name='10-9-2014 morning')

# Close the Pandas Excel writer and output the Excel file.
writer.save()


############################ walk forward validation FLOW

first_day_S54=walk_forward_validation_bis(multivariate_time_series_train,multivariate_time_series_test[23:24,:,:],5,30,0)
first_day_S1706=walk_forward_validation_bis(multivariate_time_series_train,multivariate_time_series_test[23:24,:,:],5,30,1)
first_day_R169=walk_forward_validation_bis(multivariate_time_series_train,multivariate_time_series_test[23:24,:,:],5,30,2)
first_day_S56=walk_forward_validation_bis(multivariate_time_series_train,multivariate_time_series_test[23:24,:,:],5,30,3)
first_day_R129=walk_forward_validation_bis(multivariate_time_series_train,multivariate_time_series_test[23:24,:,:],5,30,4)
first_day_S57=walk_forward_validation_bis(multivariate_time_series_train,multivariate_time_series_test[23:24,:,:],5,30,5)
first_day_R170=walk_forward_validation_bis(multivariate_time_series_train,multivariate_time_series_test[23:24,:,:],5,30,6)
first_day_S1707=walk_forward_validation_bis(multivariate_time_series_train,multivariate_time_series_test[23:24,:,:],5,30,7)
first_day_S59=walk_forward_validation_bis(multivariate_time_series_train,multivariate_time_series_test[23:24,:,:],5,30,8)
first_day_R130=walk_forward_validation_bis(multivariate_time_series_train,multivariate_time_series_test[23:24,:,:],5,30,9)
first_day_R171=walk_forward_validation_bis(multivariate_time_series_train,multivariate_time_series_test[23:24,:,:],5,30,10)
first_day_S60=walk_forward_validation_bis(multivariate_time_series_train,multivariate_time_series_test[23:24,:,:],5,30,11)
first_day_S61=walk_forward_validation_bis(multivariate_time_series_train,multivariate_time_series_test[23:24,:,:],5,30,12)


Y_pred_S54=series_test_S54_flow[1].inverse_transform(first_day_S54[0].reshape(-1,1))
Y_test_S54=series_test_S54_flow[1].inverse_transform(first_day_S54[1].reshape(-1,1))
error_S54=math.sqrt(mean_squared_error(Y_test_S54.reshape(-1,1),Y_pred_S54.reshape(-1,1)))
Y_pred_S1706=series_test_S1706_flow[1].inverse_transform(first_day_S1706[0].reshape(-1,1))
Y_test_S1706=series_test_S1706_flow[1].inverse_transform(first_day_S1706[1].reshape(-1,1))
error_S1706=math.sqrt(mean_squared_error(Y_test_S1706.reshape(-1,1),Y_pred_S1706.reshape(-1,1)))
Y_pred_R169=series_test_R169_flow[1].inverse_transform(first_day_R169[0].reshape(-1,1))
Y_test_R169=series_test_R169_flow[1].inverse_transform(first_day_R169[1].reshape(-1,1))
error_R169=math.sqrt(mean_squared_error(Y_test_R169.reshape(-1,1),Y_pred_R169.reshape(-1,1)))
Y_pred_S56=series_test_S56_flow[1].inverse_transform(first_day_S56[0].reshape(-1,1))
Y_test_S56=series_test_S56_flow[1].inverse_transform(first_day_S56[1].reshape(-1,1))
error_S56=math.sqrt(mean_squared_error(Y_test_S56.reshape(-1,1),Y_pred_S56.reshape(-1,1)))
Y_pred_R129=series_test_R129_flow[1].inverse_transform(first_day_R129[0].reshape(-1,1))
Y_test_R129=series_test_R129_flow[1].inverse_transform(first_day_R129[1].reshape(-1,1))
error_R129=math.sqrt(mean_squared_error(Y_test_R129.reshape(-1,1),Y_pred_R129.reshape(-1,1)))
Y_pred_S57=series_test_S57_flow[1].inverse_transform(first_day_S57[0].reshape(-1,1))
Y_test_S57=series_test_S57_flow[1].inverse_transform(first_day_S57[1].reshape(-1,1))
error_S57=math.sqrt(mean_squared_error(Y_test_S57.reshape(-1,1),Y_pred_S57.reshape(-1,1)))
Y_pred_R170=series_test_R170_flow[1].inverse_transform(first_day_R170[0].reshape(-1,1))
Y_test_R170=series_test_R170_flow[1].inverse_transform(first_day_R170[1].reshape(-1,1))
error_R170=math.sqrt(mean_squared_error(Y_test_R170.reshape(-1,1),Y_pred_R170.reshape(-1,1)))
Y_pred_S1707=series_test_S1707_flow[1].inverse_transform(first_day_S1707[0].reshape(-1,1))
Y_test_S1707=series_test_S1707_flow[1].inverse_transform(first_day_S1707[1].reshape(-1,1))
error_S1707=math.sqrt(mean_squared_error(Y_test_S1707.reshape(-1,1),Y_pred_S1707.reshape(-1,1)))
Y_pred_S59=series_test_S59_flow[1].inverse_transform(first_day_S59[0].reshape(-1,1))
Y_test_S59=series_test_S59_flow[1].inverse_transform(first_day_S59[1].reshape(-1,1))
error_S59=math.sqrt(mean_squared_error(Y_test_S59.reshape(-1,1),Y_pred_S59.reshape(-1,1)))
Y_pred_R130=series_test_R130_flow[1].inverse_transform(first_day_R130[0].reshape(-1,1))
Y_test_R130=series_test_R130_flow[1].inverse_transform(first_day_R130[1].reshape(-1,1))
error_R130=math.sqrt(mean_squared_error(Y_test_R130.reshape(-1,1),Y_pred_R130.reshape(-1,1)))
Y_pred_R171=series_test_R171_flow[1].inverse_transform(first_day_R171[0].reshape(-1,1))
Y_test_R171=series_test_R171_flow[1].inverse_transform(first_day_R171[1].reshape(-1,1))
error_R171=math.sqrt(mean_squared_error(Y_test_R171.reshape(-1,1),Y_pred_R171.reshape(-1,1)))
Y_pred_S60=series_test_S60_flow[1].inverse_transform(first_day_S60[0].reshape(-1,1))
Y_test_S60=series_test_S60_flow[1].inverse_transform(first_day_S60[1].reshape(-1,1))
error_S60=math.sqrt(mean_squared_error(Y_test_S60.reshape(-1,1),Y_pred_S60.reshape(-1,1)))
Y_pred_S61=series_test_S61_flow[1].inverse_transform(first_day_S61[0].reshape(-1,1))
Y_test_S61=series_test_S61_flow[1].inverse_transform(first_day_S61[1].reshape(-1,1))
error_S61=math.sqrt(mean_squared_error(Y_test_S61.reshape(-1,1),Y_pred_S61.reshape(-1,1)))

error=mean([error_S54,error_S1706,error_R169,error_S56,error_R129,error_S57,error_R170,error_S1707,error_S59,error_R130,error_R171,error_S60,error_S61])


columns = ['S54 flow (veh/h)','S54 flow (veh/h) ground truth','S1706 flow (veh/h)','S1706 flow (veh/h) ground truth', 'R169 flow (veh/h)','R169 flow (veh/h) ground truth','S56 flow (veh/h)','S56 flow (veh/h) ground truth','R129 flow (veh/h)','R129 flow (veh/h) ground truth', 'S57 flow (veh/h)','S57 flow (veh/h) ground truth','R170 flow (veh/h)','R170 flow (veh/h) ground truth','S1707 flow (veh/h)','S1707 flow (veh/h) ground truth', 'S59 flow (veh/h)','S59 flow (veh/h) ground truth','R130 flow (veh/h)','R130 flow (veh/h) ground truth','R171 flow (veh/h)','R171 flow (veh/h) ground truth', 'S60 flow (veh/h)','S60 flow (veh/h) ground truth','S61 flow (veh/h)','S61 flow (veh/h) ground truth']
index=pd.date_range("08:00", periods=10, freq="6min")
df_5 = pd.DataFrame(index=index.time, columns=columns)
df_5
df_5['S54 flow (veh/h)']=Y_pred_S54.reshape(-1,1)
df_5['S54 flow (veh/h) ground truth']=Y_test_S54.reshape(-1,1)
df_5['S1706 flow (veh/h)']=Y_pred_S1706.reshape(-1,1)
df_5['S1706 flow (veh/h) ground truth']=Y_test_S1706.reshape(-1,1)
df_5['R169 flow (veh/h)']=Y_pred_R169.reshape(-1,1)
df_5['R169 flow (veh/h) ground truth']=Y_test_R169.reshape(-1,1)
df_5['S56 flow (veh/h)']=Y_pred_S56.reshape(-1,1)
df_5['S56 flow (veh/h) ground truth']=Y_test_S56.reshape(-1,1)
df_5['R129 flow (veh/h)']=Y_pred_R129.reshape(-1,1)
df_5['R129 flow (veh/h) ground truth']=Y_test_R129.reshape(-1,1)
df_5['S57 flow (veh/h)']=Y_pred_S57.reshape(-1,1)
df_5['S57 flow (veh/h) ground truth']=Y_test_S57.reshape(-1,1)
df_5['R170 flow (veh/h)']=Y_pred_R170.reshape(-1,1)
df_5['R170 flow (veh/h) ground truth']=Y_test_R170.reshape(-1,1)
df_5['S1707 flow (veh/h)']=Y_pred_S1707.reshape(-1,1)
df_5['S1707 flow (veh/h) ground truth']=Y_test_S1707.reshape(-1,1)
df_5['S59 flow (veh/h)']=Y_pred_S59.reshape(-1,1)
df_5['S59 flow (veh/h) ground truth']=Y_test_S59.reshape(-1,1)
df_5['R130 flow (veh/h)']=Y_pred_R130.reshape(-1,1)
df_5['R130 flow (veh/h) ground truth']=Y_test_R130.reshape(-1,1)
df_5['R171 flow (veh/h)']=Y_pred_R171.reshape(-1,1)
df_5['R171 flow (veh/h) ground truth']=Y_test_R171.reshape(-1,1)
df_5['S60 flow (veh/h)']=Y_pred_S60.reshape(-1,1)
df_5['S60 flow (veh/h) ground truth']=Y_test_S60.reshape(-1,1)
df_5['S61 flow (veh/h)']=Y_pred_S61.reshape(-1,1)
df_5['S61 flow (veh/h) ground truth']=Y_test_S61.reshape(-1,1)
df_5




#10/2
df_20
#12/2
df_21
#22/3
df_22
#15/08
df_23
#10/09
df_24
#

# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter('/Users/nronzoni/Desktop/TrafficData Minnesota/Prediction with ramps/WALKFORWARD_SupportVectorRegression_prediction_flow_NC.xlsx', engine='xlsxwriter')

# Write each dataframe to a different worksheet.
df_1.to_excel(writer, sheet_name='10-2-2014 morning')
df_2.to_excel(writer, sheet_name='12-2-2014 morning')
df_3.to_excel(writer, sheet_name='22-3-2014 afternoon')
df_4.to_excel(writer, sheet_name='15-8-2014 afternoon')
df_5.to_excel(writer, sheet_name='10-9-2014 morning')

# Close the Pandas Excel writer and output the Excel file.
writer.save()

##### speed 


first_day_S54=walk_forward_validation_bis(multivariate_time_series_train_speed,multivariate_time_series_test_speed[23:24,:,:],5,30,0)
first_day_S1706=walk_forward_validation_bis(multivariate_time_series_train_speed,multivariate_time_series_test_speed[23:24,:,:],5,30,1)
first_day_R169=walk_forward_validation_bis(multivariate_time_series_train_speed,multivariate_time_series_test_speed[23:24,:,:],5,30,2)
first_day_S56=walk_forward_validation_bis(multivariate_time_series_train_speed,multivariate_time_series_test_speed[23:24,:,:],5,30,3)
first_day_R129=walk_forward_validation_bis(multivariate_time_series_train_speed,multivariate_time_series_test_speed[23:24,:,:],5,30,4)
first_day_S57=walk_forward_validation_bis(multivariate_time_series_train_speed,multivariate_time_series_test_speed[23:24,:,:],5,30,5)
first_day_R170=walk_forward_validation_bis(multivariate_time_series_train_speed,multivariate_time_series_test_speed[23:24,:,:],5,30,6)
first_day_S1707=walk_forward_validation_bis(multivariate_time_series_train_speed,multivariate_time_series_test_speed[23:24,:,:],5,30,7)
first_day_S59=walk_forward_validation_bis(multivariate_time_series_train_speed,multivariate_time_series_test_speed[23:24,:,:],5,30,8)
first_day_R130=walk_forward_validation_bis(multivariate_time_series_train_speed,multivariate_time_series_test_speed[23:24,:,:],5,30,9)
first_day_R171=walk_forward_validation_bis(multivariate_time_series_train_speed,multivariate_time_series_test_speed[23:24,:,:],5,30,10)
first_day_S60=walk_forward_validation_bis(multivariate_time_series_train_speed,multivariate_time_series_test_speed[23:24,:,:],5,30,11)
first_day_S61=walk_forward_validation_bis(multivariate_time_series_train_speed,multivariate_time_series_test_speed[23:24,:,:],5,30,12)


Y_pred_S54=series_test_S54_speed[1].inverse_transform(first_day_S54[0].reshape(-1,1))
Y_test_S54=series_test_S54_speed[1].inverse_transform(first_day_S54[1].reshape(-1,1))
error_S54=math.sqrt(mean_squared_error(Y_test_S54.reshape(-1,1),Y_pred_S54.reshape(-1,1)))
Y_pred_S1706=series_test_S1706_speed[1].inverse_transform(first_day_S1706[0].reshape(-1,1))
Y_test_S1706=series_test_S1706_speed[1].inverse_transform(first_day_S1706[1].reshape(-1,1))
error_S1706=math.sqrt(mean_squared_error(Y_test_S1706.reshape(-1,1),Y_pred_S1706.reshape(-1,1)))
Y_pred_R169=series_test_R169_speed[1].inverse_transform(first_day_R169[0].reshape(-1,1))
Y_test_R169=series_test_R169_speed[1].inverse_transform(first_day_R169[1].reshape(-1,1))
error_R169=math.sqrt(mean_squared_error(Y_test_R169.reshape(-1,1),Y_pred_R169.reshape(-1,1)))
Y_pred_S56=series_test_S56_speed[1].inverse_transform(first_day_S56[0].reshape(-1,1))
Y_test_S56=series_test_S56_speed[1].inverse_transform(first_day_S56[1].reshape(-1,1))
error_S56=math.sqrt(mean_squared_error(Y_test_S56.reshape(-1,1),Y_pred_S56.reshape(-1,1)))
Y_pred_R129=series_test_R129_speed[1].inverse_transform(first_day_R129[0].reshape(-1,1))
Y_test_R129=series_test_R129_speed[1].inverse_transform(first_day_R129[1].reshape(-1,1))
error_R129=math.sqrt(mean_squared_error(Y_test_R129.reshape(-1,1),Y_pred_R129.reshape(-1,1)))
Y_pred_S57=series_test_S57_speed[1].inverse_transform(first_day_S57[0].reshape(-1,1))
Y_test_S57=series_test_S57_speed[1].inverse_transform(first_day_S57[1].reshape(-1,1))
error_S57=math.sqrt(mean_squared_error(Y_test_S57.reshape(-1,1),Y_pred_S57.reshape(-1,1)))
Y_pred_R170=series_test_R170_speed[1].inverse_transform(first_day_R170[0].reshape(-1,1))
Y_test_R170=series_test_R170_speed[1].inverse_transform(first_day_R170[1].reshape(-1,1))
error_R170=math.sqrt(mean_squared_error(Y_test_R170.reshape(-1,1),Y_pred_R170.reshape(-1,1)))
Y_pred_S1707=series_test_S1707_speed[1].inverse_transform(first_day_S1707[0].reshape(-1,1))
Y_test_S1707=series_test_S1707_speed[1].inverse_transform(first_day_S1707[1].reshape(-1,1))
error_S1707=math.sqrt(mean_squared_error(Y_test_S1707.reshape(-1,1),Y_pred_S1707.reshape(-1,1)))
Y_pred_S59=series_test_S59_speed[1].inverse_transform(first_day_S59[0].reshape(-1,1))
Y_test_S59=series_test_S59_speed[1].inverse_transform(first_day_S59[1].reshape(-1,1))
error_S59=math.sqrt(mean_squared_error(Y_test_S59.reshape(-1,1),Y_pred_S59.reshape(-1,1)))
Y_pred_R130=series_test_R130_speed[1].inverse_transform(first_day_R130[0].reshape(-1,1))
Y_test_R130=series_test_R130_speed[1].inverse_transform(first_day_R130[1].reshape(-1,1))
error_R130=math.sqrt(mean_squared_error(Y_test_R130.reshape(-1,1),Y_pred_R130.reshape(-1,1)))
Y_pred_R171=series_test_R171_speed[1].inverse_transform(first_day_R171[0].reshape(-1,1))
Y_test_R171=series_test_R171_speed[1].inverse_transform(first_day_R171[1].reshape(-1,1))
error_R171=math.sqrt(mean_squared_error(Y_test_R171.reshape(-1,1),Y_pred_R171.reshape(-1,1)))
Y_pred_S60=series_test_S60_speed[1].inverse_transform(first_day_S60[0].reshape(-1,1))
Y_test_S60=series_test_S60_speed[1].inverse_transform(first_day_S60[1].reshape(-1,1))
error_S60=math.sqrt(mean_squared_error(Y_test_S60.reshape(-1,1),Y_pred_S60.reshape(-1,1)))
Y_pred_S61=series_test_S61_speed[1].inverse_transform(first_day_S61[0].reshape(-1,1))
Y_test_S61=series_test_S61_speed[1].inverse_transform(first_day_S61[1].reshape(-1,1))
error_S61=math.sqrt(mean_squared_error(Y_test_S61.reshape(-1,1),Y_pred_S61.reshape(-1,1)))

error=mean([error_S54,error_S1706,error_R169,error_S56,error_R129,error_S57,error_R170,error_S1707,error_S59,error_R130,error_R171,error_S60,error_S61])


columns = ['S54 speed (km/h)','S54 speed (km/h) ground truth','S1706 speed (km/h)','S1706 speed (km/h) ground truth', 'R169 speed (km/h)','R169 speed (km/h) ground truth','S56 speed (km/h)','S56 speed (km/h) ground truth','R129 speed (km/h)','R129 speed (km/h) ground truth', 'S57 speed (km/h)','S57 speed (km/h) ground truth','R170 speed (km/h)','R170 speed (km/h) ground truth','S1707 speed (km/h)','S1707 speed (km/h) ground truth', 'S59 speed (km/h)','S59 speed (km/h) ground truth','R130 speed (km/h)','R130 speed (km/h) ground truth','R171 speed (km/h)','R171 speed (km/h) ground truth', 'S60 speed (km/h)','S60 speed (km/h) ground truth','S61 speed (km/h)','S61 speed (km/h) ground truth']
index=pd.date_range("08:00", periods=10, freq="6min")
df_5= pd.DataFrame(index=index.time, columns=columns)
df_5
df_5['S54 speed (km/h)']=Y_pred_S54.reshape(-1,1)
df_5['S54 speed (km/h) ground truth']=Y_test_S54.reshape(-1,1)
df_5['S1706 speed (km/h)']=Y_pred_S1706.reshape(-1,1)
df_5['S1706 speed (km/h) ground truth']=Y_test_S1706.reshape(-1,1)
df_5['R169 speed (km/h)']=Y_pred_R169.reshape(-1,1)
df_5['R169 speed (km/h) ground truth']=Y_test_R169.reshape(-1,1)
df_5['S56 speed (km/h)']=Y_pred_S56.reshape(-1,1)
df_5['S56 speed (km/h) ground truth']=Y_test_S56.reshape(-1,1)
df_5['R129 speed (km/h)']=Y_pred_R129.reshape(-1,1)
df_5['R129 speed (km/h) ground truth']=Y_test_R129.reshape(-1,1)
df_5['S57 speed (km/h)']=Y_pred_S57.reshape(-1,1)
df_5['S57 speed (km/h) ground truth']=Y_test_S57.reshape(-1,1)
df_5['R170 speed (km/h)']=Y_pred_R170.reshape(-1,1)
df_5['R170 speed (km/h) ground truth']=Y_test_R170.reshape(-1,1)
df_5['S1707 speed (km/h)']=Y_pred_S1707.reshape(-1,1)
df_5['S1707 speed (km/h) ground truth']=Y_test_S1707.reshape(-1,1)
df_5['S59 speed (km/h)']=Y_pred_S59.reshape(-1,1)
df_5['S59 speed (km/h) ground truth']=Y_test_S59.reshape(-1,1)
df_5['R130 speed (km/h)']=Y_pred_R130.reshape(-1,1)
df_5['R130 speed (km/h) ground truth']=Y_test_R130.reshape(-1,1)
df_5['R171 speed (km/h)']=Y_pred_R171.reshape(-1,1)
df_5['R171 speed (km/h) ground truth']=Y_test_R171.reshape(-1,1)
df_5['S60 speed (km/h)']=Y_pred_S60.reshape(-1,1)
df_5['S60 speed (km/h) ground truth']=Y_test_S60.reshape(-1,1)
df_5['S61 speed (km/h)']=Y_pred_S61.reshape(-1,1)
df_5['S61 speed (km/h) ground truth']=Y_test_S61.reshape(-1,1)
df_5

#10/2
df_25
#10/9
df_26
#12/2
df_27
#22/3
df_28
#15/8
df_29

writer = pd.ExcelWriter('/Users/nronzoni/Desktop/TrafficData Minnesota/Prediction with ramps/WALKFORWARD_SupportVectorRegression_prediction_speed_NC.xlsx', engine='xlsxwriter')

# Write each dataframe to a different worksheet.
df_1.to_excel(writer, sheet_name='10-2-2014 morning')
df_2.to_excel(writer, sheet_name='12-2-2014 morning')
df_3.to_excel(writer, sheet_name='22-3-2014 afternoon')
df_4.to_excel(writer, sheet_name='15-8-2014 afternoon')
df_5.to_excel(writer, sheet_name='10-9-2014 morning')

# Close the Pandas Excel writer and output the Excel file.
writer.save()


















