#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 11:51:44 2021

@author: nronzoni
"""

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


#Scenario 2
#first cluster k=0
#S54
#flow
S54_flow_centroid=centroids[0][:,0]
S54_flow_centroid_0= series_train_S54_flow[1].inverse_transform(S54_flow_centroid.reshape((len(S54_flow_centroid), 1)))
#S1706
S1706_flow_centroid=centroids[0][:,1]
S1706_flow_centroid_0= series_train_S1706_flow[1].inverse_transform(S1706_flow_centroid.reshape((len(S1706_flow_centroid), 1)))
#R169
R169_flow_centroid=centroids[0][:,2]
R169_flow_centroid_0= series_train_R169_flow[1].inverse_transform(R169_flow_centroid.reshape((len(R169_flow_centroid), 1)))
#S56
S56_flow_centroid=centroids[0][:,3]
S56_flow_centroid_0= series_train_S56_flow[1].inverse_transform(S56_flow_centroid.reshape((len(S56_flow_centroid), 1)))
#R129
R129_flow_centroid=centroids[0][:,4]
R129_flow_centroid_0= series_train_R129_flow[1].inverse_transform(R129_flow_centroid.reshape((len(R129_flow_centroid), 1)))
#S57
S57_flow_centroid=centroids[0][:,5]
S57_flow_centroid_0= series_train_S57_flow[1].inverse_transform(S57_flow_centroid.reshape((len(S57_flow_centroid), 1)))
#R170
R170_flow_centroid=centroids[0][:,6]
R170_flow_centroid_0= series_train_R170_flow[1].inverse_transform(R170_flow_centroid.reshape((len(R170_flow_centroid), 1)))
#S1707
S1707_flow_centroid=centroids[0][:,7]
S1707_flow_centroid_0= series_train_S1707_flow[1].inverse_transform(S1707_flow_centroid.reshape((len(S1707_flow_centroid), 1)))
#S59
S59_flow_centroid=centroids[0][:,8]
S59_flow_centroid_0= series_train_S59_flow[1].inverse_transform(S59_flow_centroid.reshape((len(S59_flow_centroid), 1)))
#R130
R130_flow_centroid=centroids[0][:,9]
R130_flow_centroid_0= series_train_R130_flow[1].inverse_transform(R130_flow_centroid.reshape((len(R130_flow_centroid), 1)))
#R171
R171_flow_centroid=centroids[0][:,10]
R171_flow_centroid_0= series_train_R171_flow[1].inverse_transform(R171_flow_centroid.reshape((len(R171_flow_centroid), 1)))
#S60
S60_flow_centroid=centroids[0][:,11]
S60_flow_centroid_0= series_train_S60_flow[1].inverse_transform(S60_flow_centroid.reshape((len(S60_flow_centroid), 1)))
#S61
S61_flow_centroid=centroids[0][:,12]
S61_flow_centroid_0= series_train_S61_flow[1].inverse_transform(S61_flow_centroid.reshape((len(S61_flow_centroid), 1)))


#second cluster k=1
#S54
#flow
S54_flow_centroid=centroids[1][:,0]
S54_flow_centroid_1= series_train_S54_flow[1].inverse_transform(S54_flow_centroid.reshape((len(S54_flow_centroid), 1)))
#S1706
S1706_flow_centroid=centroids[1][:,1]
S1706_flow_centroid_1= series_train_S1706_flow[1].inverse_transform(S1706_flow_centroid.reshape((len(S1706_flow_centroid), 1)))
#R169
R169_flow_centroid=centroids[1][:,2]
R169_flow_centroid_1= series_train_R169_flow[1].inverse_transform(R169_flow_centroid.reshape((len(R169_flow_centroid), 1)))
#S56
S56_flow_centroid=centroids[1][:,3]
S56_flow_centroid_1= series_train_S56_flow[1].inverse_transform(S56_flow_centroid.reshape((len(S56_flow_centroid), 1)))
#R129
R129_flow_centroid=centroids[1][:,4]
R129_flow_centroid_1= series_train_R129_flow[1].inverse_transform(R129_flow_centroid.reshape((len(R129_flow_centroid), 1)))
#S57
S57_flow_centroid=centroids[1][:,5]
S57_flow_centroid_1= series_train_S57_flow[1].inverse_transform(S57_flow_centroid.reshape((len(S57_flow_centroid), 1)))
#R170
R170_flow_centroid=centroids[1][:,6]
R170_flow_centroid_1= series_train_R170_flow[1].inverse_transform(R170_flow_centroid.reshape((len(R170_flow_centroid), 1)))
#S1707
S1707_flow_centroid=centroids[1][:,7]
S1707_flow_centroid_1= series_train_S1707_flow[1].inverse_transform(S1707_flow_centroid.reshape((len(S1707_flow_centroid), 1)))
#S59
S59_flow_centroid=centroids[1][:,8]
S59_flow_centroid_1= series_train_S59_flow[1].inverse_transform(S59_flow_centroid.reshape((len(S59_flow_centroid), 1)))
#R130
R130_flow_centroid=centroids[1][:,9]
R130_flow_centroid_1= series_train_R130_flow[1].inverse_transform(R130_flow_centroid.reshape((len(R130_flow_centroid), 1)))
#R171
R171_flow_centroid=centroids[1][:,10]
R171_flow_centroid_1= series_train_R171_flow[1].inverse_transform(R171_flow_centroid.reshape((len(R171_flow_centroid), 1)))
#S60
S60_flow_centroid=centroids[1][:,11]
S60_flow_centroid_1= series_train_S60_flow[1].inverse_transform(S60_flow_centroid.reshape((len(S60_flow_centroid), 1)))
#S61
S61_flow_centroid=centroids[1][:,12]
S61_flow_centroid_1= series_train_S61_flow[1].inverse_transform(S61_flow_centroid.reshape((len(S61_flow_centroid), 1)))

#third cluster k=2
#S54
#flow
S54_flow_centroid=centroids[2][:,0]
S54_flow_centroid_2= series_train_S54_flow[1].inverse_transform(S54_flow_centroid.reshape((len(S54_flow_centroid), 1)))
#S1706
S1706_flow_centroid=centroids[2][:,1]
S1706_flow_centroid_2= series_train_S1706_flow[1].inverse_transform(S1706_flow_centroid.reshape((len(S1706_flow_centroid), 1)))
#R169
R169_flow_centroid=centroids[2][:,2]
R169_flow_centroid_2= series_train_R169_flow[1].inverse_transform(R169_flow_centroid.reshape((len(R169_flow_centroid), 1)))
#S56
S56_flow_centroid=centroids[2][:,3]
S56_flow_centroid_2= series_train_S56_flow[1].inverse_transform(S56_flow_centroid.reshape((len(S56_flow_centroid), 1)))
#R129
R129_flow_centroid=centroids[2][:,4]
R129_flow_centroid_2= series_train_R129_flow[1].inverse_transform(R129_flow_centroid.reshape((len(R129_flow_centroid), 1)))
#S57
S57_flow_centroid=centroids[2][:,5]
S57_flow_centroid_2= series_train_S57_flow[1].inverse_transform(S57_flow_centroid.reshape((len(S57_flow_centroid), 1)))
#R170
R170_flow_centroid=centroids[2][:,6]
R170_flow_centroid_2= series_train_R170_flow[1].inverse_transform(R170_flow_centroid.reshape((len(R170_flow_centroid), 1)))
#S1707
S1707_flow_centroid=centroids[2][:,7]
S1707_flow_centroid_2= series_train_S1707_flow[1].inverse_transform(S1707_flow_centroid.reshape((len(S1707_flow_centroid), 1)))
#S59
S59_flow_centroid=centroids[2][:,8]
S59_flow_centroid_2= series_train_S59_flow[1].inverse_transform(S59_flow_centroid.reshape((len(S59_flow_centroid), 1)))
#R130
R130_flow_centroid=centroids[2][:,9]
R130_flow_centroid_2= series_train_R130_flow[1].inverse_transform(R130_flow_centroid.reshape((len(R130_flow_centroid), 1)))
#R171
R171_flow_centroid=centroids[2][:,10]
R171_flow_centroid_2= series_train_R171_flow[1].inverse_transform(R171_flow_centroid.reshape((len(R171_flow_centroid), 1)))
#S60
S60_flow_centroid=centroids[2][:,11]
S60_flow_centroid_2= series_train_S60_flow[1].inverse_transform(S60_flow_centroid.reshape((len(S60_flow_centroid), 1)))
#S61
S61_flow_centroid=centroids[2][:,12]
S61_flow_centroid_2= series_train_S61_flow[1].inverse_transform(S61_flow_centroid.reshape((len(S61_flow_centroid), 1)))

#fourth cluster k=3
#S54
#flow
S54_flow_centroid=centroids[3][:,0]
S54_flow_centroid_3= series_train_S54_flow[1].inverse_transform(S54_flow_centroid.reshape((len(S54_flow_centroid), 1)))
#S1706
S1706_flow_centroid=centroids[3][:,1]
S1706_flow_centroid_3= series_train_S1706_flow[1].inverse_transform(S1706_flow_centroid.reshape((len(S1706_flow_centroid), 1)))
#R169
R169_flow_centroid=centroids[3][:,2]
R169_flow_centroid_3= series_train_R169_flow[1].inverse_transform(R169_flow_centroid.reshape((len(R169_flow_centroid), 1)))
#S56
S56_flow_centroid=centroids[3][:,3]
S56_flow_centroid_3= series_train_S56_flow[1].inverse_transform(S56_flow_centroid.reshape((len(S56_flow_centroid), 1)))
#R129
R129_flow_centroid=centroids[3][:,4]
R129_flow_centroid_3= series_train_R129_flow[1].inverse_transform(R129_flow_centroid.reshape((len(R129_flow_centroid), 1)))
#S57
S57_flow_centroid=centroids[3][:,5]
S57_flow_centroid_3= series_train_S57_flow[1].inverse_transform(S57_flow_centroid.reshape((len(S57_flow_centroid), 1)))
#R170
R170_flow_centroid=centroids[3][:,6]
R170_flow_centroid_3= series_train_R170_flow[1].inverse_transform(R170_flow_centroid.reshape((len(R170_flow_centroid), 1)))
#S1707
S1707_flow_centroid=centroids[3][:,7]
S1707_flow_centroid_3= series_train_S1707_flow[1].inverse_transform(S1707_flow_centroid.reshape((len(S1707_flow_centroid), 1)))
#S59
S59_flow_centroid=centroids[3][:,8]
S59_flow_centroid_3= series_train_S59_flow[1].inverse_transform(S59_flow_centroid.reshape((len(S59_flow_centroid), 1)))
#R130
R130_flow_centroid=centroids[3][:,9]
R130_flow_centroid_3= series_train_R130_flow[1].inverse_transform(R130_flow_centroid.reshape((len(R130_flow_centroid), 1)))
#R171
R171_flow_centroid=centroids[3][:,10]
R171_flow_centroid_3= series_train_R171_flow[1].inverse_transform(R171_flow_centroid.reshape((len(R171_flow_centroid), 1)))
#S60
S60_flow_centroid=centroids[3][:,11]
S60_flow_centroid_3= series_train_S60_flow[1].inverse_transform(S60_flow_centroid.reshape((len(S60_flow_centroid), 1)))
#S61
S61_flow_centroid=centroids[3][:,12]
S61_flow_centroid_3= series_train_S61_flow[1].inverse_transform(S61_flow_centroid.reshape((len(S61_flow_centroid), 1)))


#save centroids of the cluster 
columns = ['S54 flow (veh/h)','S1706 flow (veh/h)', 'R169 flow (veh/h)','S56 flow (veh/h)','R129 flow (veh/h)', 'S57 flow (veh/h)','R170 flow (veh/h)','S1707 flow (veh/h)', 'S59 flow (veh/h)','R130 flow (veh/h)','R171 flow (veh/h)', 'S60 flow (veh/h)','S61 flow (veh/h)']
index=pd.date_range("5:00", periods=180, freq="6min")
index
df_0 = pd.DataFrame(index=index.time, columns=columns)
df_0['S54 flow (veh/h)']=S54_flow_centroid_0
df_0['S1706 flow (veh/h)']=S1706_flow_centroid_0
df_0['R169 flow (veh/h)']=R169_flow_centroid_0
df_0['S56 flow (veh/h)']=S56_flow_centroid_0
df_0['R129 flow (veh/h)']=R129_flow_centroid_0
df_0['S57 flow (veh/h)']=S57_flow_centroid_0
df_0['R170 flow (veh/h)']=R170_flow_centroid_0
df_0['S1707 flow (veh/h)']=S1707_flow_centroid_0
df_0['S59 flow (veh/h)']=S59_flow_centroid_0
df_0['R130 flow (veh/h)']=R130_flow_centroid_0
df_0['R171 flow (veh/h)']=R171_flow_centroid_0
df_0['S60 flow (veh/h)']=S60_flow_centroid_0
df_0['S61 flow (veh/h)']=S61_flow_centroid_0
df_0

df_1 = pd.DataFrame(index=index.time, columns=columns)
df_1['S54 flow (veh/h)']=S54_flow_centroid_1
df_1['S1706 flow (veh/h)']=S1706_flow_centroid_1
df_1['R169 flow (veh/h)']=R169_flow_centroid_1
df_1['S56 flow (veh/h)']=S56_flow_centroid_1
df_1['R129 flow (veh/h)']=R129_flow_centroid_1
df_1['S57 flow (veh/h)']=S57_flow_centroid_1
df_1['R170 flow (veh/h)']=R170_flow_centroid_1
df_1['S1707 flow (veh/h)']=S1707_flow_centroid_1
df_1['S59 flow (veh/h)']=S59_flow_centroid_1
df_1['R130 flow (veh/h)']=R130_flow_centroid_1
df_1['R171 flow (veh/h)']=R171_flow_centroid_1
df_1['S60 flow (veh/h)']=S60_flow_centroid_1
df_1['S61 flow (veh/h)']=S61_flow_centroid_1
df_1

df_2 = pd.DataFrame(index=index.time, columns=columns)
df_2['S54 flow (veh/h)']=S54_flow_centroid_2
df_2['S1706 flow (veh/h)']=S1706_flow_centroid_2
df_2['R169 flow (veh/h)']=R169_flow_centroid_2
df_2['S56 flow (veh/h)']=S56_flow_centroid_2
df_2['R129 flow (veh/h)']=R129_flow_centroid_2
df_2['S57 flow (veh/h)']=S57_flow_centroid_2
df_2['R170 flow (veh/h)']=R170_flow_centroid_2
df_2['S1707 flow (veh/h)']=S1707_flow_centroid_2
df_2['S59 flow (veh/h)']=S59_flow_centroid_2
df_2['R130 flow (veh/h)']=R130_flow_centroid_2
df_2['R171 flow (veh/h)']=R171_flow_centroid_2
df_2['S60 flow (veh/h)']=S60_flow_centroid_2
df_2['S61 flow (veh/h)']=S61_flow_centroid_2
df_2

df_3 = pd.DataFrame(index=index.time, columns=columns)
df_3['S54 flow (veh/h)']=S54_flow_centroid_3
df_3['S1706 flow (veh/h)']=S1706_flow_centroid_3
df_3['R169 flow (veh/h)']=R169_flow_centroid_3
df_3['S56 flow (veh/h)']=S56_flow_centroid_3
df_3['R129 flow (veh/h)']=R129_flow_centroid_3
df_3['S57 flow (veh/h)']=S57_flow_centroid_3
df_3['R170 flow (veh/h)']=R170_flow_centroid_3
df_3['S1707 flow (veh/h)']=S1707_flow_centroid_3
df_3['S59 flow (veh/h)']=S59_flow_centroid_3
df_3['R130 flow (veh/h)']=R130_flow_centroid_3
df_3['R171 flow (veh/h)']=R171_flow_centroid_3
df_3['S60 flow (veh/h)']=S60_flow_centroid_3
df_3['S61 flow (veh/h)']=S61_flow_centroid_3
df_3



# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter('/Users/nronzoni/Desktop/TrafficData Minnesota/Scenario2.xlsx', engine='xlsxwriter')

# Write each dataframe to a different worksheet.
df_0.to_excel(writer, sheet_name='k=0')
df_1.to_excel(writer, sheet_name='k=1')
df_2.to_excel(writer, sheet_name='k=2')
df_3.to_excel(writer, sheet_name='k=3')
# Close the Pandas Excel writer and output the Excel file.
writer.save()

#scenario 5
#first cluster k=0
#S54
#flow
S54_flow_centroid=centroids[0][:,0]
S54_flow_centroid_0= series_train_S54_flow[1].inverse_transform(S54_flow_centroid.reshape((len(S54_flow_centroid), 1)))
#S1706
S1706_flow_centroid=centroids[0][:,1]
S1706_flow_centroid_0= series_train_S1706_flow[1].inverse_transform(S1706_flow_centroid.reshape((len(S1706_flow_centroid), 1)))

#S56
S56_flow_centroid=centroids[0][:,2]
S56_flow_centroid_0= series_train_S56_flow[1].inverse_transform(S56_flow_centroid.reshape((len(S56_flow_centroid), 1)))

#S57
S57_flow_centroid=centroids[0][:,3]
S57_flow_centroid_0= series_train_S57_flow[1].inverse_transform(S57_flow_centroid.reshape((len(S57_flow_centroid), 1)))

#S1707
S1707_flow_centroid=centroids[0][:,4]
S1707_flow_centroid_0= series_train_S1707_flow[1].inverse_transform(S1707_flow_centroid.reshape((len(S1707_flow_centroid), 1)))
#S59
S59_flow_centroid=centroids[0][:,5]
S59_flow_centroid_0= series_train_S59_flow[1].inverse_transform(S59_flow_centroid.reshape((len(S59_flow_centroid), 1)))

#S60
S60_flow_centroid=centroids[0][:,6]
S60_flow_centroid_0= series_train_S60_flow[1].inverse_transform(S60_flow_centroid.reshape((len(S60_flow_centroid), 1)))
#S61
S61_flow_centroid=centroids[0][:,7]
S61_flow_centroid_0= series_train_S61_flow[1].inverse_transform(S61_flow_centroid.reshape((len(S61_flow_centroid), 1)))


#second cluster k=1
#S54
#flow
S54_flow_centroid=centroids[1][:,0]
S54_flow_centroid_1= series_train_S54_flow[1].inverse_transform(S54_flow_centroid.reshape((len(S54_flow_centroid), 1)))
#S1706
S1706_flow_centroid=centroids[1][:,1]
S1706_flow_centroid_1= series_train_S1706_flow[1].inverse_transform(S1706_flow_centroid.reshape((len(S1706_flow_centroid), 1)))

#S56
S56_flow_centroid=centroids[1][:,2]
S56_flow_centroid_1= series_train_S56_flow[1].inverse_transform(S56_flow_centroid.reshape((len(S56_flow_centroid), 1)))

#S57
S57_flow_centroid=centroids[1][:,3]
S57_flow_centroid_1= series_train_S57_flow[1].inverse_transform(S57_flow_centroid.reshape((len(S57_flow_centroid), 1)))

#S1707
S1707_flow_centroid=centroids[1][:,4]
S1707_flow_centroid_1= series_train_S1707_flow[1].inverse_transform(S1707_flow_centroid.reshape((len(S1707_flow_centroid), 1)))
#S59
S59_flow_centroid=centroids[1][:,5]
S59_flow_centroid_1= series_train_S59_flow[1].inverse_transform(S59_flow_centroid.reshape((len(S59_flow_centroid), 1)))

#S60
S60_flow_centroid=centroids[1][:,6]
S60_flow_centroid_1= series_train_S60_flow[1].inverse_transform(S60_flow_centroid.reshape((len(S60_flow_centroid), 1)))
#S61
S61_flow_centroid=centroids[1][:,7]
S61_flow_centroid_1= series_train_S61_flow[1].inverse_transform(S61_flow_centroid.reshape((len(S61_flow_centroid), 1)))

#third cluster k=2
#S54
#flow
S54_flow_centroid=centroids[2][:,0]
S54_flow_centroid_2= series_train_S54_flow[1].inverse_transform(S54_flow_centroid.reshape((len(S54_flow_centroid), 1)))
#S1706
S1706_flow_centroid=centroids[2][:,1]
S1706_flow_centroid_2= series_train_S1706_flow[1].inverse_transform(S1706_flow_centroid.reshape((len(S1706_flow_centroid), 1)))

#S56
S56_flow_centroid=centroids[2][:,2]
S56_flow_centroid_2= series_train_S56_flow[1].inverse_transform(S56_flow_centroid.reshape((len(S56_flow_centroid), 1)))
#S57
S57_flow_centroid=centroids[2][:,3]
S57_flow_centroid_2= series_train_S57_flow[1].inverse_transform(S57_flow_centroid.reshape((len(S57_flow_centroid), 1)))

#S1707
S1707_flow_centroid=centroids[2][:,4]
S1707_flow_centroid_2= series_train_S1707_flow[1].inverse_transform(S1707_flow_centroid.reshape((len(S1707_flow_centroid), 1)))
#S59
S59_flow_centroid=centroids[2][:,5]
S59_flow_centroid_2= series_train_S59_flow[1].inverse_transform(S59_flow_centroid.reshape((len(S59_flow_centroid), 1)))

#S60
S60_flow_centroid=centroids[2][:,6]
S60_flow_centroid_2= series_train_S60_flow[1].inverse_transform(S60_flow_centroid.reshape((len(S60_flow_centroid), 1)))
#S61
S61_flow_centroid=centroids[2][:,7]
S61_flow_centroid_2= series_train_S61_flow[1].inverse_transform(S61_flow_centroid.reshape((len(S61_flow_centroid), 1)))

#fourth cluster k=3
#S54
#flow
S54_flow_centroid=centroids[3][:,0]
S54_flow_centroid_3= series_train_S54_flow[1].inverse_transform(S54_flow_centroid.reshape((len(S54_flow_centroid), 1)))
#S1706
S1706_flow_centroid=centroids[3][:,1]
S1706_flow_centroid_3= series_train_S1706_flow[1].inverse_transform(S1706_flow_centroid.reshape((len(S1706_flow_centroid), 1)))

#S56
S56_flow_centroid=centroids[3][:,2]
S56_flow_centroid_3= series_train_S56_flow[1].inverse_transform(S56_flow_centroid.reshape((len(S56_flow_centroid), 1)))

#S57
S57_flow_centroid=centroids[3][:,3]
S57_flow_centroid_3= series_train_S57_flow[1].inverse_transform(S57_flow_centroid.reshape((len(S57_flow_centroid), 1)))

#S1707
S1707_flow_centroid=centroids[3][:,4]
S1707_flow_centroid_3= series_train_S1707_flow[1].inverse_transform(S1707_flow_centroid.reshape((len(S1707_flow_centroid), 1)))
#S59
S59_flow_centroid=centroids[3][:,5]
S59_flow_centroid_3= series_train_S59_flow[1].inverse_transform(S59_flow_centroid.reshape((len(S59_flow_centroid), 1)))

#S60
S60_flow_centroid=centroids[3][:,6]
S60_flow_centroid_3= series_train_S60_flow[1].inverse_transform(S60_flow_centroid.reshape((len(S60_flow_centroid), 1)))
#S61
S61_flow_centroid=centroids[3][:,7]
S61_flow_centroid_3= series_train_S61_flow[1].inverse_transform(S61_flow_centroid.reshape((len(S61_flow_centroid), 1)))


#save centroids of the cluster 
columns = ['S54 flow (veh/h)','S1706 flow (veh/h)','S56 flow (veh/h)', 'S57 flow (veh/h)','S1707 flow (veh/h)', 'S59 flow (veh/h)', 'S60 flow (veh/h)','S61 flow (veh/h)']
index=pd.date_range("5:00", periods=180, freq="6min")
index
df_0 = pd.DataFrame(index=index.time, columns=columns)
df_0['S54 flow (veh/h)']=S54_flow_centroid_0
df_0['S1706 flow (veh/h)']=S1706_flow_centroid_0
df_0['S56 flow (veh/h)']=S56_flow_centroid_0
df_0['S57 flow (veh/h)']=S57_flow_centroid_0
df_0['S1707 flow (veh/h)']=S1707_flow_centroid_0
df_0['S59 flow (veh/h)']=S59_flow_centroid_0
df_0['S60 flow (veh/h)']=S60_flow_centroid_0
df_0['S61 flow (veh/h)']=S61_flow_centroid_0
df_0

df_1 = pd.DataFrame(index=index.time, columns=columns)
df_1['S54 flow (veh/h)']=S54_flow_centroid_1
df_1['S1706 flow (veh/h)']=S1706_flow_centroid_1
df_1['S56 flow (veh/h)']=S56_flow_centroid_1
df_1['S57 flow (veh/h)']=S57_flow_centroid_1
df_1['S1707 flow (veh/h)']=S1707_flow_centroid_1
df_1['S59 flow (veh/h)']=S59_flow_centroid_1
df_1['S60 flow (veh/h)']=S60_flow_centroid_1
df_1['S61 flow (veh/h)']=S61_flow_centroid_1
df_1

df_2 = pd.DataFrame(index=index.time, columns=columns)
df_2['S54 flow (veh/h)']=S54_flow_centroid_2
df_2['S1706 flow (veh/h)']=S1706_flow_centroid_2
df_2['S56 flow (veh/h)']=S56_flow_centroid_2
df_2['S57 flow (veh/h)']=S57_flow_centroid_2
df_2['S1707 flow (veh/h)']=S1707_flow_centroid_2
df_2['S59 flow (veh/h)']=S59_flow_centroid_2
df_2['S60 flow (veh/h)']=S60_flow_centroid_2
df_2['S61 flow (veh/h)']=S61_flow_centroid_2
df_2

df_3 = pd.DataFrame(index=index.time, columns=columns)
df_3['S54 flow (veh/h)']=S54_flow_centroid_3
df_3['S1706 flow (veh/h)']=S1706_flow_centroid_3
df_3['S56 flow (veh/h)']=S56_flow_centroid_3
df_3['S57 flow (veh/h)']=S57_flow_centroid_3
df_3['S1707 flow (veh/h)']=S1707_flow_centroid_3
df_3['S59 flow (veh/h)']=S59_flow_centroid_3
df_3['S60 flow (veh/h)']=S60_flow_centroid_3
df_3['S61 flow (veh/h)']=S61_flow_centroid_3
df_3



# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter('/Users/nronzoni/Desktop/TrafficData Minnesota/Scenario5.xlsx', engine='xlsxwriter')

# Write each dataframe to a different worksheet.
df_0.to_excel(writer, sheet_name='k=0')
df_1.to_excel(writer, sheet_name='k=1')
df_2.to_excel(writer, sheet_name='k=2')
df_3.to_excel(writer, sheet_name='k=3')
# Close the Pandas Excel writer and output the Excel file.
writer.save()

#Scenario 4 
#first cluster k=0
#S54
S54_flow_centroid=centroids[0][:,0]
S54_flow_centroid_0= series_train_S54_flow[1].inverse_transform(S54_flow_centroid.reshape((len(S54_flow_centroid), 1)))
S54_speed_centroid=centroids[0][:,1]
S54_speed_centroid_0= series_train_S54_speed[1].inverse_transform(S54_speed_centroid.reshape((len(S54_speed_centroid), 1)))
#S1706
S1706_flow_centroid=centroids[0][:,2]
S1706_flow_centroid_0= series_train_S1706_flow[1].inverse_transform(S1706_flow_centroid.reshape((len(S1706_flow_centroid), 1)))
S1706_speed_centroid=centroids[0][:,3]
S1706_speed_centroid_0= series_train_S1706_speed[1].inverse_transform(S1706_speed_centroid.reshape((len(S1706_speed_centroid), 1)))
#R169
R169_flow_centroid=centroids[0][:,4]
R169_flow_centroid_0= series_train_R169_flow[1].inverse_transform(R169_flow_centroid.reshape((len(R169_flow_centroid), 1)))
R169_speed_centroid=centroids[0][:,5]
R169_speed_centroid_0= series_train_R169_speed[1].inverse_transform(R169_speed_centroid.reshape((len(R169_speed_centroid), 1)))
#S56
S56_flow_centroid=centroids[0][:,6]
S56_flow_centroid_0= series_train_S56_flow[1].inverse_transform(S56_flow_centroid.reshape((len(S56_flow_centroid), 1)))
S56_speed_centroid=centroids[0][:,7]
S56_speed_centroid_0= series_train_S56_speed[1].inverse_transform(S56_speed_centroid.reshape((len(S56_speed_centroid), 1)))
#R129
R129_flow_centroid=centroids[0][:,8]
R129_flow_centroid_0= series_train_R129_flow[1].inverse_transform(R129_flow_centroid.reshape((len(R129_flow_centroid), 1)))
R129_speed_centroid=centroids[0][:,9]
R129_speed_centroid_0= series_train_R129_speed[1].inverse_transform(R129_speed_centroid.reshape((len(R129_speed_centroid), 1)))
#S57
S57_flow_centroid=centroids[0][:,10]
S57_flow_centroid_0= series_train_S57_flow[1].inverse_transform(S57_flow_centroid.reshape((len(S57_flow_centroid), 1)))
S57_speed_centroid=centroids[0][:,11]
S57_speed_centroid_0= series_train_S57_speed[1].inverse_transform(S57_speed_centroid.reshape((len(S57_speed_centroid), 1)))
#R170
R170_flow_centroid=centroids[0][:,12]
R170_flow_centroid_0= series_train_R170_flow[1].inverse_transform(R170_flow_centroid.reshape((len(R170_flow_centroid), 1)))
R170_speed_centroid=centroids[0][:,13]
R170_speed_centroid_0= series_train_R170_speed[1].inverse_transform(R170_speed_centroid.reshape((len(R170_speed_centroid), 1)))
#S1707
S1707_flow_centroid=centroids[0][:,14]
S1707_flow_centroid_0= series_train_S1707_flow[1].inverse_transform(S1707_flow_centroid.reshape((len(S1707_flow_centroid), 1)))
S1707_speed_centroid=centroids[0][:,15]
S1707_speed_centroid_0= series_train_S1707_speed[1].inverse_transform(S1707_speed_centroid.reshape((len(S1707_speed_centroid), 1)))
#S59
S59_flow_centroid=centroids[0][:,16]
S59_flow_centroid_0= series_train_S59_flow[1].inverse_transform(S59_flow_centroid.reshape((len(S59_flow_centroid), 1)))
S59_speed_centroid=centroids[0][:,17]
S59_speed_centroid_0= series_train_S59_speed[1].inverse_transform(S59_speed_centroid.reshape((len(S59_speed_centroid), 1)))
#R130
R130_flow_centroid=centroids[0][:,18]
R130_flow_centroid_0= series_train_R130_flow[1].inverse_transform(R130_flow_centroid.reshape((len(R130_flow_centroid), 1)))
R130_speed_centroid=centroids[0][:,19]
R130_speed_centroid_0= series_train_R130_speed[1].inverse_transform(R130_speed_centroid.reshape((len(R130_speed_centroid), 1)))
#R171
R171_flow_centroid=centroids[0][:,20]
R171_flow_centroid_0= series_train_R171_flow[1].inverse_transform(R171_flow_centroid.reshape((len(R171_flow_centroid), 1)))
R171_speed_centroid=centroids[0][:,21]
R171_speed_centroid_0= series_train_R171_speed[1].inverse_transform(R171_speed_centroid.reshape((len(R171_speed_centroid), 1)))
#S60
S60_flow_centroid=centroids[0][:,22]
S60_flow_centroid_0= series_train_S60_flow[1].inverse_transform(S60_flow_centroid.reshape((len(S60_flow_centroid), 1)))
S60_speed_centroid=centroids[0][:,23]
S60_speed_centroid_0= series_train_S60_speed[1].inverse_transform(S60_speed_centroid.reshape((len(S60_speed_centroid), 1)))
#S61
S61_flow_centroid=centroids[0][:,24]
S61_flow_centroid_0= series_train_S61_flow[1].inverse_transform(S61_flow_centroid.reshape((len(S61_flow_centroid), 1)))
S61_speed_centroid=centroids[0][:,25]
S61_speed_centroid_0= series_train_S61_speed[1].inverse_transform(S61_speed_centroid.reshape((len(S61_speed_centroid), 1)))

#second cluster k=1
S54_flow_centroid=centroids[1][:,0]
S54_flow_centroid_1= series_train_S54_flow[1].inverse_transform(S54_flow_centroid.reshape((len(S54_flow_centroid), 1)))
S54_speed_centroid=centroids[1][:,1]
S54_speed_centroid_1= series_train_S54_speed[1].inverse_transform(S54_speed_centroid.reshape((len(S54_speed_centroid), 1)))
#S1706
S1706_flow_centroid=centroids[1][:,2]
S1706_flow_centroid_1= series_train_S1706_flow[1].inverse_transform(S1706_flow_centroid.reshape((len(S1706_flow_centroid), 1)))
S1706_speed_centroid=centroids[1][:,3]
S1706_speed_centroid_1= series_train_S1706_speed[1].inverse_transform(S1706_speed_centroid.reshape((len(S1706_speed_centroid), 1)))
#R169
R169_flow_centroid=centroids[1][:,4]
R169_flow_centroid_1= series_train_R169_flow[1].inverse_transform(R169_flow_centroid.reshape((len(R169_flow_centroid), 1)))
R169_speed_centroid=centroids[1][:,5]
R169_speed_centroid_1= series_train_R169_speed[1].inverse_transform(R169_speed_centroid.reshape((len(R169_speed_centroid), 1)))
#S56
S56_flow_centroid=centroids[1][:,6]
S56_flow_centroid_1= series_train_S56_flow[1].inverse_transform(S56_flow_centroid.reshape((len(S56_flow_centroid), 1)))
S56_speed_centroid=centroids[1][:,7]
S56_speed_centroid_1= series_train_S56_speed[1].inverse_transform(S56_speed_centroid.reshape((len(S56_speed_centroid), 1)))
#R129
R129_flow_centroid=centroids[1][:,8]
R129_flow_centroid_1= series_train_R129_flow[1].inverse_transform(R129_flow_centroid.reshape((len(R129_flow_centroid), 1)))
R129_speed_centroid=centroids[1][:,9]
R129_speed_centroid_1= series_train_R129_speed[1].inverse_transform(R129_speed_centroid.reshape((len(R129_speed_centroid), 1)))
#S57
S57_flow_centroid=centroids[1][:,10]
S57_flow_centroid_1= series_train_S57_flow[1].inverse_transform(S57_flow_centroid.reshape((len(S57_flow_centroid), 1)))
S57_speed_centroid=centroids[1][:,11]
S57_speed_centroid_1= series_train_S57_speed[1].inverse_transform(S57_speed_centroid.reshape((len(S57_speed_centroid), 1)))
#R170
R170_flow_centroid=centroids[1][:,12]
R170_flow_centroid_1= series_train_R170_flow[1].inverse_transform(R170_flow_centroid.reshape((len(R170_flow_centroid), 1)))
R170_speed_centroid=centroids[1][:,13]
R170_speed_centroid_1= series_train_R170_speed[1].inverse_transform(R170_speed_centroid.reshape((len(R170_speed_centroid), 1)))
#S1707
S1707_flow_centroid=centroids[1][:,14]
S1707_flow_centroid_1= series_train_S1707_flow[1].inverse_transform(S1707_flow_centroid.reshape((len(S1707_flow_centroid), 1)))
S1707_speed_centroid=centroids[1][:,15]
S1707_speed_centroid_1= series_train_S1707_speed[1].inverse_transform(S1707_speed_centroid.reshape((len(S1707_speed_centroid), 1)))
#S59
S59_flow_centroid=centroids[1][:,16]
S59_flow_centroid_1= series_train_S59_flow[1].inverse_transform(S59_flow_centroid.reshape((len(S59_flow_centroid), 1)))
S59_speed_centroid=centroids[1][:,17]
S59_speed_centroid_1= series_train_S59_speed[1].inverse_transform(S59_speed_centroid.reshape((len(S59_speed_centroid), 1)))
#R130
R130_flow_centroid=centroids[1][:,18]
R130_flow_centroid_1= series_train_R130_flow[1].inverse_transform(R130_flow_centroid.reshape((len(R130_flow_centroid), 1)))
R130_speed_centroid=centroids[1][:,19]
R130_speed_centroid_1= series_train_R130_speed[1].inverse_transform(R130_speed_centroid.reshape((len(R130_speed_centroid), 1)))
#R171
R171_flow_centroid=centroids[1][:,20]
R171_flow_centroid_1= series_train_R171_flow[1].inverse_transform(R171_flow_centroid.reshape((len(R171_flow_centroid), 1)))
R171_speed_centroid=centroids[1][:,21]
R171_speed_centroid_1= series_train_R171_speed[1].inverse_transform(R171_speed_centroid.reshape((len(R171_speed_centroid), 1)))
#S60
S60_flow_centroid=centroids[1][:,22]
S60_flow_centroid_1= series_train_S60_flow[1].inverse_transform(S60_flow_centroid.reshape((len(S60_flow_centroid), 1)))
S60_speed_centroid=centroids[1][:,23]
S60_speed_centroid_1= series_train_S60_speed[1].inverse_transform(S60_speed_centroid.reshape((len(S60_speed_centroid), 1)))
#S61
S61_flow_centroid=centroids[1][:,24]
S61_flow_centroid_1= series_train_S61_flow[1].inverse_transform(S61_flow_centroid.reshape((len(S61_flow_centroid), 1)))
S61_speed_centroid=centroids[1][:,25]
S61_speed_centroid_1= series_train_S61_speed[1].inverse_transform(S61_speed_centroid.reshape((len(S61_speed_centroid), 1)))

#save centroids of the cluster 
columns = ['S54 flow (veh/h)','S54 speed (km/h)','S1706 flow (veh/h)','S1706 speed (km/h)', 'R169 flow (veh/h)','R169 speed (km/h)','S56 flow (veh/h)','S56 speed (km/h)','R129 flow (veh/h)','R129 speed (km/h)','S57 flow (veh/h)', 'S57 speed (km/h)','R170 flow (veh/h)','R170 speed (km/h)','S1707 flow (veh/h)','S1707 speed (km/h)','S59 flow (veh/h)', 'S59 speed (km/h)','R130 flow (veh/h)','R130 speed (km/h)','R171 flow (veh/h)','R171 speed (km/h)','S60 flow (km/h)', 'S60 speed (km/h)','S61 flow (km/h)','S61 speed (km/h)']
index=pd.date_range("5:00", periods=180, freq="6min")
index
df_0 = pd.DataFrame(index=index.time, columns=columns)
df_0['S54 flow (veh/h)']=S54_flow_centroid_0
df_0['S54 speed (km/h)']=S54_speed_centroid_0
df_0['S1706 flow (veh/h)']=S1706_flow_centroid_0
df_0['S1706 speed (km/h)']=S1706_speed_centroid_0
df_0['R169 flow (veh/h)']=R169_flow_centroid_0
df_0['R169 speed (km/h)']=R169_speed_centroid_0
df_0['S56 flow (veh/h)']=S56_flow_centroid_0
df_0['S56 speed (km/h)']=S56_speed_centroid_0
df_0['R129 flow (veh/h)']=R129_flow_centroid_0
df_0['R129 speed (km/h)']=R129_speed_centroid_0
df_0['S57 flow (veh/h)']=S57_flow_centroid_0
df_0['S57 speed (km/h)']=S57_speed_centroid_0
df_0['R170 flow (veh/h)']=R170_flow_centroid_0
df_0['R170 speed (km/h)']=R170_speed_centroid_0
df_0['S1707 flow (veh/h)']=S1707_flow_centroid_0
df_0['S1707 speed (km/h)']=S1707_speed_centroid_0
df_0['S59 flow (veh/h)']=S59_flow_centroid_0
df_0['S59 speed (km/h)']=S59_speed_centroid_0
df_0['R130 flow (veh/h)']=R130_flow_centroid_0
df_0['R130 speed (km/h)']=R130_speed_centroid_0
df_0['R171 flow (veh/h)']=R171_flow_centroid_0
df_0['R171 speed (km/h)']=R171_speed_centroid_0
df_0['S60 flow (km/h)']=S60_flow_centroid_0
df_0['S60 speed (km/h)']=S60_speed_centroid_0
df_0['S61 flow (km/h)']=S61_flow_centroid_0
df_0['S61 speed (km/h)']=S61_speed_centroid_0
df_0

df_1 = pd.DataFrame(index=index.time, columns=columns)

df_1['S54 flow (veh/h)']=S54_flow_centroid_1
df_1['S54 speed (km/h)']=S54_speed_centroid_1
df_1['S1706 flow (veh/h)']=S1706_flow_centroid_1
df_1['S1706 speed (km/h)']=S1706_speed_centroid_1
df_1['R169 flow (veh/h)']=R169_flow_centroid_1
df_1['R169 speed (km/h)']=R169_speed_centroid_1
df_1['S56 flow (veh/h)']=S56_flow_centroid_1
df_1['S56 speed (km/h)']=S56_speed_centroid_1
df_1['R129 flow (veh/h)']=R129_flow_centroid_1
df_1['R129 speed (km/h)']=R129_speed_centroid_1
df_1['S57 flow (veh/h)']=S57_flow_centroid_1
df_1['S57 speed (km/h)']=S57_speed_centroid_1
df_1['R170 flow (veh/h)']=R170_flow_centroid_1
df_1['R170 speed (km/h)']=R170_speed_centroid_1
df_1['S1707 flow (veh/h)']=S1707_flow_centroid_1
df_1['S1707 speed (km/h)']=S1707_speed_centroid_1
df_1['S59 flow (veh/h)']=S59_flow_centroid_1
df_1['S59 speed (km/h)']=S59_speed_centroid_1
df_1['R130 flow (veh/h)']=R130_flow_centroid_1
df_1['R130 speed (km/h)']=R130_speed_centroid_1
df_1['R171 flow (veh/h)']=R171_flow_centroid_1
df_1['R171 speed (km/h)']=R171_speed_centroid_1
df_1['S60 flow (km/h)']=S60_flow_centroid_1
df_1['S60 speed (km/h)']=S60_speed_centroid_1
df_1['S61 flow (km/h)']=S61_flow_centroid_1
df_1['S61 speed (km/h)']=S61_speed_centroid_1
df_1

# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter('/Users/nronzoni/Desktop/TrafficData Minnesota/Scenario4.xlsx', engine='xlsxwriter')

# Write each dataframe to a different worksheet.
df_0.to_excel(writer, sheet_name='k=0')
df_1.to_excel(writer, sheet_name='k=1')
# Close the Pandas Excel writer and output the Excel file.
writer.save()


# Scenario 6
#first cluster k=0
#S54
S54_speed_centroid=centroids[0][:,0]
S54_speed_centroid_0= series_train_S54_speed[1].inverse_transform(S54_speed_centroid.reshape((len(S54_speed_centroid), 1)))
#S1706
S1706_speed_centroid=centroids[0][:,1]
S1706_speed_centroid_0= series_train_S1706_speed[1].inverse_transform(S1706_speed_centroid.reshape((len(S1706_speed_centroid), 1)))
#R169
R169_speed_centroid=centroids[0][:,2]
R169_speed_centroid_0= series_train_R169_speed[1].inverse_transform(R169_speed_centroid.reshape((len(R169_speed_centroid), 1)))
#S56
S56_speed_centroid=centroids[0][:,3]
S56_speed_centroid_0= series_train_S56_speed[1].inverse_transform(S56_speed_centroid.reshape((len(S56_speed_centroid), 1)))
#R129
R129_speed_centroid=centroids[0][:,4]
R129_speed_centroid_0= series_train_R129_speed[1].inverse_transform(R129_speed_centroid.reshape((len(R129_speed_centroid), 1)))
#S57
S57_speed_centroid=centroids[0][:,5]
S57_speed_centroid_0= series_train_S57_speed[1].inverse_transform(S57_speed_centroid.reshape((len(S57_speed_centroid), 1)))
#R170
R170_speed_centroid=centroids[0][:,6]
R170_speed_centroid_0= series_train_R170_speed[1].inverse_transform(R170_speed_centroid.reshape((len(R170_speed_centroid), 1)))
#S1707
S1707_speed_centroid=centroids[0][:,7]
S1707_speed_centroid_0= series_train_S1707_speed[1].inverse_transform(S1707_speed_centroid.reshape((len(S1707_speed_centroid), 1)))
#S59
S59_speed_centroid=centroids[0][:,8]
S59_speed_centroid_0= series_train_S59_speed[1].inverse_transform(S59_speed_centroid.reshape((len(S59_speed_centroid), 1)))
#R130
R130_speed_centroid=centroids[0][:,9]
R130_speed_centroid_0= series_train_R130_speed[1].inverse_transform(R130_speed_centroid.reshape((len(R130_speed_centroid), 1)))
#R171
R171_speed_centroid=centroids[0][:,10]
R171_speed_centroid_0= series_train_R171_speed[1].inverse_transform(R171_speed_centroid.reshape((len(R171_speed_centroid), 1)))
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
R169_speed_centroid=centroids[1][:,2]
R169_speed_centroid_1= series_train_R169_speed[1].inverse_transform(R169_speed_centroid.reshape((len(R169_speed_centroid), 1)))
#S56
S56_speed_centroid=centroids[1][:,3]
S56_speed_centroid_1= series_train_S56_speed[1].inverse_transform(S56_speed_centroid.reshape((len(S56_speed_centroid), 1)))
#R129
R129_speed_centroid=centroids[1][:,4]
R129_speed_centroid_1= series_train_R129_speed[1].inverse_transform(R129_speed_centroid.reshape((len(R129_speed_centroid), 1)))
#S57
S57_speed_centroid=centroids[1][:,5]
S57_speed_centroid_1= series_train_S57_speed[1].inverse_transform(S57_speed_centroid.reshape((len(S57_speed_centroid), 1)))
#R170
R170_speed_centroid=centroids[1][:,6]
R170_speed_centroid_1= series_train_R170_speed[1].inverse_transform(R170_speed_centroid.reshape((len(R170_speed_centroid), 1)))
#S1707
S1707_speed_centroid=centroids[1][:,7]
S1707_speed_centroid_1= series_train_S1707_speed[1].inverse_transform(S1707_speed_centroid.reshape((len(S1707_speed_centroid), 1)))
#S59
S59_speed_centroid=centroids[1][:,8]
S59_speed_centroid_1= series_train_S59_speed[1].inverse_transform(S59_speed_centroid.reshape((len(S59_speed_centroid), 1)))
#R130
R130_speed_centroid=centroids[1][:,9]
R130_speed_centroid_1= series_train_R130_speed[1].inverse_transform(R130_speed_centroid.reshape((len(R130_speed_centroid), 1)))
#R171
R171_speed_centroid=centroids[1][:,10]
R171_speed_centroid_1= series_train_R171_speed[1].inverse_transform(R171_speed_centroid.reshape((len(R171_speed_centroid), 1)))
#S60
S60_speed_centroid=centroids[1][:,11]
S60_speed_centroid_1= series_train_S60_speed[1].inverse_transform(S60_speed_centroid.reshape((len(S60_speed_centroid), 1)))
#S61
S61_speed_centroid=centroids[1][:,12]
S61_speed_centroid_1= series_train_S61_speed[1].inverse_transform(S61_speed_centroid.reshape((len(S61_speed_centroid), 1)))

#save centroids of the cluster 
columns = ['S54 speed (km/h)','S1706 speed (km/h)', 'R169 speed (km/h)','S56 speed (km/h)','R129 speed (km/h)', 'S57 speed (km/h)','R170 speed (km/h)','S1707 speed (km/h)', 'S59 speed (km/h)','R130 speed (km/h)','R171 speed (km/h)', 'S60 speed (km/h)','S61 speed (km/h)']
index=pd.date_range("5:00", periods=180, freq="6min")
index
df_0 = pd.DataFrame(index=index.time, columns=columns)
df_0['S54 speed (km/h)']=S54_speed_centroid_0
df_0['S1706 speed (km/h)']=S1706_speed_centroid_0
df_0['R169 speed (km/h)']=R169_speed_centroid_0
df_0['S56 speed (km/h)']=S56_speed_centroid_0
df_0['R129 speed (km/h)']=R129_speed_centroid_0
df_0['S57 speed (km/h)']=S57_speed_centroid_0
df_0['R170 speed (km/h)']=R170_speed_centroid_0
df_0['S1707 speed (km/h)']=S1707_speed_centroid_0
df_0['S59 speed (km/h)']=S59_speed_centroid_0
df_0['R130 speed (km/h)']=R130_speed_centroid_0
df_0['R171 speed (km/h)']=R171_speed_centroid_0
df_0['S60 speed (km/h)']=S60_speed_centroid_0
df_0['S61 speed (km/h)']=S61_speed_centroid_0

df_0

df_1 = pd.DataFrame(index=index.time, columns=columns)
df_1['S54 speed (km/h)']=S54_speed_centroid_1
df_1['S1706 speed (km/h)']=S1706_speed_centroid_1
df_1['R169 speed (km/h)']=R169_speed_centroid_1
df_1['S56 speed (km/h)']=S56_speed_centroid_1
df_1['R129 speed (km/h)']=R129_speed_centroid_1
df_1['S57 speed (km/h)']=S57_speed_centroid_1
df_1['R170 speed (km/h)']=R170_speed_centroid_1
df_1['S1707 speed (km/h)']=S1707_speed_centroid_1
df_1['S59 speed (km/h)']=S59_speed_centroid_1
df_1['R130 speed (km/h)']=R130_speed_centroid_1
df_1['R171 speed (km/h)']=R171_speed_centroid_1
df_1['S60 speed (km/h)']=S60_speed_centroid_1
df_1['S61 speed (km/h)']=S61_speed_centroid_1

df_1

# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter('/Users/nronzoni/Desktop/TrafficData Minnesota/Scenario6.xlsx', engine='xlsxwriter')

# Write each dataframe to a different worksheet.
df_0.to_excel(writer, sheet_name='k=0')
df_1.to_excel(writer, sheet_name='k=1')
# Close the Pandas Excel writer and output the Excel file.
writer.save()



#scenario 7
#first cluster k=0
#S54
#flow
S54_speed_centroid=centroids[0][:,0]
S54_speed_centroid_0= series_train_S54_speed[1].inverse_transform(S54_speed_centroid.reshape((len(S54_speed_centroid), 1)))
#S1706
S1706_speed_centroid=centroids[0][:,1]
S1706_speed_centroid_0= series_train_S1706_speed[1].inverse_transform(S1706_speed_centroid.reshape((len(S1706_speed_centroid), 1)))

#S56
S56_speed_centroid=centroids[0][:,2]
S56_speed_centroid_0= series_train_S56_speed[1].inverse_transform(S56_speed_centroid.reshape((len(S56_speed_centroid), 1)))

#S57
S57_speed_centroid=centroids[0][:,3]
S57_speed_centroid_0= series_train_S57_speed[1].inverse_transform(S57_speed_centroid.reshape((len(S57_speed_centroid), 1)))

#S1707
S1707_speed_centroid=centroids[0][:,4]
S1707_speed_centroid_0= series_train_S1707_speed[1].inverse_transform(S1707_speed_centroid.reshape((len(S1707_speed_centroid), 1)))
#S59
S59_speed_centroid=centroids[0][:,5]
S59_speed_centroid_0= series_train_S59_speed[1].inverse_transform(S59_speed_centroid.reshape((len(S59_speed_centroid), 1)))

#S60
S60_speed_centroid=centroids[0][:,6]
S60_speed_centroid_0= series_train_S60_speed[1].inverse_transform(S60_speed_centroid.reshape((len(S60_speed_centroid), 1)))
#S61
S61_speed_centroid=centroids[0][:,7]
S61_speed_centroid_0= series_train_S61_speed[1].inverse_transform(S61_speed_centroid.reshape((len(S61_speed_centroid), 1)))


#second cluster k=1
#S54
#flow
S54_speed_centroid=centroids[1][:,0]
S54_speed_centroid_1= series_train_S54_speed[1].inverse_transform(S54_speed_centroid.reshape((len(S54_speed_centroid), 1)))
#S1706
S1706_speed_centroid=centroids[1][:,1]
S1706_speed_centroid_1= series_train_S1706_speed[1].inverse_transform(S1706_speed_centroid.reshape((len(S1706_speed_centroid), 1)))

#S56
S56_speed_centroid=centroids[1][:,2]
S56_speed_centroid_1= series_train_S56_speed[1].inverse_transform(S56_speed_centroid.reshape((len(S56_speed_centroid), 1)))

#S57
S57_speed_centroid=centroids[1][:,3]
S57_speed_centroid_1= series_train_S57_speed[1].inverse_transform(S57_speed_centroid.reshape((len(S57_speed_centroid), 1)))

#S1707
S1707_speed_centroid=centroids[1][:,4]
S1707_speed_centroid_1= series_train_S1707_speed[1].inverse_transform(S1707_speed_centroid.reshape((len(S1707_speed_centroid), 1)))
#S59
S59_speed_centroid=centroids[1][:,5]
S59_speed_centroid_1= series_train_S59_speed[1].inverse_transform(S59_speed_centroid.reshape((len(S59_speed_centroid), 1)))

#S60
S60_speed_centroid=centroids[1][:,6]
S60_speed_centroid_1= series_train_S60_speed[1].inverse_transform(S60_speed_centroid.reshape((len(S60_speed_centroid), 1)))
#S61
S61_speed_centroid=centroids[1][:,7]
S61_speed_centroid_1= series_train_S61_speed[1].inverse_transform(S61_speed_centroid.reshape((len(S61_speed_centroid), 1)))

#third cluster k=2
#S54
#flow
S54_speed_centroid=centroids[2][:,0]
S54_speed_centroid_2= series_train_S54_speed[1].inverse_transform(S54_speed_centroid.reshape((len(S54_speed_centroid), 1)))
#S1706
S1706_speed_centroid=centroids[2][:,1]
S1706_speed_centroid_2= series_train_S1706_speed[1].inverse_transform(S1706_speed_centroid.reshape((len(S1706_speed_centroid), 1)))

#S56
S56_speed_centroid=centroids[2][:,2]
S56_speed_centroid_2= series_train_S56_speed[1].inverse_transform(S56_speed_centroid.reshape((len(S56_speed_centroid), 1)))

#S57
S57_speed_centroid=centroids[2][:,3]
S57_speed_centroid_2= series_train_S57_speed[1].inverse_transform(S57_speed_centroid.reshape((len(S57_speed_centroid), 1)))

#S1707
S1707_speed_centroid=centroids[2][:,4]
S1707_speed_centroid_2= series_train_S1707_speed[1].inverse_transform(S1707_speed_centroid.reshape((len(S1707_speed_centroid), 1)))
#S59
S59_speed_centroid=centroids[2][:,5]
S59_speed_centroid_2= series_train_S59_speed[1].inverse_transform(S59_speed_centroid.reshape((len(S59_speed_centroid), 1)))

#S60
S60_speed_centroid=centroids[2][:,6]
S60_speed_centroid_2= series_train_S60_speed[1].inverse_transform(S60_speed_centroid.reshape((len(S60_speed_centroid), 1)))
#S61
S61_speed_centroid=centroids[2][:,7]
S61_speed_centroid_2= series_train_S61_speed[1].inverse_transform(S61_speed_centroid.reshape((len(S61_speed_centroid), 1)))

#save centroids of the cluster 
columns = ['S54 speed (km/h)','S1706 speed (km/h)','S56 speed (km/h)', 'S57 speed (km/h)','S1707 speed (km/h)', 'S59 speed (km/h)', 'S60 speed (km/h)','S61 speed (km/h)']
index=pd.date_range("5:00", periods=180, freq="6min")
index
df_0 = pd.DataFrame(index=index.time, columns=columns)
df_0['S54 speed (km/h)']=S54_speed_centroid_0
df_0['S1706 speed (km/h)']=S1706_speed_centroid_0
df_0['S56 speed (km/h)']=S56_speed_centroid_0
df_0['S57 speed (km/h)']=S57_speed_centroid_0
df_0['S1707 speed (km/h)']=S1707_speed_centroid_0
df_0['S59 speed (km/h)']=S59_speed_centroid_0
df_0['S60 speed (km/h)']=S60_speed_centroid_0
df_0['S61 speed (km/h)']=S61_speed_centroid_0
df_0

df_1 = pd.DataFrame(index=index.time, columns=columns)
df_1['S54 speed (km/h)']=S54_speed_centroid_1
df_1['S1706 speed (km/h)']=S1706_speed_centroid_1
df_1['S56 speed (km/h)']=S56_speed_centroid_1
df_1['S57 speed (km/h)']=S57_speed_centroid_1
df_1['S1707 speed (km/h)']=S1707_speed_centroid_1
df_1['S59 speed (km/h)']=S59_speed_centroid_1
df_1['S60 speed (km/h)']=S60_speed_centroid_1
df_1['S61 speed (km/h)']=S61_speed_centroid_1
df_1

df_2 = pd.DataFrame(index=index.time, columns=columns)
df_2['S54 speed (km/h)']=S54_speed_centroid_2
df_2['S1706 speed (km/h)']=S1706_speed_centroid_2
df_2['S56 speed (km/h)']=S56_speed_centroid_2
df_2['S57 speed (km/h)']=S57_speed_centroid_2
df_2['S1707 speed (km/h)']=S1707_speed_centroid_2
df_2['S59 speed (km/h)']=S59_speed_centroid_2
df_2['S60 speed (km/h)']=S60_speed_centroid_2
df_2['S61 speed (km/h)']=S61_speed_centroid_2
df_2



# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter('/Users/nronzoni/Desktop/TrafficData Minnesota/Scenario7.xlsx', engine='xlsxwriter')

# Write each dataframe to a different worksheet.
df_0.to_excel(writer, sheet_name='k=0')
df_1.to_excel(writer, sheet_name='k=1')
df_2.to_excel(writer, sheet_name='k=2')
# Close the Pandas Excel writer and output the Excel file.
writer.save()

# Scenario 8
#first cluster k=0
#S54
S54_density_centroid=centroids[0][:,0]
S54_density_centroid_0= series_train_S54_density[1].inverse_transform(S54_density_centroid.reshape((len(S54_density_centroid), 1)))
#S1706
S1706_density_centroid=centroids[0][:,1]
S1706_density_centroid_0= series_train_S1706_density[1].inverse_transform(S1706_density_centroid.reshape((len(S1706_density_centroid), 1)))
#R169
R169_density_centroid=centroids[0][:,2]
R169_density_centroid_0= series_train_R169_density[1].inverse_transform(R169_density_centroid.reshape((len(R169_density_centroid), 1)))
#S56
S56_density_centroid=centroids[0][:,3]
S56_density_centroid_0= series_train_S56_density[1].inverse_transform(S56_density_centroid.reshape((len(S56_density_centroid), 1)))
#R129
R129_density_centroid=centroids[0][:,4]
R129_density_centroid_0= series_train_R129_density[1].inverse_transform(R129_density_centroid.reshape((len(R129_density_centroid), 1)))
#S57
S57_density_centroid=centroids[0][:,5]
S57_density_centroid_0= series_train_S57_density[1].inverse_transform(S57_density_centroid.reshape((len(S57_density_centroid), 1)))
#R170
R170_density_centroid=centroids[0][:,6]
R170_density_centroid_0= series_train_R170_density[1].inverse_transform(R170_density_centroid.reshape((len(R170_density_centroid), 1)))
#S1707
S1707_density_centroid=centroids[0][:,7]
S1707_density_centroid_0= series_train_S1707_density[1].inverse_transform(S1707_density_centroid.reshape((len(S1707_density_centroid), 1)))
#S59
S59_density_centroid=centroids[0][:,8]
S59_density_centroid_0= series_train_S59_density[1].inverse_transform(S59_density_centroid.reshape((len(S59_density_centroid), 1)))
#R130
R130_density_centroid=centroids[0][:,9]
R130_density_centroid_0= series_train_R130_density[1].inverse_transform(R130_density_centroid.reshape((len(R130_density_centroid), 1)))
#R171
R171_density_centroid=centroids[0][:,10]
R171_density_centroid_0= series_train_R171_density[1].inverse_transform(R171_density_centroid.reshape((len(R171_density_centroid), 1)))
#S60
S60_density_centroid=centroids[0][:,11]
S60_density_centroid_0= series_train_S60_density[1].inverse_transform(S60_density_centroid.reshape((len(S60_density_centroid), 1)))
#S61
S61_density_centroid=centroids[0][:,12]
S61_density_centroid_0= series_train_S61_density[1].inverse_transform(S61_density_centroid.reshape((len(S61_density_centroid), 1)))

#second cluster k=1
#S54
S54_density_centroid=centroids[1][:,0]
S54_density_centroid_1= series_train_S54_density[1].inverse_transform(S54_density_centroid.reshape((len(S54_density_centroid), 1)))
#S1706
S1706_density_centroid=centroids[1][:,1]
S1706_density_centroid_1= series_train_S1706_density[1].inverse_transform(S1706_density_centroid.reshape((len(S1706_density_centroid), 1)))
#R169
R169_density_centroid=centroids[1][:,2]
R169_density_centroid_1= series_train_R169_density[1].inverse_transform(R169_density_centroid.reshape((len(R169_density_centroid), 1)))
#S56
S56_density_centroid=centroids[1][:,3]
S56_density_centroid_1= series_train_S56_density[1].inverse_transform(S56_density_centroid.reshape((len(S56_density_centroid), 1)))
#R129
R129_density_centroid=centroids[1][:,4]
R129_density_centroid_1= series_train_R129_density[1].inverse_transform(R129_density_centroid.reshape((len(R129_density_centroid), 1)))
#S57
S57_density_centroid=centroids[1][:,5]
S57_density_centroid_1= series_train_S57_density[1].inverse_transform(S57_density_centroid.reshape((len(S57_density_centroid), 1)))
#R170
R170_density_centroid=centroids[1][:,6]
R170_density_centroid_1= series_train_R170_density[1].inverse_transform(R170_density_centroid.reshape((len(R170_density_centroid), 1)))
#S1707
S1707_density_centroid=centroids[1][:,7]
S1707_density_centroid_1= series_train_S1707_density[1].inverse_transform(S1707_density_centroid.reshape((len(S1707_density_centroid), 1)))
#S59
S59_density_centroid=centroids[1][:,8]
S59_density_centroid_1= series_train_S59_density[1].inverse_transform(S59_density_centroid.reshape((len(S59_density_centroid), 1)))
#R130
R130_density_centroid=centroids[1][:,9]
R130_density_centroid_1= series_train_R130_density[1].inverse_transform(R130_density_centroid.reshape((len(R130_density_centroid), 1)))
#R171
R171_density_centroid=centroids[1][:,10]
R171_density_centroid_1= series_train_R171_density[1].inverse_transform(R171_density_centroid.reshape((len(R171_density_centroid), 1)))
#S60
S60_density_centroid=centroids[1][:,11]
S60_density_centroid_1= series_train_S60_density[1].inverse_transform(S60_density_centroid.reshape((len(S60_density_centroid), 1)))
#S61
S61_density_centroid=centroids[1][:,12]
S61_density_centroid_1= series_train_S61_density[1].inverse_transform(S61_density_centroid.reshape((len(S61_density_centroid), 1)))

#third cluster k=2
#S54
S54_density_centroid=centroids[2][:,0]
S54_density_centroid_2= series_train_S54_density[1].inverse_transform(S54_density_centroid.reshape((len(S54_density_centroid), 1)))
#S1706
S1706_density_centroid=centroids[2][:,1]
S1706_density_centroid_2= series_train_S1706_density[1].inverse_transform(S1706_density_centroid.reshape((len(S1706_density_centroid), 1)))
#R169
R169_density_centroid=centroids[2][:,2]
R169_density_centroid_2= series_train_R169_density[1].inverse_transform(R169_density_centroid.reshape((len(R169_density_centroid), 1)))
#S56
S56_density_centroid=centroids[2][:,3]
S56_density_centroid_2= series_train_S56_density[1].inverse_transform(S56_density_centroid.reshape((len(S56_density_centroid), 1)))
#R129
R129_density_centroid=centroids[2][:,4]
R129_density_centroid_2= series_train_R129_density[1].inverse_transform(R129_density_centroid.reshape((len(R129_density_centroid), 1)))
#S57
S57_density_centroid=centroids[2][:,5]
S57_density_centroid_2= series_train_S57_density[1].inverse_transform(S57_density_centroid.reshape((len(S57_density_centroid), 1)))
#R170
R170_density_centroid=centroids[2][:,6]
R170_density_centroid_2= series_train_R170_density[1].inverse_transform(R170_density_centroid.reshape((len(R170_density_centroid), 1)))
#S1707
S1707_density_centroid=centroids[2][:,7]
S1707_density_centroid_2= series_train_S1707_density[1].inverse_transform(S1707_density_centroid.reshape((len(S1707_density_centroid), 1)))
#S59
S59_density_centroid=centroids[2][:,8]
S59_density_centroid_2= series_train_S59_density[1].inverse_transform(S59_density_centroid.reshape((len(S59_density_centroid), 1)))
#R130
R130_density_centroid=centroids[2][:,9]
R130_density_centroid_2= series_train_R130_density[1].inverse_transform(R130_density_centroid.reshape((len(R130_density_centroid), 1)))
#R171
R171_density_centroid=centroids[2][:,10]
R171_density_centroid_2= series_train_R171_density[1].inverse_transform(R171_density_centroid.reshape((len(R171_density_centroid), 1)))
#S60
S60_density_centroid=centroids[2][:,11]
S60_density_centroid_2= series_train_S60_density[1].inverse_transform(S60_density_centroid.reshape((len(S60_density_centroid), 1)))
#S61
S61_density_centroid=centroids[2][:,12]
S61_density_centroid_2= series_train_S61_density[1].inverse_transform(S61_density_centroid.reshape((len(S61_density_centroid), 1)))

#fourth cluster k=3
#S54
S54_density_centroid=centroids[3][:,0]
S54_density_centroid_3= series_train_S54_density[1].inverse_transform(S54_density_centroid.reshape((len(S54_density_centroid), 1)))
#S1706
S1706_density_centroid=centroids[3][:,1]
S1706_density_centroid_3= series_train_S1706_density[1].inverse_transform(S1706_density_centroid.reshape((len(S1706_density_centroid), 1)))
#R169
R169_density_centroid=centroids[3][:,2]
R169_density_centroid_3= series_train_R169_density[1].inverse_transform(R169_density_centroid.reshape((len(R169_density_centroid), 1)))
#S56
S56_density_centroid=centroids[3][:,3]
S56_density_centroid_3= series_train_S56_density[1].inverse_transform(S56_density_centroid.reshape((len(S56_density_centroid), 1)))
#R129
R129_density_centroid=centroids[3][:,4]
R129_density_centroid_3= series_train_R129_density[1].inverse_transform(R129_density_centroid.reshape((len(R129_density_centroid), 1)))
#S57
S57_density_centroid=centroids[3][:,5]
S57_density_centroid_3= series_train_S57_density[1].inverse_transform(S57_density_centroid.reshape((len(S57_density_centroid), 1)))
#R170
R170_density_centroid=centroids[3][:,6]
R170_density_centroid_3= series_train_R170_density[1].inverse_transform(R170_density_centroid.reshape((len(R170_density_centroid), 1)))
#S1707
S1707_density_centroid=centroids[3][:,7]
S1707_density_centroid_3= series_train_S1707_density[1].inverse_transform(S1707_density_centroid.reshape((len(S1707_density_centroid), 1)))
#S59
S59_density_centroid=centroids[3][:,8]
S59_density_centroid_3= series_train_S59_density[1].inverse_transform(S59_density_centroid.reshape((len(S59_density_centroid), 1)))
#R130
R130_density_centroid=centroids[3][:,9]
R130_density_centroid_3= series_train_R130_density[1].inverse_transform(R130_density_centroid.reshape((len(R130_density_centroid), 1)))
#R171
R171_density_centroid=centroids[3][:,10]
R171_density_centroid_3= series_train_R171_density[1].inverse_transform(R171_density_centroid.reshape((len(R171_density_centroid), 1)))
#S60
S60_density_centroid=centroids[3][:,11]
S60_density_centroid_3= series_train_S60_density[1].inverse_transform(S60_density_centroid.reshape((len(S60_density_centroid), 1)))
#S61
S61_density_centroid=centroids[3][:,12]
S61_density_centroid_3= series_train_S61_density[1].inverse_transform(S61_density_centroid.reshape((len(S61_density_centroid), 1)))

#fifth cluster k=4
#S54
S54_density_centroid=centroids[4][:,0]
S54_density_centroid_4= series_train_S54_density[1].inverse_transform(S54_density_centroid.reshape((len(S54_density_centroid), 1)))
#S1706
S1706_density_centroid=centroids[4][:,1]
S1706_density_centroid_4= series_train_S1706_density[1].inverse_transform(S1706_density_centroid.reshape((len(S1706_density_centroid), 1)))
#R169
R169_density_centroid=centroids[4][:,2]
R169_density_centroid_4= series_train_R169_density[1].inverse_transform(R169_density_centroid.reshape((len(R169_density_centroid), 1)))
#S56
S56_density_centroid=centroids[4][:,3]
S56_density_centroid_4= series_train_S56_density[1].inverse_transform(S56_density_centroid.reshape((len(S56_density_centroid), 1)))
#R129
R129_density_centroid=centroids[4][:,4]
R129_density_centroid_4= series_train_R129_density[1].inverse_transform(R129_density_centroid.reshape((len(R129_density_centroid), 1)))
#S57
S57_density_centroid=centroids[4][:,5]
S57_density_centroid_4= series_train_S57_density[1].inverse_transform(S57_density_centroid.reshape((len(S57_density_centroid), 1)))
#R170
R170_density_centroid=centroids[4][:,6]
R170_density_centroid_4= series_train_R170_density[1].inverse_transform(R170_density_centroid.reshape((len(R170_density_centroid), 1)))
#S1707
S1707_density_centroid=centroids[4][:,7]
S1707_density_centroid_4= series_train_S1707_density[1].inverse_transform(S1707_density_centroid.reshape((len(S1707_density_centroid), 1)))
#S59
S59_density_centroid=centroids[4][:,8]
S59_density_centroid_4= series_train_S59_density[1].inverse_transform(S59_density_centroid.reshape((len(S59_density_centroid), 1)))
#R130
R130_density_centroid=centroids[4][:,9]
R130_density_centroid_4= series_train_R130_density[1].inverse_transform(R130_density_centroid.reshape((len(R130_density_centroid), 1)))
#R171
R171_density_centroid=centroids[4][:,10]
R171_density_centroid_4= series_train_R171_density[1].inverse_transform(R171_density_centroid.reshape((len(R171_density_centroid), 1)))
#S60
S60_density_centroid=centroids[4][:,11]
S60_density_centroid_4= series_train_S60_density[1].inverse_transform(S60_density_centroid.reshape((len(S60_density_centroid), 1)))
#S61
S61_density_centroid=centroids[4][:,12]
S61_density_centroid_4= series_train_S61_density[1].inverse_transform(S61_density_centroid.reshape((len(S61_density_centroid), 1)))

#save centroids of the cluster 
columns = ['S54 density (veh/km)','S1706 density (veh/km)', 'R169 density (veh/km)','S56 density (veh/km)','R129 density (veh/km)', 'S57 density (veh/km)','R170 density (veh/km)','S1707 density (veh/km)', 'S59 density (veh/km)','R130 density (veh/km)','R171 density (veh/km)', 'S60 density (veh/km)','S61 density (veh/km)']
index=pd.date_range("5:00", periods=180, freq="6min")
index
df_0 = pd.DataFrame(index=index.time, columns=columns)
df_0['S54 density (veh/km)']=S54_density_centroid_0
df_0['S1706 density (veh/km)']=S1706_density_centroid_0
df_0['R169 density (veh/km)']=R169_density_centroid_0
df_0['S56 density (veh/km)']=S56_density_centroid_0
df_0['R129 density (veh/km)']=R129_density_centroid_0
df_0['S57 density (veh/km)']=S57_density_centroid_0
df_0['R170 density (veh/km)']=R170_density_centroid_0
df_0['S1707 density (veh/km)']=S1707_density_centroid_0
df_0['S59 density (veh/km)']=S59_density_centroid_0
df_0['R130 density (veh/km)']=R130_density_centroid_0
df_0['R171 density (veh/km)']=R171_density_centroid_0
df_0['S60 density (veh/km)']=S60_density_centroid_0
df_0['S61 density (veh/km)']=S61_density_centroid_0

df_0

df_1 = pd.DataFrame(index=index.time, columns=columns)
df_1['S54 density (veh/km)']=S54_density_centroid_1
df_1['S1706 density (veh/km)']=S1706_density_centroid_1
df_1['R169 density (veh/km)']=R169_density_centroid_1
df_1['S56 density (veh/km)']=S56_density_centroid_1
df_1['R129 density (veh/km)']=R129_density_centroid_1
df_1['S57 density (veh/km)']=S57_density_centroid_1
df_1['R170 density (veh/km)']=R170_density_centroid_1
df_1['S1707 density (veh/km)']=S1707_density_centroid_1
df_1['S59 density (veh/km)']=S59_density_centroid_1
df_1['R130 density (veh/km)']=R130_density_centroid_1
df_1['R171 density (veh/km)']=R171_density_centroid_1
df_1['S60 density (veh/km)']=S60_density_centroid_1
df_1['S61 density (veh/km)']=S61_density_centroid_1
df_1

df_2 = pd.DataFrame(index=index.time, columns=columns)
df_2['S54 density (veh/km)']=S54_density_centroid_2
df_2['S1706 density (veh/km)']=S1706_density_centroid_2
df_2['R169 density (veh/km)']=R169_density_centroid_2
df_2['S56 density (veh/km)']=S56_density_centroid_2
df_2['R129 density (veh/km)']=R129_density_centroid_2
df_2['S57 density (veh/km)']=S57_density_centroid_2
df_2['R170 density (veh/km)']=R170_density_centroid_2
df_2['S1707 density (veh/km)']=S1707_density_centroid_2
df_2['S59 density (veh/km)']=S59_density_centroid_2
df_2['R130 density (veh/km)']=R130_density_centroid_2
df_2['R171 density (veh/km)']=R171_density_centroid_2
df_2['S60 density (veh/km)']=S60_density_centroid_2
df_2['S61 density (veh/km)']=S61_density_centroid_2
df_2

df_3= pd.DataFrame(index=index.time, columns=columns)
df_3['S54 density (veh/km)']=S54_density_centroid_3
df_3['S1706 density (veh/km)']=S1706_density_centroid_3
df_3['R169 density (veh/km)']=R169_density_centroid_3
df_3['S56 density (veh/km)']=S56_density_centroid_3
df_3['R129 density (veh/km)']=R129_density_centroid_3
df_3['S57 density (veh/km)']=S57_density_centroid_3
df_3['R170 density (veh/km)']=R170_density_centroid_3
df_3['S1707 density (veh/km)']=S1707_density_centroid_3
df_3['S59 density (veh/km)']=S59_density_centroid_3
df_3['R130 density (veh/km)']=R130_density_centroid_3
df_3['R171 density (veh/km)']=R171_density_centroid_3
df_3['S60 density (veh/km)']=S60_density_centroid_3
df_3['S61 density (veh/km)']=S61_density_centroid_3
df_3

df_4= pd.DataFrame(index=index.time, columns=columns)
df_4['S54 density (veh/km)']=S54_density_centroid_4
df_4['S1706 density (veh/km)']=S1706_density_centroid_4
df_4['R169 density (veh/km)']=R169_density_centroid_4
df_4['S56 density (veh/km)']=S56_density_centroid_4
df_4['R129 density (veh/km)']=R129_density_centroid_4
df_4['S57 density (veh/km)']=S57_density_centroid_4
df_4['R170 density (veh/km)']=R170_density_centroid_4
df_4['S1707 density (veh/km)']=S1707_density_centroid_4
df_4['S59 density (veh/km)']=S59_density_centroid_4
df_4['R130 density (veh/km)']=R130_density_centroid_4
df_4['R171 density (veh/km)']=R171_density_centroid_4
df_4['S60 density (veh/km)']=S60_density_centroid_4
df_4['S61 density (veh/km)']=S61_density_centroid_4
df_4
# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter('/Users/nronzoni/Desktop/TrafficData Minnesota/Scenario8.xlsx', engine='xlsxwriter')

# Write each dataframe to a different worksheet.
df_0.to_excel(writer, sheet_name='k=0')
df_1.to_excel(writer, sheet_name='k=1')
df_2.to_excel(writer, sheet_name='k=2')
df_3.to_excel(writer, sheet_name='k=3')
df_4.to_excel(writer, sheet_name='k=4')
# Close the Pandas Excel writer and output the Excel file.
writer.save()

# Scenario 9
#first cluster k=0
#S54
S54_density_centroid=centroids[0][:,0]
S54_density_centroid_0= series_train_S54_density[1].inverse_transform(S54_density_centroid.reshape((len(S54_density_centroid), 1)))
#S1706
S1706_density_centroid=centroids[0][:,1]
S1706_density_centroid_0= series_train_S1706_density[1].inverse_transform(S1706_density_centroid.reshape((len(S1706_density_centroid), 1)))

#S56
S56_density_centroid=centroids[0][:,2]
S56_density_centroid_0= series_train_S56_density[1].inverse_transform(S56_density_centroid.reshape((len(S56_density_centroid), 1)))

#S57
S57_density_centroid=centroids[0][:,3]
S57_density_centroid_0= series_train_S57_density[1].inverse_transform(S57_density_centroid.reshape((len(S57_density_centroid), 1)))

#S1707
S1707_density_centroid=centroids[0][:,4]
S1707_density_centroid_0= series_train_S1707_density[1].inverse_transform(S1707_density_centroid.reshape((len(S1707_density_centroid), 1)))
#S59
S59_density_centroid=centroids[0][:,5]
S59_density_centroid_0= series_train_S59_density[1].inverse_transform(S59_density_centroid.reshape((len(S59_density_centroid), 1)))

#S60
S60_density_centroid=centroids[0][:,6]
S60_density_centroid_0= series_train_S60_density[1].inverse_transform(S60_density_centroid.reshape((len(S60_density_centroid), 1)))
#S61
S61_density_centroid=centroids[0][:,7]
S61_density_centroid_0= series_train_S61_density[1].inverse_transform(S61_density_centroid.reshape((len(S61_density_centroid), 1)))

#second cluster k=1
#S54
S54_density_centroid=centroids[1][:,0]
S54_density_centroid_1= series_train_S54_density[1].inverse_transform(S54_density_centroid.reshape((len(S54_density_centroid), 1)))
#S1706
S1706_density_centroid=centroids[1][:,1]
S1706_density_centroid_1= series_train_S1706_density[1].inverse_transform(S1706_density_centroid.reshape((len(S1706_density_centroid), 1)))

#S56
S56_density_centroid=centroids[1][:,2]
S56_density_centroid_1= series_train_S56_density[1].inverse_transform(S56_density_centroid.reshape((len(S56_density_centroid), 1)))

#S57
S57_density_centroid=centroids[1][:,3]
S57_density_centroid_1= series_train_S57_density[1].inverse_transform(S57_density_centroid.reshape((len(S57_density_centroid), 1)))

#S1707
S1707_density_centroid=centroids[1][:,4]
S1707_density_centroid_1= series_train_S1707_density[1].inverse_transform(S1707_density_centroid.reshape((len(S1707_density_centroid), 1)))
#S59
S59_density_centroid=centroids[1][:,5]
S59_density_centroid_1= series_train_S59_density[1].inverse_transform(S59_density_centroid.reshape((len(S59_density_centroid), 1)))

#S60
S60_density_centroid=centroids[1][:,6]
S60_density_centroid_1= series_train_S60_density[1].inverse_transform(S60_density_centroid.reshape((len(S60_density_centroid), 1)))
#S61
S61_density_centroid=centroids[1][:,7]
S61_density_centroid_1= series_train_S61_density[1].inverse_transform(S61_density_centroid.reshape((len(S61_density_centroid), 1)))

#third cluster k=2
#S54
S54_density_centroid=centroids[2][:,0]
S54_density_centroid_2= series_train_S54_density[1].inverse_transform(S54_density_centroid.reshape((len(S54_density_centroid), 1)))
#S1706
S1706_density_centroid=centroids[2][:,1]
S1706_density_centroid_2= series_train_S1706_density[1].inverse_transform(S1706_density_centroid.reshape((len(S1706_density_centroid), 1)))

#S56
S56_density_centroid=centroids[2][:,2]
S56_density_centroid_2= series_train_S56_density[1].inverse_transform(S56_density_centroid.reshape((len(S56_density_centroid), 1)))

#S57
S57_density_centroid=centroids[2][:,3]
S57_density_centroid_2= series_train_S57_density[1].inverse_transform(S57_density_centroid.reshape((len(S57_density_centroid), 1)))

#S1707
S1707_density_centroid=centroids[2][:,4]
S1707_density_centroid_2= series_train_S1707_density[1].inverse_transform(S1707_density_centroid.reshape((len(S1707_density_centroid), 1)))
#S59
S59_density_centroid=centroids[2][:,5]
S59_density_centroid_2= series_train_S59_density[1].inverse_transform(S59_density_centroid.reshape((len(S59_density_centroid), 1)))

#S60
S60_density_centroid=centroids[2][:,6]
S60_density_centroid_2= series_train_S60_density[1].inverse_transform(S60_density_centroid.reshape((len(S60_density_centroid), 1)))
#S61
S61_density_centroid=centroids[2][:,7]
S61_density_centroid_2= series_train_S61_density[1].inverse_transform(S61_density_centroid.reshape((len(S61_density_centroid), 1)))

#fourth cluster k=3
#S54
S54_density_centroid=centroids[3][:,0]
S54_density_centroid_3= series_train_S54_density[1].inverse_transform(S54_density_centroid.reshape((len(S54_density_centroid), 1)))
#S1706
S1706_density_centroid=centroids[3][:,1]
S1706_density_centroid_3= series_train_S1706_density[1].inverse_transform(S1706_density_centroid.reshape((len(S1706_density_centroid), 1)))
#S56
S56_density_centroid=centroids[3][:,2]
S56_density_centroid_3= series_train_S56_density[1].inverse_transform(S56_density_centroid.reshape((len(S56_density_centroid), 1)))
#S57
S57_density_centroid=centroids[3][:,3]
S57_density_centroid_3= series_train_S57_density[1].inverse_transform(S57_density_centroid.reshape((len(S57_density_centroid), 1)))

#S1707
S1707_density_centroid=centroids[3][:,4]
S1707_density_centroid_3= series_train_S1707_density[1].inverse_transform(S1707_density_centroid.reshape((len(S1707_density_centroid), 1)))
#S59
S59_density_centroid=centroids[3][:,5]
S59_density_centroid_3= series_train_S59_density[1].inverse_transform(S59_density_centroid.reshape((len(S59_density_centroid), 1)))

#S60
S60_density_centroid=centroids[3][:,6]
S60_density_centroid_3= series_train_S60_density[1].inverse_transform(S60_density_centroid.reshape((len(S60_density_centroid), 1)))
#S61
S61_density_centroid=centroids[3][:,7]
S61_density_centroid_3= series_train_S61_density[1].inverse_transform(S61_density_centroid.reshape((len(S61_density_centroid), 1)))

#fifth cluster k=4
#S54
S54_density_centroid=centroids[4][:,0]
S54_density_centroid_4= series_train_S54_density[1].inverse_transform(S54_density_centroid.reshape((len(S54_density_centroid), 1)))
#S1706
S1706_density_centroid=centroids[4][:,1]
S1706_density_centroid_4= series_train_S1706_density[1].inverse_transform(S1706_density_centroid.reshape((len(S1706_density_centroid), 1)))

#S56
S56_density_centroid=centroids[4][:,2]
S56_density_centroid_4= series_train_S56_density[1].inverse_transform(S56_density_centroid.reshape((len(S56_density_centroid), 1)))

#S57
S57_density_centroid=centroids[4][:,3]
S57_density_centroid_4= series_train_S57_density[1].inverse_transform(S57_density_centroid.reshape((len(S57_density_centroid), 1)))

#S1707
S1707_density_centroid=centroids[4][:,4]
S1707_density_centroid_4= series_train_S1707_density[1].inverse_transform(S1707_density_centroid.reshape((len(S1707_density_centroid), 1)))
#S59
S59_density_centroid=centroids[4][:,5]
S59_density_centroid_4= series_train_S59_density[1].inverse_transform(S59_density_centroid.reshape((len(S59_density_centroid), 1)))

#S60
S60_density_centroid=centroids[4][:,6]
S60_density_centroid_4= series_train_S60_density[1].inverse_transform(S60_density_centroid.reshape((len(S60_density_centroid), 1)))
#S61
S61_density_centroid=centroids[4][:,7]
S61_density_centroid_4= series_train_S61_density[1].inverse_transform(S61_density_centroid.reshape((len(S61_density_centroid), 1)))

#save centroids of the cluster 
columns = ['S54 density (veh/km)','S1706 density (veh/km)','S56 density (veh/km)', 'S57 density (veh/km)','S1707 density (veh/km)', 'S59 density (veh/km)', 'S60 density (veh/km)','S61 density (veh/km)']
index=pd.date_range("5:00", periods=180, freq="6min")
index
df_0 = pd.DataFrame(index=index.time, columns=columns)
df_0['S54 density (veh/km)']=S54_density_centroid_0
df_0['S1706 density (veh/km)']=S1706_density_centroid_0
df_0['S56 density (veh/km)']=S56_density_centroid_0
df_0['S57 density (veh/km)']=S57_density_centroid_0
df_0['S1707 density (veh/km)']=S1707_density_centroid_0
df_0['S59 density (veh/km)']=S59_density_centroid_0
df_0['S60 density (veh/km)']=S60_density_centroid_0
df_0['S61 density (veh/km)']=S61_density_centroid_0

df_0

df_1 = pd.DataFrame(index=index.time, columns=columns)
df_1['S54 density (veh/km)']=S54_density_centroid_1
df_1['S1706 density (veh/km)']=S1706_density_centroid_1
df_1['S56 density (veh/km)']=S56_density_centroid_1
df_1['S57 density (veh/km)']=S57_density_centroid_1
df_1['S1707 density (veh/km)']=S1707_density_centroid_1
df_1['S59 density (veh/km)']=S59_density_centroid_1
df_1['S60 density (veh/km)']=S60_density_centroid_1
df_1['S61 density (veh/km)']=S61_density_centroid_1
df_1

df_2 = pd.DataFrame(index=index.time, columns=columns)
df_2['S54 density (veh/km)']=S54_density_centroid_2
df_2['S1706 density (veh/km)']=S1706_density_centroid_2
df_2['S56 density (veh/km)']=S56_density_centroid_2
df_2['S57 density (veh/km)']=S57_density_centroid_2
df_2['S1707 density (veh/km)']=S1707_density_centroid_2
df_2['S59 density (veh/km)']=S59_density_centroid_2
df_2['S60 density (veh/km)']=S60_density_centroid_2
df_2['S61 density (veh/km)']=S61_density_centroid_2
df_2

df_3= pd.DataFrame(index=index.time, columns=columns)
df_3['S54 density (veh/km)']=S54_density_centroid_3
df_3['S1706 density (veh/km)']=S1706_density_centroid_3
df_3['S56 density (veh/km)']=S56_density_centroid_3
df_3['S57 density (veh/km)']=S57_density_centroid_3
df_3['S1707 density (veh/km)']=S1707_density_centroid_3
df_3['S59 density (veh/km)']=S59_density_centroid_3
df_3['S60 density (veh/km)']=S60_density_centroid_3
df_3['S61 density (veh/km)']=S61_density_centroid_3
df_3

df_4= pd.DataFrame(index=index.time, columns=columns)
df_4['S54 density (veh/km)']=S54_density_centroid_4
df_4['S1706 density (veh/km)']=S1706_density_centroid_4
df_4['S56 density (veh/km)']=S56_density_centroid_4
df_4['S57 density (veh/km)']=S57_density_centroid_4
df_4['S1707 density (veh/km)']=S1707_density_centroid_4
df_4['S59 density (veh/km)']=S59_density_centroid_4
df_4['S60 density (veh/km)']=S60_density_centroid_4
df_4['S61 density (veh/km)']=S61_density_centroid_4
df_4
# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter('/Users/nronzoni/Desktop/TrafficData Minnesota/Scenario9.xlsx', engine='xlsxwriter')

# Write each dataframe to a different worksheet.
df_0.to_excel(writer, sheet_name='k=0')
df_1.to_excel(writer, sheet_name='k=1')
df_2.to_excel(writer, sheet_name='k=2')
df_3.to_excel(writer, sheet_name='k=3')
df_4.to_excel(writer, sheet_name='k=4')
# Close the Pandas Excel writer and output the Excel file.
writer.save()


#Scenario 10 
#first cluster k=0
#S54
S54_flow_centroid=centroids[0][:,0]
S54_flow_centroid_0= series_train_S54_flow[1].inverse_transform(S54_flow_centroid.reshape((len(S54_flow_centroid), 1)))
S54_density_centroid=centroids[0][:,1]
S54_density_centroid_0= series_train_S54_density[1].inverse_transform(S54_density_centroid.reshape((len(S54_density_centroid), 1)))
#S1706
S1706_flow_centroid=centroids[0][:,2]
S1706_flow_centroid_0= series_train_S1706_flow[1].inverse_transform(S1706_flow_centroid.reshape((len(S1706_flow_centroid), 1)))
S1706_density_centroid=centroids[0][:,3]
S1706_density_centroid_0= series_train_S1706_density[1].inverse_transform(S1706_density_centroid.reshape((len(S1706_density_centroid), 1)))
#R169
R169_flow_centroid=centroids[0][:,4]
R169_flow_centroid_0= series_train_R169_flow[1].inverse_transform(R169_flow_centroid.reshape((len(R169_flow_centroid), 1)))
R169_density_centroid=centroids[0][:,5]
R169_density_centroid_0= series_train_R169_density[1].inverse_transform(R169_density_centroid.reshape((len(R169_density_centroid), 1)))
#S56
S56_flow_centroid=centroids[0][:,6]
S56_flow_centroid_0= series_train_S56_flow[1].inverse_transform(S56_flow_centroid.reshape((len(S56_flow_centroid), 1)))
S56_density_centroid=centroids[0][:,7]
S56_density_centroid_0= series_train_S56_density[1].inverse_transform(S56_density_centroid.reshape((len(S56_density_centroid), 1)))
#R129
R129_flow_centroid=centroids[0][:,8]
R129_flow_centroid_0= series_train_R129_flow[1].inverse_transform(R129_flow_centroid.reshape((len(R129_flow_centroid), 1)))
R129_density_centroid=centroids[0][:,9]
R129_density_centroid_0= series_train_R129_density[1].inverse_transform(R129_density_centroid.reshape((len(R129_density_centroid), 1)))
#S57
S57_flow_centroid=centroids[0][:,10]
S57_flow_centroid_0= series_train_S57_flow[1].inverse_transform(S57_flow_centroid.reshape((len(S57_flow_centroid), 1)))
S57_density_centroid=centroids[0][:,11]
S57_density_centroid_0= series_train_S57_density[1].inverse_transform(S57_density_centroid.reshape((len(S57_density_centroid), 1)))
#R170
R170_flow_centroid=centroids[0][:,12]
R170_flow_centroid_0= series_train_R170_flow[1].inverse_transform(R170_flow_centroid.reshape((len(R170_flow_centroid), 1)))
R170_density_centroid=centroids[0][:,13]
R170_density_centroid_0= series_train_R170_density[1].inverse_transform(R170_density_centroid.reshape((len(R170_density_centroid), 1)))
#S1707
S1707_flow_centroid=centroids[0][:,14]
S1707_flow_centroid_0= series_train_S1707_flow[1].inverse_transform(S1707_flow_centroid.reshape((len(S1707_flow_centroid), 1)))
S1707_density_centroid=centroids[0][:,15]
S1707_density_centroid_0= series_train_S1707_density[1].inverse_transform(S1707_density_centroid.reshape((len(S1707_density_centroid), 1)))
#S59
S59_flow_centroid=centroids[0][:,16]
S59_flow_centroid_0= series_train_S59_flow[1].inverse_transform(S59_flow_centroid.reshape((len(S59_flow_centroid), 1)))
S59_density_centroid=centroids[0][:,17]
S59_density_centroid_0= series_train_S59_density[1].inverse_transform(S59_density_centroid.reshape((len(S59_density_centroid), 1)))
#R130
R130_flow_centroid=centroids[0][:,18]
R130_flow_centroid_0= series_train_R130_flow[1].inverse_transform(R130_flow_centroid.reshape((len(R130_flow_centroid), 1)))
R130_density_centroid=centroids[0][:,19]
R130_density_centroid_0= series_train_R130_density[1].inverse_transform(R130_density_centroid.reshape((len(R130_density_centroid), 1)))
#R171
R171_flow_centroid=centroids[0][:,20]
R171_flow_centroid_0= series_train_R171_flow[1].inverse_transform(R171_flow_centroid.reshape((len(R171_flow_centroid), 1)))
R171_density_centroid=centroids[0][:,21]
R171_density_centroid_0= series_train_R171_density[1].inverse_transform(R171_density_centroid.reshape((len(R171_density_centroid), 1)))
#S60
S60_flow_centroid=centroids[0][:,22]
S60_flow_centroid_0= series_train_S60_flow[1].inverse_transform(S60_flow_centroid.reshape((len(S60_flow_centroid), 1)))
S60_density_centroid=centroids[0][:,23]
S60_density_centroid_0= series_train_S60_density[1].inverse_transform(S60_density_centroid.reshape((len(S60_density_centroid), 1)))
#S61
S61_flow_centroid=centroids[0][:,24]
S61_flow_centroid_0= series_train_S61_flow[1].inverse_transform(S61_flow_centroid.reshape((len(S61_flow_centroid), 1)))
S61_density_centroid=centroids[0][:,25]
S61_density_centroid_0= series_train_S61_density[1].inverse_transform(S61_density_centroid.reshape((len(S61_density_centroid), 1)))

#second cluster k=1
#S54
S54_flow_centroid=centroids[1][:,0]
S54_flow_centroid_1= series_train_S54_flow[1].inverse_transform(S54_flow_centroid.reshape((len(S54_flow_centroid), 1)))
S54_density_centroid=centroids[1][:,1]
S54_density_centroid_1= series_train_S54_density[1].inverse_transform(S54_density_centroid.reshape((len(S54_density_centroid), 1)))
#S1706
S1706_flow_centroid=centroids[1][:,2]
S1706_flow_centroid_1= series_train_S1706_flow[1].inverse_transform(S1706_flow_centroid.reshape((len(S1706_flow_centroid), 1)))
S1706_density_centroid=centroids[1][:,3]
S1706_density_centroid_1= series_train_S1706_density[1].inverse_transform(S1706_density_centroid.reshape((len(S1706_density_centroid), 1)))
#R169
R169_flow_centroid=centroids[1][:,4]
R169_flow_centroid_1= series_train_R169_flow[1].inverse_transform(R169_flow_centroid.reshape((len(R169_flow_centroid), 1)))
R169_density_centroid=centroids[1][:,5]
R169_density_centroid_1= series_train_R169_density[1].inverse_transform(R169_density_centroid.reshape((len(R169_density_centroid), 1)))
#S56
S56_flow_centroid=centroids[1][:,6]
S56_flow_centroid_1= series_train_S56_flow[1].inverse_transform(S56_flow_centroid.reshape((len(S56_flow_centroid), 1)))
S56_density_centroid=centroids[1][:,7]
S56_density_centroid_1= series_train_S56_density[1].inverse_transform(S56_density_centroid.reshape((len(S56_density_centroid), 1)))
#R129
R129_flow_centroid=centroids[1][:,8]
R129_flow_centroid_1= series_train_R129_flow[1].inverse_transform(R129_flow_centroid.reshape((len(R129_flow_centroid), 1)))
R129_density_centroid=centroids[1][:,9]
R129_density_centroid_1= series_train_R129_density[1].inverse_transform(R129_density_centroid.reshape((len(R129_density_centroid), 1)))
#S57
S57_flow_centroid=centroids[1][:,10]
S57_flow_centroid_1= series_train_S57_flow[1].inverse_transform(S57_flow_centroid.reshape((len(S57_flow_centroid), 1)))
S57_density_centroid=centroids[1][:,11]
S57_density_centroid_1= series_train_S57_density[1].inverse_transform(S57_density_centroid.reshape((len(S57_density_centroid), 1)))
#R170
R170_flow_centroid=centroids[1][:,12]
R170_flow_centroid_1= series_train_R170_flow[1].inverse_transform(R170_flow_centroid.reshape((len(R170_flow_centroid), 1)))
R170_density_centroid=centroids[1][:,13]
R170_density_centroid_1= series_train_R170_density[1].inverse_transform(R170_density_centroid.reshape((len(R170_density_centroid), 1)))
#S1707
S1707_flow_centroid=centroids[1][:,14]
S1707_flow_centroid_1= series_train_S1707_flow[1].inverse_transform(S1707_flow_centroid.reshape((len(S1707_flow_centroid), 1)))
S1707_density_centroid=centroids[1][:,15]
S1707_density_centroid_1= series_train_S1707_density[1].inverse_transform(S1707_density_centroid.reshape((len(S1707_density_centroid), 1)))
#S59
S59_flow_centroid=centroids[1][:,16]
S59_flow_centroid_1= series_train_S59_flow[1].inverse_transform(S59_flow_centroid.reshape((len(S59_flow_centroid), 1)))
S59_density_centroid=centroids[1][:,17]
S59_density_centroid_1= series_train_S59_density[1].inverse_transform(S59_density_centroid.reshape((len(S59_density_centroid), 1)))
#R130
R130_flow_centroid=centroids[1][:,18]
R130_flow_centroid_1= series_train_R130_flow[1].inverse_transform(R130_flow_centroid.reshape((len(R130_flow_centroid), 1)))
R130_density_centroid=centroids[1][:,19]
R130_density_centroid_1= series_train_R130_density[1].inverse_transform(R130_density_centroid.reshape((len(R130_density_centroid), 1)))
#R171
R171_flow_centroid=centroids[1][:,20]
R171_flow_centroid_1= series_train_R171_flow[1].inverse_transform(R171_flow_centroid.reshape((len(R171_flow_centroid), 1)))
R171_density_centroid=centroids[1][:,21]
R171_density_centroid_1= series_train_R171_density[1].inverse_transform(R171_density_centroid.reshape((len(R171_density_centroid), 1)))
#S60
S60_flow_centroid=centroids[1][:,22]
S60_flow_centroid_1= series_train_S60_flow[1].inverse_transform(S60_flow_centroid.reshape((len(S60_flow_centroid), 1)))
S60_density_centroid=centroids[1][:,23]
S60_density_centroid_1= series_train_S60_density[1].inverse_transform(S60_density_centroid.reshape((len(S60_density_centroid), 1)))
#S61
S61_flow_centroid=centroids[1][:,24]
S61_flow_centroid_1= series_train_S61_flow[1].inverse_transform(S61_flow_centroid.reshape((len(S61_flow_centroid), 1)))
S61_density_centroid=centroids[1][:,25]
S61_density_centroid_1= series_train_S61_density[1].inverse_transform(S61_density_centroid.reshape((len(S61_density_centroid), 1)))

#third cluster k=2
#S54
S54_flow_centroid=centroids[2][:,0]
S54_flow_centroid_2= series_train_S54_flow[1].inverse_transform(S54_flow_centroid.reshape((len(S54_flow_centroid), 1)))
S54_density_centroid=centroids[2][:,1]
S54_density_centroid_2= series_train_S54_density[1].inverse_transform(S54_density_centroid.reshape((len(S54_density_centroid), 1)))
#S1706
S1706_flow_centroid=centroids[2][:,2]
S1706_flow_centroid_2= series_train_S1706_flow[1].inverse_transform(S1706_flow_centroid.reshape((len(S1706_flow_centroid), 1)))
S1706_density_centroid=centroids[2][:,3]
S1706_density_centroid_2= series_train_S1706_density[1].inverse_transform(S1706_density_centroid.reshape((len(S1706_density_centroid), 1)))
#R169
R169_flow_centroid=centroids[2][:,4]
R169_flow_centroid_2= series_train_R169_flow[1].inverse_transform(R169_flow_centroid.reshape((len(R169_flow_centroid), 1)))
R169_density_centroid=centroids[2][:,5]
R169_density_centroid_2= series_train_R169_density[1].inverse_transform(R169_density_centroid.reshape((len(R169_density_centroid), 1)))
#S56
S56_flow_centroid=centroids[2][:,6]
S56_flow_centroid_2= series_train_S56_flow[1].inverse_transform(S56_flow_centroid.reshape((len(S56_flow_centroid), 1)))
S56_density_centroid=centroids[2][:,7]
S56_density_centroid_2= series_train_S56_density[1].inverse_transform(S56_density_centroid.reshape((len(S56_density_centroid), 1)))
#R129
R129_flow_centroid=centroids[2][:,8]
R129_flow_centroid_2= series_train_R129_flow[1].inverse_transform(R129_flow_centroid.reshape((len(R129_flow_centroid), 1)))
R129_density_centroid=centroids[2][:,9]
R129_density_centroid_2= series_train_R129_density[1].inverse_transform(R129_density_centroid.reshape((len(R129_density_centroid), 1)))
#S57
S57_flow_centroid=centroids[2][:,10]
S57_flow_centroid_2= series_train_S57_flow[1].inverse_transform(S57_flow_centroid.reshape((len(S57_flow_centroid), 1)))
S57_density_centroid=centroids[2][:,11]
S57_density_centroid_2= series_train_S57_density[1].inverse_transform(S57_density_centroid.reshape((len(S57_density_centroid), 1)))
#R170
R170_flow_centroid=centroids[2][:,12]
R170_flow_centroid_2= series_train_R170_flow[1].inverse_transform(R170_flow_centroid.reshape((len(R170_flow_centroid), 1)))
R170_density_centroid=centroids[2][:,13]
R170_density_centroid_2= series_train_R170_density[1].inverse_transform(R170_density_centroid.reshape((len(R170_density_centroid), 1)))
#S1707
S1707_flow_centroid=centroids[2][:,14]
S1707_flow_centroid_2= series_train_S1707_flow[1].inverse_transform(S1707_flow_centroid.reshape((len(S1707_flow_centroid), 1)))
S1707_density_centroid=centroids[2][:,15]
S1707_density_centroid_2= series_train_S1707_density[1].inverse_transform(S1707_density_centroid.reshape((len(S1707_density_centroid), 1)))
#S59
S59_flow_centroid=centroids[2][:,16]
S59_flow_centroid_2= series_train_S59_flow[1].inverse_transform(S59_flow_centroid.reshape((len(S59_flow_centroid), 1)))
S59_density_centroid=centroids[2][:,17]
S59_density_centroid_2= series_train_S59_density[1].inverse_transform(S59_density_centroid.reshape((len(S59_density_centroid), 1)))
#R130
R130_flow_centroid=centroids[2][:,18]
R130_flow_centroid_2= series_train_R130_flow[1].inverse_transform(R130_flow_centroid.reshape((len(R130_flow_centroid), 1)))
R130_density_centroid=centroids[2][:,19]
R130_density_centroid_2= series_train_R130_density[1].inverse_transform(R130_density_centroid.reshape((len(R130_density_centroid), 1)))
#R171
R171_flow_centroid=centroids[2][:,20]
R171_flow_centroid_2= series_train_R171_flow[1].inverse_transform(R171_flow_centroid.reshape((len(R171_flow_centroid), 1)))
R171_density_centroid=centroids[2][:,21]
R171_density_centroid_2= series_train_R171_density[1].inverse_transform(R171_density_centroid.reshape((len(R171_density_centroid), 1)))
#S60
S60_flow_centroid=centroids[2][:,22]
S60_flow_centroid_2= series_train_S60_flow[1].inverse_transform(S60_flow_centroid.reshape((len(S60_flow_centroid), 1)))
S60_density_centroid=centroids[2][:,23]
S60_density_centroid_2= series_train_S60_density[1].inverse_transform(S60_density_centroid.reshape((len(S60_density_centroid), 1)))
#S61
S61_flow_centroid=centroids[2][:,24]
S61_flow_centroid_2= series_train_S61_flow[1].inverse_transform(S61_flow_centroid.reshape((len(S61_flow_centroid), 1)))
S61_density_centroid=centroids[2][:,25]
S61_density_centroid_2= series_train_S61_density[1].inverse_transform(S61_density_centroid.reshape((len(S61_density_centroid), 1)))

#fourth cluster k=3
#S54
S54_flow_centroid=centroids[3][:,0]
S54_flow_centroid_3= series_train_S54_flow[1].inverse_transform(S54_flow_centroid.reshape((len(S54_flow_centroid), 1)))
S54_density_centroid=centroids[3][:,1]
S54_density_centroid_3= series_train_S54_density[1].inverse_transform(S54_density_centroid.reshape((len(S54_density_centroid), 1)))
#S1706
S1706_flow_centroid=centroids[3][:,2]
S1706_flow_centroid_3= series_train_S1706_flow[1].inverse_transform(S1706_flow_centroid.reshape((len(S1706_flow_centroid), 1)))
S1706_density_centroid=centroids[3][:,3]
S1706_density_centroid_3= series_train_S1706_density[1].inverse_transform(S1706_density_centroid.reshape((len(S1706_density_centroid), 1)))
#R169
R169_flow_centroid=centroids[3][:,4]
R169_flow_centroid_3= series_train_R169_flow[1].inverse_transform(R169_flow_centroid.reshape((len(R169_flow_centroid), 1)))
R169_density_centroid=centroids[3][:,5]
R169_density_centroid_3= series_train_R169_density[1].inverse_transform(R169_density_centroid.reshape((len(R169_density_centroid), 1)))
#S56
S56_flow_centroid=centroids[3][:,6]
S56_flow_centroid_3= series_train_S56_flow[1].inverse_transform(S56_flow_centroid.reshape((len(S56_flow_centroid), 1)))
S56_density_centroid=centroids[3][:,7]
S56_density_centroid_3= series_train_S56_density[1].inverse_transform(S56_density_centroid.reshape((len(S56_density_centroid), 1)))
#R129
R129_flow_centroid=centroids[3][:,8]
R129_flow_centroid_3= series_train_R129_flow[1].inverse_transform(R129_flow_centroid.reshape((len(R129_flow_centroid), 1)))
R129_density_centroid=centroids[3][:,9]
R129_density_centroid_3= series_train_R129_density[1].inverse_transform(R129_density_centroid.reshape((len(R129_density_centroid), 1)))
#S57
S57_flow_centroid=centroids[3][:,10]
S57_flow_centroid_3= series_train_S57_flow[1].inverse_transform(S57_flow_centroid.reshape((len(S57_flow_centroid), 1)))
S57_density_centroid=centroids[3][:,11]
S57_density_centroid_3= series_train_S57_density[1].inverse_transform(S57_density_centroid.reshape((len(S57_density_centroid), 1)))
#R170
R170_flow_centroid=centroids[3][:,12]
R170_flow_centroid_3= series_train_R170_flow[1].inverse_transform(R170_flow_centroid.reshape((len(R170_flow_centroid), 1)))
R170_density_centroid=centroids[3][:,13]
R170_density_centroid_3= series_train_R170_density[1].inverse_transform(R170_density_centroid.reshape((len(R170_density_centroid), 1)))
#S1707
S1707_flow_centroid=centroids[3][:,14]
S1707_flow_centroid_3= series_train_S1707_flow[1].inverse_transform(S1707_flow_centroid.reshape((len(S1707_flow_centroid), 1)))
S1707_density_centroid=centroids[3][:,15]
S1707_density_centroid_3= series_train_S1707_density[1].inverse_transform(S1707_density_centroid.reshape((len(S1707_density_centroid), 1)))
#S59
S59_flow_centroid=centroids[3][:,16]
S59_flow_centroid_3= series_train_S59_flow[1].inverse_transform(S59_flow_centroid.reshape((len(S59_flow_centroid), 1)))
S59_density_centroid=centroids[3][:,17]
S59_density_centroid_3= series_train_S59_density[1].inverse_transform(S59_density_centroid.reshape((len(S59_density_centroid), 1)))
#R130
R130_flow_centroid=centroids[3][:,18]
R130_flow_centroid_3= series_train_R130_flow[1].inverse_transform(R130_flow_centroid.reshape((len(R130_flow_centroid), 1)))
R130_density_centroid=centroids[3][:,19]
R130_density_centroid_3= series_train_R130_density[1].inverse_transform(R130_density_centroid.reshape((len(R130_density_centroid), 1)))
#R171
R171_flow_centroid=centroids[3][:,20]
R171_flow_centroid_3= series_train_R171_flow[1].inverse_transform(R171_flow_centroid.reshape((len(R171_flow_centroid), 1)))
R171_density_centroid=centroids[3][:,21]
R171_density_centroid_3= series_train_R171_density[1].inverse_transform(R171_density_centroid.reshape((len(R171_density_centroid), 1)))
#S60
S60_flow_centroid=centroids[3][:,22]
S60_flow_centroid_3= series_train_S60_flow[1].inverse_transform(S60_flow_centroid.reshape((len(S60_flow_centroid), 1)))
S60_density_centroid=centroids[3][:,23]
S60_density_centroid_3= series_train_S60_density[1].inverse_transform(S60_density_centroid.reshape((len(S60_density_centroid), 1)))
#S61
S61_flow_centroid=centroids[3][:,24]
S61_flow_centroid_3= series_train_S61_flow[1].inverse_transform(S61_flow_centroid.reshape((len(S61_flow_centroid), 1)))
S61_density_centroid=centroids[3][:,25]
S61_density_centroid_3= series_train_S61_density[1].inverse_transform(S61_density_centroid.reshape((len(S61_density_centroid), 1)))

#fifth cluster k=4
#S54
S54_flow_centroid=centroids[4][:,0]
S54_flow_centroid_4= series_train_S54_flow[1].inverse_transform(S54_flow_centroid.reshape((len(S54_flow_centroid), 1)))
S54_density_centroid=centroids[4][:,1]
S54_density_centroid_4= series_train_S54_density[1].inverse_transform(S54_density_centroid.reshape((len(S54_density_centroid), 1)))
#S1706
S1706_flow_centroid=centroids[4][:,2]
S1706_flow_centroid_4= series_train_S1706_flow[1].inverse_transform(S1706_flow_centroid.reshape((len(S1706_flow_centroid), 1)))
S1706_density_centroid=centroids[4][:,3]
S1706_density_centroid_4= series_train_S1706_density[1].inverse_transform(S1706_density_centroid.reshape((len(S1706_density_centroid), 1)))
#R169
R169_flow_centroid=centroids[4][:,4]
R169_flow_centroid_4= series_train_R169_flow[1].inverse_transform(R169_flow_centroid.reshape((len(R169_flow_centroid), 1)))
R169_density_centroid=centroids[4][:,5]
R169_density_centroid_4= series_train_R169_density[1].inverse_transform(R169_density_centroid.reshape((len(R169_density_centroid), 1)))
#S56
S56_flow_centroid=centroids[4][:,6]
S56_flow_centroid_4= series_train_S56_flow[1].inverse_transform(S56_flow_centroid.reshape((len(S56_flow_centroid), 1)))
S56_density_centroid=centroids[4][:,7]
S56_density_centroid_4= series_train_S56_density[1].inverse_transform(S56_density_centroid.reshape((len(S56_density_centroid), 1)))
#R129
R129_flow_centroid=centroids[4][:,8]
R129_flow_centroid_4= series_train_R129_flow[1].inverse_transform(R129_flow_centroid.reshape((len(R129_flow_centroid), 1)))
R129_density_centroid=centroids[4][:,9]
R129_density_centroid_4= series_train_R129_density[1].inverse_transform(R129_density_centroid.reshape((len(R129_density_centroid), 1)))
#S57
S57_flow_centroid=centroids[4][:,10]
S57_flow_centroid_4= series_train_S57_flow[1].inverse_transform(S57_flow_centroid.reshape((len(S57_flow_centroid), 1)))
S57_density_centroid=centroids[4][:,11]
S57_density_centroid_4= series_train_S57_density[1].inverse_transform(S57_density_centroid.reshape((len(S57_density_centroid), 1)))
#R170
R170_flow_centroid=centroids[4][:,12]
R170_flow_centroid_4= series_train_R170_flow[1].inverse_transform(R170_flow_centroid.reshape((len(R170_flow_centroid), 1)))
R170_density_centroid=centroids[4][:,13]
R170_density_centroid_4= series_train_R170_density[1].inverse_transform(R170_density_centroid.reshape((len(R170_density_centroid), 1)))
#S1707
S1707_flow_centroid=centroids[4][:,14]
S1707_flow_centroid_4= series_train_S1707_flow[1].inverse_transform(S1707_flow_centroid.reshape((len(S1707_flow_centroid), 1)))
S1707_density_centroid=centroids[4][:,15]
S1707_density_centroid_4= series_train_S1707_density[1].inverse_transform(S1707_density_centroid.reshape((len(S1707_density_centroid), 1)))
#S59
S59_flow_centroid=centroids[4][:,16]
S59_flow_centroid_4= series_train_S59_flow[1].inverse_transform(S59_flow_centroid.reshape((len(S59_flow_centroid), 1)))
S59_density_centroid=centroids[4][:,17]
S59_density_centroid_4= series_train_S59_density[1].inverse_transform(S59_density_centroid.reshape((len(S59_density_centroid), 1)))
#R130
R130_flow_centroid=centroids[4][:,18]
R130_flow_centroid_4= series_train_R130_flow[1].inverse_transform(R130_flow_centroid.reshape((len(R130_flow_centroid), 1)))
R130_density_centroid=centroids[4][:,19]
R130_density_centroid_4= series_train_R130_density[1].inverse_transform(R130_density_centroid.reshape((len(R130_density_centroid), 1)))
#R171
R171_flow_centroid=centroids[4][:,20]
R171_flow_centroid_4= series_train_R171_flow[1].inverse_transform(R171_flow_centroid.reshape((len(R171_flow_centroid), 1)))
R171_density_centroid=centroids[4][:,21]
R171_density_centroid_4= series_train_R171_density[1].inverse_transform(R171_density_centroid.reshape((len(R171_density_centroid), 1)))
#S60
S60_flow_centroid=centroids[4][:,22]
S60_flow_centroid_4= series_train_S60_flow[1].inverse_transform(S60_flow_centroid.reshape((len(S60_flow_centroid), 1)))
S60_density_centroid=centroids[4][:,23]
S60_density_centroid_4= series_train_S60_density[1].inverse_transform(S60_density_centroid.reshape((len(S60_density_centroid), 1)))
#S61
S61_flow_centroid=centroids[4][:,24]
S61_flow_centroid_4= series_train_S61_flow[1].inverse_transform(S61_flow_centroid.reshape((len(S61_flow_centroid), 1)))
S61_density_centroid=centroids[4][:,25]
S61_density_centroid_4= series_train_S61_density[1].inverse_transform(S61_density_centroid.reshape((len(S61_density_centroid), 1)))



#save centroids of the cluster 
columns = ['S54 flow (veh/h)','S54 density (veh/km)','S1706 flow (veh/h)','S1706 density (veh/km)', 'R169 flow (veh/h)','R169 density (veh/km)','S56 flow (veh/h)','S56 density (veh/km)','R129 flow (veh/h)','R129 density (veh/km)','S57 flow (veh/h)', 'S57 density (veh/km)','R170 flow (veh/h)','R170 density (veh/km)','S1707 flow (veh/h)','S1707 density (veh/km)','S59 flow (veh/h)', 'S59 density (veh/km)','R130 flow (veh/h)','R130 density (veh/km)','R171 flow (veh/h)','R171 density (veh/km)','S60 flow (km/h)', 'S60 density (veh/km)','S61 flow (km/h)','S61 density (veh/km)']
index=pd.date_range("5:00", periods=180, freq="6min")
index
df_0 = pd.DataFrame(index=index.time, columns=columns)
df_0['S54 flow (veh/h)']=S54_flow_centroid_0
df_0['S54 density (veh/km)']=S54_density_centroid_0
df_0['S1706 flow (veh/h)']=S1706_flow_centroid_0
df_0['S1706 density (veh/km)']=S1706_density_centroid_0
df_0['R169 flow (veh/h)']=R169_flow_centroid_0
df_0['R169 density (veh/km)']=R169_density_centroid_0
df_0['S56 flow (veh/h)']=S56_flow_centroid_0
df_0['S56 density (veh/km)']=S56_density_centroid_0
df_0['R129 flow (veh/h)']=R129_flow_centroid_0
df_0['R129 density (veh/km)']=R129_density_centroid_0
df_0['S57 flow (veh/h)']=S57_flow_centroid_0
df_0['S57 density (veh/km)']=S57_density_centroid_0
df_0['R170 flow (veh/h)']=R170_flow_centroid_0
df_0['R170 density (veh/km)']=R170_density_centroid_0
df_0['S1707 flow (veh/h)']=S1707_flow_centroid_0
df_0['S1707 density (veh/km)']=S1707_density_centroid_0
df_0['S59 flow (veh/h)']=S59_flow_centroid_0
df_0['S59 density (veh/km)']=S59_density_centroid_0
df_0['R130 flow (veh/h)']=R130_flow_centroid_0
df_0['R130 density (veh/km)']=R130_density_centroid_0
df_0['R171 flow (veh/h)']=R171_flow_centroid_0
df_0['R171 density (veh/km)']=R171_density_centroid_0
df_0['S60 flow (km/h)']=S60_flow_centroid_0
df_0['S60 density (veh/km)']=S60_density_centroid_0
df_0['S61 flow (km/h)']=S61_flow_centroid_0
df_0['S61 density (veh/km)']=S61_density_centroid_0

df_0

df_1 = pd.DataFrame(index=index.time, columns=columns)
df_1['S54 flow (veh/h)']=S54_flow_centroid_1
df_1['S54 density (veh/km)']=S54_density_centroid_1
df_1['S1706 flow (veh/h)']=S1706_flow_centroid_1
df_1['S1706 density (veh/km)']=S1706_density_centroid_1
df_1['R169 flow (veh/h)']=R169_flow_centroid_1
df_1['R169 density (veh/km)']=R169_density_centroid_1
df_1['S56 flow (veh/h)']=S56_flow_centroid_1
df_1['S56 density (veh/km)']=S56_density_centroid_1
df_1['R129 flow (veh/h)']=R129_flow_centroid_1
df_1['R129 density (veh/km)']=R129_density_centroid_1
df_1['S57 flow (veh/h)']=S57_flow_centroid_1
df_1['S57 density (veh/km)']=S57_density_centroid_1
df_1['R170 flow (veh/h)']=R170_flow_centroid_1
df_1['R170 density (veh/km)']=R170_density_centroid_1
df_1['S1707 flow (veh/h)']=S1707_flow_centroid_1
df_1['S1707 density (veh/km)']=S1707_density_centroid_1
df_1['S59 flow (veh/h)']=S59_flow_centroid_1
df_1['S59 density (veh/km)']=S59_density_centroid_1
df_1['R130 flow (veh/h)']=R130_flow_centroid_1
df_1['R130 density (veh/km)']=R130_density_centroid_1
df_1['R171 flow (veh/h)']=R171_flow_centroid_1
df_1['R171 density (veh/km)']=R171_density_centroid_1
df_1['S60 flow (km/h)']=S60_flow_centroid_1
df_1['S60 density (veh/km)']=S60_density_centroid_1
df_1['S61 flow (km/h)']=S61_flow_centroid_1
df_1['S61 density (veh/km)']=S61_density_centroid_1
df_1

df_2 = pd.DataFrame(index=index.time, columns=columns)
df_2['S54 flow (veh/h)']=S54_flow_centroid_2
df_2['S54 density (veh/km)']=S54_density_centroid_2
df_2['S1706 flow (veh/h)']=S1706_flow_centroid_2
df_2['S1706 density (veh/km)']=S1706_density_centroid_2
df_2['R169 flow (veh/h)']=R169_flow_centroid_2
df_2['R169 density (veh/km)']=R169_density_centroid_2
df_2['S56 flow (veh/h)']=S56_flow_centroid_2
df_2['S56 density (veh/km)']=S56_density_centroid_2
df_2['R129 flow (veh/h)']=R129_flow_centroid_2
df_2['R129 density (veh/km)']=R129_density_centroid_2
df_2['S57 flow (veh/h)']=S57_flow_centroid_2
df_2['S57 density (veh/km)']=S57_density_centroid_2
df_2['R170 flow (veh/h)']=R170_flow_centroid_2
df_2['R170 density (veh/km)']=R170_density_centroid_2
df_2['S1707 flow (veh/h)']=S1707_flow_centroid_2
df_2['S1707 density (veh/km)']=S1707_density_centroid_2
df_2['S59 flow (veh/h)']=S59_flow_centroid_2
df_2['S59 density (veh/km)']=S59_density_centroid_2
df_2['R130 flow (veh/h)']=R130_flow_centroid_2
df_2['R130 density (veh/km)']=R130_density_centroid_2
df_2['R171 flow (veh/h)']=R171_flow_centroid_2
df_2['R171 density (veh/km)']=R171_density_centroid_2
df_2['S60 flow (km/h)']=S60_flow_centroid_2
df_2['S60 density (veh/km)']=S60_density_centroid_2
df_2['S61 flow (km/h)']=S61_flow_centroid_2
df_2['S61 density (veh/km)']=S61_density_centroid_2
df_2

df_3 = pd.DataFrame(index=index.time, columns=columns)
df_3['S54 flow (veh/h)']=S54_flow_centroid_3
df_3['S54 density (veh/km)']=S54_density_centroid_3
df_3['S1706 flow (veh/h)']=S1706_flow_centroid_3
df_3['S1706 density (veh/km)']=S1706_density_centroid_3
df_3['R169 flow (veh/h)']=R169_flow_centroid_3
df_3['R169 density (veh/km)']=R169_density_centroid_3
df_3['S56 flow (veh/h)']=S56_flow_centroid_3
df_3['S56 density (veh/km)']=S56_density_centroid_3
df_3['R129 flow (veh/h)']=R129_flow_centroid_3
df_3['R129 density (veh/km)']=R129_density_centroid_3
df_3['S57 flow (veh/h)']=S57_flow_centroid_3
df_3['S57 density (veh/km)']=S57_density_centroid_3
df_3['R170 flow (veh/h)']=R170_flow_centroid_3
df_3['R170 density (veh/km)']=R170_density_centroid_3
df_3['S1707 flow (veh/h)']=S1707_flow_centroid_3
df_3['S1707 density (veh/km)']=S1707_density_centroid_3
df_3['S59 flow (veh/h)']=S59_flow_centroid_3
df_3['S59 density (veh/km)']=S59_density_centroid_3
df_3['R130 flow (veh/h)']=R130_flow_centroid_3
df_3['R130 density (veh/km)']=R130_density_centroid_3
df_3['R171 flow (veh/h)']=R171_flow_centroid_3
df_3['R171 density (veh/km)']=R171_density_centroid_3
df_3['S60 flow (km/h)']=S60_flow_centroid_3
df_3['S60 density (veh/km)']=S60_density_centroid_3
df_3['S61 flow (km/h)']=S61_flow_centroid_3
df_3['S61 density (veh/km)']=S61_density_centroid_3
df_3

df_4 = pd.DataFrame(index=index.time, columns=columns)
df_4['S54 flow (veh/h)']=S54_flow_centroid_4
df_4['S54 density (veh/km)']=S54_density_centroid_4
df_4['S1706 flow (veh/h)']=S1706_flow_centroid_4
df_4['S1706 density (veh/km)']=S1706_density_centroid_4
df_4['R169 flow (veh/h)']=R169_flow_centroid_4
df_4['R169 density (veh/km)']=R169_density_centroid_4
df_4['S56 flow (veh/h)']=S56_flow_centroid_4
df_4['S56 density (veh/km)']=S56_density_centroid_4
df_4['R129 flow (veh/h)']=R129_flow_centroid_4
df_4['R129 density (veh/km)']=R129_density_centroid_4
df_4['S57 flow (veh/h)']=S57_flow_centroid_4
df_4['S57 density (veh/km)']=S57_density_centroid_4
df_4['R170 flow (veh/h)']=R170_flow_centroid_4
df_4['R170 density (veh/km)']=R170_density_centroid_4
df_4['S1707 flow (veh/h)']=S1707_flow_centroid_4
df_4['S1707 density (veh/km)']=S1707_density_centroid_4
df_4['S59 flow (veh/h)']=S59_flow_centroid_4
df_4['S59 density (veh/km)']=S59_density_centroid_4
df_4['R130 flow (veh/h)']=R130_flow_centroid_4
df_4['R130 density (veh/km)']=R130_density_centroid_4
df_4['R171 flow (veh/h)']=R171_flow_centroid_4
df_4['R171 density (veh/km)']=R171_density_centroid_4
df_4['S60 flow (km/h)']=S60_flow_centroid_4
df_4['S60 density (veh/km)']=S60_density_centroid_4
df_4['S61 flow (km/h)']=S61_flow_centroid_4
df_4['S61 density (veh/km)']=S61_density_centroid_4
df_4



# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter('/Users/nronzoni/Desktop/TrafficData Minnesota/Scenario10.xlsx', engine='xlsxwriter')

# Write each dataframe to a different worksheet.
df_0.to_excel(writer, sheet_name='k=0')
df_1.to_excel(writer, sheet_name='k=1')
df_2.to_excel(writer, sheet_name='k=2')
df_3.to_excel(writer, sheet_name='k=3')
df_4.to_excel(writer, sheet_name='k=4')
# Close the Pandas Excel writer and output the Excel file.
writer.save()

#Scenario 11
#first cluster k=0
#S54
S54_speed_centroid=centroids[0][:,0]
S54_speed_centroid_0= series_train_S54_speed[1].inverse_transform(S54_speed_centroid.reshape((len(S54_speed_centroid), 1)))
S54_density_centroid=centroids[0][:,1]
S54_density_centroid_0= series_train_S54_density[1].inverse_transform(S54_density_centroid.reshape((len(S54_density_centroid), 1)))
#S1706
S1706_speed_centroid=centroids[0][:,2]
S1706_speed_centroid_0= series_train_S1706_speed[1].inverse_transform(S1706_speed_centroid.reshape((len(S1706_speed_centroid), 1)))
S1706_density_centroid=centroids[0][:,3]
S1706_density_centroid_0= series_train_S1706_density[1].inverse_transform(S1706_density_centroid.reshape((len(S1706_density_centroid), 1)))
#R169
R169_speed_centroid=centroids[0][:,4]
R169_speed_centroid_0= series_train_R169_speed[1].inverse_transform(R169_speed_centroid.reshape((len(R169_speed_centroid), 1)))
R169_density_centroid=centroids[0][:,5]
R169_density_centroid_0= series_train_R169_density[1].inverse_transform(R169_density_centroid.reshape((len(R169_density_centroid), 1)))
#S56
S56_speed_centroid=centroids[0][:,6]
S56_speed_centroid_0= series_train_S56_speed[1].inverse_transform(S56_speed_centroid.reshape((len(S56_speed_centroid), 1)))
S56_density_centroid=centroids[0][:,7]
S56_density_centroid_0= series_train_S56_density[1].inverse_transform(S56_density_centroid.reshape((len(S56_density_centroid), 1)))
#R129
R129_speed_centroid=centroids[0][:,8]
R129_speed_centroid_0= series_train_R129_speed[1].inverse_transform(R129_speed_centroid.reshape((len(R129_speed_centroid), 1)))
R129_density_centroid=centroids[0][:,9]
R129_density_centroid_0= series_train_R129_density[1].inverse_transform(R129_density_centroid.reshape((len(R129_density_centroid), 1)))
#S57
S57_speed_centroid=centroids[0][:,10]
S57_speed_centroid_0= series_train_S57_speed[1].inverse_transform(S57_speed_centroid.reshape((len(S57_speed_centroid), 1)))
S57_density_centroid=centroids[0][:,11]
S57_density_centroid_0= series_train_S57_density[1].inverse_transform(S57_density_centroid.reshape((len(S57_density_centroid), 1)))
#R170
R170_speed_centroid=centroids[0][:,12]
R170_speed_centroid_0= series_train_R170_speed[1].inverse_transform(R170_speed_centroid.reshape((len(R170_speed_centroid), 1)))
R170_density_centroid=centroids[0][:,13]
R170_density_centroid_0= series_train_R170_density[1].inverse_transform(R170_density_centroid.reshape((len(R170_density_centroid), 1)))
#S1707
S1707_speed_centroid=centroids[0][:,14]
S1707_speed_centroid_0= series_train_S1707_speed[1].inverse_transform(S1707_speed_centroid.reshape((len(S1707_speed_centroid), 1)))
S1707_density_centroid=centroids[0][:,15]
S1707_density_centroid_0= series_train_S1707_density[1].inverse_transform(S1707_density_centroid.reshape((len(S1707_density_centroid), 1)))
#S59
S59_speed_centroid=centroids[0][:,16]
S59_speed_centroid_0= series_train_S59_speed[1].inverse_transform(S59_speed_centroid.reshape((len(S59_speed_centroid), 1)))
S59_density_centroid=centroids[0][:,17]
S59_density_centroid_0= series_train_S59_density[1].inverse_transform(S59_density_centroid.reshape((len(S59_density_centroid), 1)))
#R130
R130_speed_centroid=centroids[0][:,18]
R130_speed_centroid_0= series_train_R130_speed[1].inverse_transform(R130_speed_centroid.reshape((len(R130_speed_centroid), 1)))
R130_density_centroid=centroids[0][:,19]
R130_density_centroid_0= series_train_R130_density[1].inverse_transform(R130_density_centroid.reshape((len(R130_density_centroid), 1)))
#R171
R171_speed_centroid=centroids[0][:,20]
R171_speed_centroid_0= series_train_R171_speed[1].inverse_transform(R171_speed_centroid.reshape((len(R171_speed_centroid), 1)))
R171_density_centroid=centroids[0][:,21]
R171_density_centroid_0= series_train_R171_density[1].inverse_transform(R171_density_centroid.reshape((len(R171_density_centroid), 1)))
#S60
S60_speed_centroid=centroids[0][:,22]
S60_speed_centroid_0= series_train_S60_speed[1].inverse_transform(S60_speed_centroid.reshape((len(S60_speed_centroid), 1)))
S60_density_centroid=centroids[0][:,23]
S60_density_centroid_0= series_train_S60_density[1].inverse_transform(S60_density_centroid.reshape((len(S60_density_centroid), 1)))
#S61
S61_speed_centroid=centroids[0][:,24]
S61_speed_centroid_0= series_train_S61_speed[1].inverse_transform(S61_speed_centroid.reshape((len(S61_speed_centroid), 1)))
S61_density_centroid=centroids[0][:,25]
S61_density_centroid_0= series_train_S61_density[1].inverse_transform(S61_density_centroid.reshape((len(S61_density_centroid), 1)))

#second cluster k=1
#S54
S54_speed_centroid=centroids[1][:,0]
S54_speed_centroid_1= series_train_S54_speed[1].inverse_transform(S54_speed_centroid.reshape((len(S54_speed_centroid), 1)))
S54_density_centroid=centroids[1][:,1]
S54_density_centroid_1= series_train_S54_density[1].inverse_transform(S54_density_centroid.reshape((len(S54_density_centroid), 1)))
#S1706
S1706_speed_centroid=centroids[1][:,2]
S1706_speed_centroid_1= series_train_S1706_speed[1].inverse_transform(S1706_speed_centroid.reshape((len(S1706_speed_centroid), 1)))
S1706_density_centroid=centroids[1][:,3]
S1706_density_centroid_1= series_train_S1706_density[1].inverse_transform(S1706_density_centroid.reshape((len(S1706_density_centroid), 1)))
#R169
R169_speed_centroid=centroids[1][:,4]
R169_speed_centroid_1= series_train_R169_speed[1].inverse_transform(R169_speed_centroid.reshape((len(R169_speed_centroid), 1)))
R169_density_centroid=centroids[1][:,5]
R169_density_centroid_1= series_train_R169_density[1].inverse_transform(R169_density_centroid.reshape((len(R169_density_centroid), 1)))
#S56
S56_speed_centroid=centroids[1][:,6]
S56_speed_centroid_1= series_train_S56_speed[1].inverse_transform(S56_speed_centroid.reshape((len(S56_speed_centroid), 1)))
S56_density_centroid=centroids[1][:,7]
S56_density_centroid_1= series_train_S56_density[1].inverse_transform(S56_density_centroid.reshape((len(S56_density_centroid), 1)))
#R129
R129_speed_centroid=centroids[1][:,8]
R129_speed_centroid_1= series_train_R129_speed[1].inverse_transform(R129_speed_centroid.reshape((len(R129_speed_centroid), 1)))
R129_density_centroid=centroids[1][:,9]
R129_density_centroid_1= series_train_R129_density[1].inverse_transform(R129_density_centroid.reshape((len(R129_density_centroid), 1)))
#S57
S57_speed_centroid=centroids[1][:,10]
S57_speed_centroid_1= series_train_S57_speed[1].inverse_transform(S57_speed_centroid.reshape((len(S57_speed_centroid), 1)))
S57_density_centroid=centroids[1][:,11]
S57_density_centroid_1= series_train_S57_density[1].inverse_transform(S57_density_centroid.reshape((len(S57_density_centroid), 1)))
#R170
R170_speed_centroid=centroids[1][:,12]
R170_speed_centroid_1= series_train_R170_speed[1].inverse_transform(R170_speed_centroid.reshape((len(R170_speed_centroid), 1)))
R170_density_centroid=centroids[1][:,13]
R170_density_centroid_1= series_train_R170_density[1].inverse_transform(R170_density_centroid.reshape((len(R170_density_centroid), 1)))
#S1707
S1707_speed_centroid=centroids[1][:,14]
S1707_speed_centroid_1= series_train_S1707_speed[1].inverse_transform(S1707_speed_centroid.reshape((len(S1707_speed_centroid), 1)))
S1707_density_centroid=centroids[1][:,15]
S1707_density_centroid_1= series_train_S1707_density[1].inverse_transform(S1707_density_centroid.reshape((len(S1707_density_centroid), 1)))
#S59
S59_speed_centroid=centroids[1][:,16]
S59_speed_centroid_1= series_train_S59_speed[1].inverse_transform(S59_speed_centroid.reshape((len(S59_speed_centroid), 1)))
S59_density_centroid=centroids[1][:,17]
S59_density_centroid_1= series_train_S59_density[1].inverse_transform(S59_density_centroid.reshape((len(S59_density_centroid), 1)))
#R130
R130_speed_centroid=centroids[1][:,18]
R130_speed_centroid_1= series_train_R130_speed[1].inverse_transform(R130_speed_centroid.reshape((len(R130_speed_centroid), 1)))
R130_density_centroid=centroids[1][:,19]
R130_density_centroid_1= series_train_R130_density[1].inverse_transform(R130_density_centroid.reshape((len(R130_density_centroid), 1)))
#R171
R171_speed_centroid=centroids[1][:,20]
R171_speed_centroid_1= series_train_R171_speed[1].inverse_transform(R171_speed_centroid.reshape((len(R171_speed_centroid), 1)))
R171_density_centroid=centroids[1][:,21]
R171_density_centroid_1= series_train_R171_density[1].inverse_transform(R171_density_centroid.reshape((len(R171_density_centroid), 1)))
#S60
S60_speed_centroid=centroids[1][:,22]
S60_speed_centroid_1= series_train_S60_speed[1].inverse_transform(S60_speed_centroid.reshape((len(S60_speed_centroid), 1)))
S60_density_centroid=centroids[1][:,23]
S60_density_centroid_1= series_train_S60_density[1].inverse_transform(S60_density_centroid.reshape((len(S60_density_centroid), 1)))
#S61
S61_speed_centroid=centroids[1][:,24]
S61_speed_centroid_1= series_train_S61_speed[1].inverse_transform(S61_speed_centroid.reshape((len(S61_speed_centroid), 1)))
S61_density_centroid=centroids[1][:,25]
S61_density_centroid_1= series_train_S61_density[1].inverse_transform(S61_density_centroid.reshape((len(S61_density_centroid), 1)))

#third cluster k=2
#S54
S54_speed_centroid=centroids[2][:,0]
S54_speed_centroid_2= series_train_S54_speed[1].inverse_transform(S54_speed_centroid.reshape((len(S54_speed_centroid), 1)))
S54_density_centroid=centroids[2][:,1]
S54_density_centroid_2= series_train_S54_density[1].inverse_transform(S54_density_centroid.reshape((len(S54_density_centroid), 1)))
#S1706
S1706_speed_centroid=centroids[2][:,2]
S1706_speed_centroid_2= series_train_S1706_speed[1].inverse_transform(S1706_speed_centroid.reshape((len(S1706_speed_centroid), 1)))
S1706_density_centroid=centroids[2][:,3]
S1706_density_centroid_2= series_train_S1706_density[1].inverse_transform(S1706_density_centroid.reshape((len(S1706_density_centroid), 1)))
#R169
R169_speed_centroid=centroids[2][:,4]
R169_speed_centroid_2= series_train_R169_speed[1].inverse_transform(R169_speed_centroid.reshape((len(R169_speed_centroid), 1)))
R169_density_centroid=centroids[2][:,5]
R169_density_centroid_2= series_train_R169_density[1].inverse_transform(R169_density_centroid.reshape((len(R169_density_centroid), 1)))
#S56
S56_speed_centroid=centroids[2][:,6]
S56_speed_centroid_2= series_train_S56_speed[1].inverse_transform(S56_speed_centroid.reshape((len(S56_speed_centroid), 1)))
S56_density_centroid=centroids[2][:,7]
S56_density_centroid_2= series_train_S56_density[1].inverse_transform(S56_density_centroid.reshape((len(S56_density_centroid), 1)))
#R129
R129_speed_centroid=centroids[2][:,8]
R129_speed_centroid_2= series_train_R129_speed[1].inverse_transform(R129_speed_centroid.reshape((len(R129_speed_centroid), 1)))
R129_density_centroid=centroids[2][:,9]
R129_density_centroid_2= series_train_R129_density[1].inverse_transform(R129_density_centroid.reshape((len(R129_density_centroid), 1)))
#S57
S57_speed_centroid=centroids[2][:,10]
S57_speed_centroid_2= series_train_S57_speed[1].inverse_transform(S57_speed_centroid.reshape((len(S57_speed_centroid), 1)))
S57_density_centroid=centroids[2][:,11]
S57_density_centroid_2= series_train_S57_density[1].inverse_transform(S57_density_centroid.reshape((len(S57_density_centroid), 1)))
#R170
R170_speed_centroid=centroids[2][:,12]
R170_speed_centroid_2= series_train_R170_speed[1].inverse_transform(R170_speed_centroid.reshape((len(R170_speed_centroid), 1)))
R170_density_centroid=centroids[2][:,13]
R170_density_centroid_2= series_train_R170_density[1].inverse_transform(R170_density_centroid.reshape((len(R170_density_centroid), 1)))
#S1707
S1707_speed_centroid=centroids[2][:,14]
S1707_speed_centroid_2= series_train_S1707_speed[1].inverse_transform(S1707_speed_centroid.reshape((len(S1707_speed_centroid), 1)))
S1707_density_centroid=centroids[2][:,15]
S1707_density_centroid_2= series_train_S1707_density[1].inverse_transform(S1707_density_centroid.reshape((len(S1707_density_centroid), 1)))
#S59
S59_speed_centroid=centroids[2][:,16]
S59_speed_centroid_2= series_train_S59_speed[1].inverse_transform(S59_speed_centroid.reshape((len(S59_speed_centroid), 1)))
S59_density_centroid=centroids[2][:,17]
S59_density_centroid_2= series_train_S59_density[1].inverse_transform(S59_density_centroid.reshape((len(S59_density_centroid), 1)))
#R130
R130_speed_centroid=centroids[2][:,18]
R130_speed_centroid_2= series_train_R130_speed[1].inverse_transform(R130_speed_centroid.reshape((len(R130_speed_centroid), 1)))
R130_density_centroid=centroids[2][:,19]
R130_density_centroid_2= series_train_R130_density[1].inverse_transform(R130_density_centroid.reshape((len(R130_density_centroid), 1)))
#R171
R171_speed_centroid=centroids[2][:,20]
R171_speed_centroid_2= series_train_R171_speed[1].inverse_transform(R171_speed_centroid.reshape((len(R171_speed_centroid), 1)))
R171_density_centroid=centroids[2][:,21]
R171_density_centroid_2= series_train_R171_density[1].inverse_transform(R171_density_centroid.reshape((len(R171_density_centroid), 1)))
#S60
S60_speed_centroid=centroids[2][:,22]
S60_speed_centroid_2= series_train_S60_speed[1].inverse_transform(S60_speed_centroid.reshape((len(S60_speed_centroid), 1)))
S60_density_centroid=centroids[2][:,23]
S60_density_centroid_2= series_train_S60_density[1].inverse_transform(S60_density_centroid.reshape((len(S60_density_centroid), 1)))
#S61
S61_speed_centroid=centroids[2][:,24]
S61_speed_centroid_2= series_train_S61_speed[1].inverse_transform(S61_speed_centroid.reshape((len(S61_speed_centroid), 1)))
S61_density_centroid=centroids[2][:,25]
S61_density_centroid_2= series_train_S61_density[1].inverse_transform(S61_density_centroid.reshape((len(S61_density_centroid), 1)))

#save centroids of the cluster 
columns = ['S54 speed (km/h)','S54 density (veh/km)','S1706 speed (km/h)','S1706 density (veh/km)', 'R169 speed (km/h)','R169 density (veh/km)','S56 speed (km/h)','S56 density (veh/km)','R129 speed (km/h)','R129 density (veh/km)','S57 speed (km/h)', 'S57 density (veh/km)','R170 speed (km/h)','R170 density (veh/km)','S1707 speed (km/h)','S1707 density (veh/km)','S59 speed (km/h)', 'S59 density (veh/km)','R130 speed (km/h)','R130 density (veh/km)','R171 speed (km/h)','R171 density (veh/km)','S60 speed (km/h)', 'S60 density (veh/km)','S61 speed (km/h)','S61 density (veh/km)']
index=pd.date_range("5:00", periods=180, freq="6min")
index
df_0 = pd.DataFrame(index=index.time, columns=columns)
df_0['S54 speed (km/h)']=S54_speed_centroid_0
df_0['S54 density (veh/km)']=S54_density_centroid_0
df_0['S1706 speed (km/h)']=S1706_speed_centroid_0
df_0['S1706 density (veh/km)']=S1706_density_centroid_0
df_0['R169 speed (km/h)']=R169_speed_centroid_0
df_0['R169 density (veh/km)']=R169_density_centroid_0
df_0['S56 speed (km/h)']=S56_speed_centroid_0
df_0['S56 density (veh/km)']=S56_density_centroid_0
df_0['R129 speed (km/h)']=R129_speed_centroid_0
df_0['R129 density (veh/km)']=R129_density_centroid_0
df_0['S57 speed (km/h)']=S57_speed_centroid_0
df_0['S57 density (veh/km)']=S57_density_centroid_0
df_0['R170 speed (km/h)']=R170_speed_centroid_0
df_0['R170 density (veh/km)']=R170_density_centroid_0
df_0['S1707 speed (km/h)']=S1707_speed_centroid_0
df_0['S1707 density (veh/km)']=S1707_density_centroid_0
df_0['S59 speed (km/h)']=S59_speed_centroid_0
df_0['S59 density (veh/km)']=S59_density_centroid_0
df_0['R130 speed (km/h)']=R130_speed_centroid_0
df_0['R130 density (veh/km)']=R130_density_centroid_0
df_0['R171 speed (km/h)']=R171_speed_centroid_0
df_0['R171 density (veh/km)']=R171_density_centroid_0
df_0['S60 speed (km/h)']=S60_speed_centroid_0
df_0['S60 density (veh/km)']=S60_density_centroid_0
df_0['S61 speed (km/h)']=S61_speed_centroid_0
df_0['S61 density (veh/km)']=S61_density_centroid_0

df_0

df_1= pd.DataFrame(index=index.time, columns=columns)
df_1['S54 speed (km/h)']=S54_speed_centroid_1
df_1['S54 density (veh/km)']=S54_density_centroid_1
df_1['S1706 speed (km/h)']=S1706_speed_centroid_1
df_1['S1706 density (veh/km)']=S1706_density_centroid_1
df_1['R169 speed (km/h)']=R169_speed_centroid_1
df_1['R169 density (veh/km)']=R169_density_centroid_1
df_1['S56 speed (km/h)']=S56_speed_centroid_1
df_1['S56 density (veh/km)']=S56_density_centroid_1
df_1['R129 speed (km/h)']=R129_speed_centroid_1
df_1['R129 density (veh/km)']=R129_density_centroid_1
df_1['S57 speed (km/h)']=S57_speed_centroid_1
df_1['S57 density (veh/km)']=S57_density_centroid_1
df_1['R170 speed (km/h)']=R170_speed_centroid_1
df_1['R170 density (veh/km)']=R170_density_centroid_1
df_1['S1707 speed (km/h)']=S1707_speed_centroid_1
df_1['S1707 density (veh/km)']=S1707_density_centroid_1
df_1['S59 speed (km/h)']=S59_speed_centroid_1
df_1['S59 density (veh/km)']=S59_density_centroid_1
df_1['R130 speed (km/h)']=R130_speed_centroid_1
df_1['R130 density (veh/km)']=R130_density_centroid_1
df_1['R171 speed (km/h)']=R171_speed_centroid_1
df_1['R171 density (veh/km)']=R171_density_centroid_1
df_1['S60 speed (km/h)']=S60_speed_centroid_1
df_1['S60 density (veh/km)']=S60_density_centroid_1
df_1['S61 speed (km/h)']=S61_speed_centroid_1
df_1['S61 density (veh/km)']=S61_density_centroid_1

df_1

df_2= pd.DataFrame(index=index.time, columns=columns)
df_2['S54 speed (km/h)']=S54_speed_centroid_2
df_2['S54 density (veh/km)']=S54_density_centroid_2
df_2['S1706 speed (km/h)']=S1706_speed_centroid_2
df_2['S1706 density (veh/km)']=S1706_density_centroid_2
df_2['R169 speed (km/h)']=R169_speed_centroid_2
df_2['R169 density (veh/km)']=R169_density_centroid_2
df_2['S56 speed (km/h)']=S56_speed_centroid_2
df_2['S56 density (veh/km)']=S56_density_centroid_2
df_2['R129 speed (km/h)']=R129_speed_centroid_2
df_2['R129 density (veh/km)']=R129_density_centroid_2
df_2['S57 speed (km/h)']=S57_speed_centroid_2
df_2['S57 density (veh/km)']=S57_density_centroid_2
df_2['R170 speed (km/h)']=R170_speed_centroid_2
df_2['R170 density (veh/km)']=R170_density_centroid_2
df_2['S1707 speed (km/h)']=S1707_speed_centroid_2
df_2['S1707 density (veh/km)']=S1707_density_centroid_2
df_2['S59 speed (km/h)']=S59_speed_centroid_2
df_2['S59 density (veh/km)']=S59_density_centroid_2
df_2['R130 speed (km/h)']=R130_speed_centroid_2
df_2['R130 density (veh/km)']=R130_density_centroid_2
df_2['R171 speed (km/h)']=R171_speed_centroid_2
df_2['R171 density (veh/km)']=R171_density_centroid_2
df_2['S60 speed (km/h)']=S60_speed_centroid_2
df_2['S60 density (veh/km)']=S60_density_centroid_2
df_2['S61 speed (km/h)']=S61_speed_centroid_2
df_2['S61 density (veh/km)']=S61_density_centroid_2

df_2



# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter('/Users/nronzoni/Desktop/TrafficData Minnesota/Scenario11.xlsx', engine='xlsxwriter')

# Write each dataframe to a different worksheet.
df_0.to_excel(writer, sheet_name='k=0')
df_1.to_excel(writer, sheet_name='k=1')
df_2.to_excel(writer, sheet_name='k=2')

# Close the Pandas Excel writer and output the Excel file.
writer.save()




