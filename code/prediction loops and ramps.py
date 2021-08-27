#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 11:48:13 2021

@author: nronzoni
"""
#################################################### FLOW ###########################################################
#### multistep prediction  classification approach 
S54_day=classification_pred_same_tris(multivariate_time_series_train,multivariate_time_series_test[23:24,:,:],30,10,10,0)
S1706_day=classification_pred_same_tris(multivariate_time_series_train,multivariate_time_series_test[23:24,:,:],30,10,10,1)
R169_day=classification_pred_same_tris(multivariate_time_series_train,multivariate_time_series_test[23:24,:,:],30,10,10,2)
S56_day=classification_pred_same_tris(multivariate_time_series_train,multivariate_time_series_test[23:24,:,:],30,10,10,3)
R129_day=classification_pred_same_tris(multivariate_time_series_train,multivariate_time_series_test[23:24,:,:],30,10,10,4)
S57_day=classification_pred_same_tris(multivariate_time_series_train,multivariate_time_series_test[23:24,:,:],30,10,10,5)
R170_day=classification_pred_same_tris(multivariate_time_series_train,multivariate_time_series_test[23:24,:,:],30,10,10,6)
S1707_day=classification_pred_same_tris(multivariate_time_series_train,multivariate_time_series_test[23:24,:,:],30,10,10,7)
S59_day=classification_pred_same_tris(multivariate_time_series_train,multivariate_time_series_test[23:24,:,:],30,10,10,8)
R130_day=classification_pred_same_tris(multivariate_time_series_train,multivariate_time_series_test[23:24,:,:],30,10,10,9)
R171_day=classification_pred_same_tris(multivariate_time_series_train,multivariate_time_series_test[23:24,:,:],30,10,10,10)
S60_day=classification_pred_same_tris(multivariate_time_series_train,multivariate_time_series_test[23:24,:,:],30,10,10,11)
S61_day=classification_pred_same_tris(multivariate_time_series_train,multivariate_time_series_test[23:24,:,:],30,10,10,12)

#prediction
S54_day[1].shape
#ground_truth
S54_day[2].shape
columns = ['S54 flow (veh/h)','S54 flow (veh/h) ground truth','S1706 flow (veh/h)','S1706 flow (veh/h) ground truth', 'R169 flow (veh/h)','R169 flow (veh/h) ground truth','S56 flow (veh/h)','S56 flow (veh/h) ground truth','R129 flow (veh/h)','R129 flow (veh/h) ground truth', 'S57 flow (veh/h)','S57 flow (veh/h) ground truth','R170 flow (veh/h)','R170 flow (veh/h) ground truth','S1707 flow (veh/h)','S1707 flow (veh/h) ground truth', 'S59 flow (veh/h)','S59 flow (veh/h) ground truth','R130 flow (veh/h)','R130 flow (veh/h) ground truth','R171 flow (veh/h)','R171 flow (veh/h) ground truth', 'S60 flow (veh/h)','S60 flow (veh/h) ground truth','S61 flow (veh/h)','S61 flow (veh/h) ground truth']
index=pd.date_range("08:00", periods=10, freq="6min")
df_5= pd.DataFrame(index=index.time, columns=columns)
df_5

# use the inverse of the normalization factor 
Y_pred_S54=series_train_S54_flow[1].inverse_transform(S54_day[1])
Y_test_S54=series_test_S54_flow[1].inverse_transform(S54_day[2].reshape(-1,1))
error_S54=math.sqrt(mean_squared_error(Y_test_S54,Y_pred_S54.reshape(-1,1)))
Y_pred_S1706=series_train_S1706_flow[1].inverse_transform(S1706_day[1])
Y_test_S1706=series_test_S1706_flow[1].inverse_transform(S1706_day[2].reshape(-1,1))
error_S1706=math.sqrt(mean_squared_error(Y_test_S1706,Y_pred_S1706.reshape(-1,1)))
Y_pred_R169=series_train_R169_flow[1].inverse_transform(R169_day[1])
Y_test_R169=series_test_R169_flow[1].inverse_transform(R169_day[2].reshape(-1,1))
error_R169=math.sqrt(mean_squared_error(Y_test_R169,Y_pred_R169.reshape(-1,1)))
Y_pred_S56=series_train_S56_flow[1].inverse_transform(S56_day[1])
Y_test_S56=series_test_S56_flow[1].inverse_transform(S56_day[2].reshape(-1,1))
error_S56=math.sqrt(mean_squared_error(Y_test_S56,Y_pred_S56.reshape(-1,1)))
Y_pred_R129=series_train_R129_flow[1].inverse_transform(R129_day[1])
Y_test_R129=series_test_R129_flow[1].inverse_transform(R129_day[2].reshape(-1,1))
error_R129=math.sqrt(mean_squared_error(Y_test_R129,Y_pred_R129.reshape(-1,1)))
Y_pred_S57=series_train_S57_flow[1].inverse_transform(S57_day[1])
Y_test_S57=series_test_S57_flow[1].inverse_transform(S57_day[2].reshape(-1,1))
error_S57=math.sqrt(mean_squared_error(Y_test_S57,Y_pred_S57.reshape(-1,1)))
Y_pred_R170=series_train_R170_flow[1].inverse_transform(R170_day[1])
Y_test_R170=series_test_R170_flow[1].inverse_transform(R170_day[2].reshape(-1,1))
error_R170=math.sqrt(mean_squared_error(Y_test_R170,Y_pred_R170.reshape(-1,1)))
Y_pred_S1707=series_train_S1707_flow[1].inverse_transform(S1707_day[1])
Y_test_S1707=series_test_S1707_flow[1].inverse_transform(S1707_day[2].reshape(-1,1))
error_S1707=math.sqrt(mean_squared_error(Y_test_S1707,Y_pred_S1707.reshape(-1,1)))
Y_pred_S59=series_train_S59_flow[1].inverse_transform(S59_day[1])
Y_test_S59=series_test_S59_flow[1].inverse_transform(S59_day[2].reshape(-1,1))
error_S59=math.sqrt(mean_squared_error(Y_test_S59,Y_pred_S59.reshape(-1,1)))
Y_pred_R130=series_train_R130_flow[1].inverse_transform(R130_day[1])
Y_test_R130=series_test_R130_flow[1].inverse_transform(R130_day[2].reshape(-1,1))
error_R130=math.sqrt(mean_squared_error(Y_test_R130,Y_pred_R130.reshape(-1,1)))
Y_pred_R171=series_train_R171_flow[1].inverse_transform(R171_day[1])
Y_test_R171=series_test_R171_flow[1].inverse_transform(R171_day[2].reshape(-1,1))
error_R171=math.sqrt(mean_squared_error(Y_test_R171,Y_pred_R171.reshape(-1,1)))
Y_pred_S60=series_train_S60_flow[1].inverse_transform(S60_day[1])
Y_test_S60=series_test_S60_flow[1].inverse_transform(S60_day[2].reshape(-1,1))
error_S60=math.sqrt(mean_squared_error(Y_test_S60,Y_pred_S60.reshape(-1,1)))
Y_pred_S61=series_train_S61_flow[1].inverse_transform(S61_day[1])
Y_test_S61=series_test_S61_flow[1].inverse_transform(S61_day[2].reshape(-1,1))
error_S61=math.sqrt(mean_squared_error(Y_test_S61,Y_pred_S61.reshape(-1,1)))



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
writer = pd.ExcelWriter('/Users/nronzoni/Desktop/single loop/Classification_prediction_flow_1HOUR.xlsx', engine='xlsxwriter')

# Write each dataframe to a different worksheet.
df_1.to_excel(writer, sheet_name='10-2-2014 morning')
df_2.to_excel(writer, sheet_name='12-2-2014 morning')
df_3.to_excel(writer, sheet_name='22-3-2014 afternoon')
df_4.to_excel(writer, sheet_name='15-8-2014 afternoon')
df_5.to_excel(writer, sheet_name='10-9-2014 afternoon')

# Close the Pandas Excel writer and output the Excel file.
writer.save()

# multistep prediction SVR approach

first_day_S54=SVR_pred_d_speed_bis(multivariate_time_series_train,multivariate_time_series_test[18:19,:,:],95,10,10,0)
first_day_S1706=SVR_pred_d_speed_bis(multivariate_time_series_train,multivariate_time_series_test[18:19,:,:],95,10,10,1)
first_day_R169=SVR_pred_d_speed_bis(multivariate_time_series_train,multivariate_time_series_test[18:19,:,:],95,10,10,2)
first_day_S56=SVR_pred_d_speed_bis(multivariate_time_series_train,multivariate_time_series_test[18:19,:,:],95,10,10,3)
first_day_R129=SVR_pred_d_speed_bis(multivariate_time_series_train,multivariate_time_series_test[18:19,:,:],95,10,10,4)
first_day_S57=SVR_pred_d_speed_bis(multivariate_time_series_train,multivariate_time_series_test[18:19,:,:],95,10,10,5)
first_day_R170=SVR_pred_d_speed_bis(multivariate_time_series_train,multivariate_time_series_test[18:19,:,:],95,10,10,6)
first_day_S1707=SVR_pred_d_speed_bis(multivariate_time_series_train,multivariate_time_series_test[18:19,:,:],95,10,10,7)
first_day_S59=SVR_pred_d_speed_bis(multivariate_time_series_train,multivariate_time_series_test[18:19,:,:],95,10,10,8)
first_day_R130=SVR_pred_d_speed_bis(multivariate_time_series_train,multivariate_time_series_test[18:19,:,:],95,10,10,9)
first_day_R171=SVR_pred_d_speed_bis(multivariate_time_series_train,multivariate_time_series_test[18:19,:,:],95,10,10,10)
first_day_S60=SVR_pred_d_speed_bis(multivariate_time_series_train,multivariate_time_series_test[18:19,:,:],95,10,10,11)
first_day_S61=SVR_pred_d_speed_bis(multivariate_time_series_train,multivariate_time_series_test[18:19,:,:],95,10,10,12)


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



columns = ['S54 flow (veh/h)','S54 flow (veh/h) ground truth','S1706 flow (veh/h)','S1706 flow (veh/h) ground truth', 'R169 flow (veh/h)','R169 flow (veh/h) ground truth','S56 flow (veh/h)','S56 flow (veh/h) ground truth','R129 flow (veh/h)','R129 flow (veh/h) ground truth', 'S57 flow (veh/h)','S57 flow (veh/h) ground truth','R170 flow (veh/h)','R170 flow (veh/h) ground truth','S1707 flow (veh/h)','S1707 flow (veh/h) ground truth', 'S59 flow (veh/h)','S59 flow (veh/h) ground truth','R130 flow (veh/h)','R130 flow (veh/h) ground truth','R171 flow (veh/h)','R171 flow (veh/h) ground truth', 'S60 flow (veh/h)','S60 flow (veh/h) ground truth','S61 flow (veh/h)','S61 flow (veh/h) ground truth']
index=pd.date_range("14:30", periods=10, freq="6min")
df_4 = pd.DataFrame(index=index.time, columns=columns)
df_4
df_4['S54 flow (veh/h)']=Y_pred_S54.reshape(-1,1)
df_4['S54 flow (veh/h) ground truth']=Y_test_S54.reshape(-1,1)
df_4['S1706 flow (veh/h)']=Y_pred_S1706.reshape(-1,1)
df_4['S1706 flow (veh/h) ground truth']=Y_test_S1706.reshape(-1,1)
df_4['R169 flow (veh/h)']=Y_pred_R169.reshape(-1,1)
df_4['R169 flow (veh/h) ground truth']=Y_test_R169.reshape(-1,1)
df_4['S56 flow (veh/h)']=Y_pred_S56.reshape(-1,1)
df_4['S56 flow (veh/h) ground truth']=Y_test_S56.reshape(-1,1)
df_4['R129 flow (veh/h)']=Y_pred_R129.reshape(-1,1)
df_4['R129 flow (veh/h) ground truth']=Y_test_R129.reshape(-1,1)
df_4['S57 flow (veh/h)']=Y_pred_S57.reshape(-1,1)
df_4['S57 flow (veh/h) ground truth']=Y_test_S57.reshape(-1,1)
df_4['R170 flow (veh/h)']=Y_pred_R170.reshape(-1,1)
df_4['R170 flow (veh/h) ground truth']=Y_test_R170.reshape(-1,1)
df_4['S1707 flow (veh/h)']=Y_pred_S1707.reshape(-1,1)
df_4['S1707 flow (veh/h) ground truth']=Y_test_S1707.reshape(-1,1)
df_4['S59 flow (veh/h)']=Y_pred_S59.reshape(-1,1)
df_4['S59 flow (veh/h) ground truth']=Y_test_S59.reshape(-1,1)
df_4['R130 flow (veh/h)']=Y_pred_R130.reshape(-1,1)
df_4['R130 flow (veh/h) ground truth']=Y_test_R130.reshape(-1,1)
df_4['R171 flow (veh/h)']=Y_pred_R171.reshape(-1,1)
df_4['R171 flow (veh/h) ground truth']=Y_test_R171.reshape(-1,1)
df_4['S60 flow (veh/h)']=Y_pred_S60.reshape(-1,1)
df_4['S60 flow (veh/h) ground truth']=Y_test_S60.reshape(-1,1)
df_4['S61 flow (veh/h)']=Y_pred_S61.reshape(-1,1)
df_4['S61 flow (veh/h) ground truth']=Y_test_S61.reshape(-1,1)
df_4


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
writer = pd.ExcelWriter('/Users/nronzoni/Desktop/single loop/SupportVectorRegression_prediction_flow_1HOUR.xlsx', engine='xlsxwriter')

# Write each dataframe to a different worksheet.
df_1.to_excel(writer, sheet_name='10-2-2014 morning')
df_2.to_excel(writer, sheet_name='12-2-2014 morning')
df_3.to_excel(writer, sheet_name='22-3-2014 afternoon')
df_4.to_excel(writer, sheet_name='15-8-2014 afternoon')
df_5.to_excel(writer, sheet_name='10-9-2014 morning')

# Close the Pandas Excel writer and output the Excel file.
writer.save()


## walk forward validation 

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


#################################################### SPEED  ##########################

# multistep classification 
S54_day=classification_pred_same_tris(multivariate_time_series_train_speed,multivariate_time_series_test_speed[23:24,:,:],30,10,10,0)
S1706_day=classification_pred_same_tris(multivariate_time_series_train_speed,multivariate_time_series_test_speed[23:24,:,:],30,10,10,1)
R169_day=classification_pred_same_tris(multivariate_time_series_train_speed,multivariate_time_series_test_speed[23:24,:,:],30,10,10,2)
S56_day=classification_pred_same_tris(multivariate_time_series_train_speed,multivariate_time_series_test_speed[23:24,:,:],30,10,10,3)
R129_day=classification_pred_same_tris(multivariate_time_series_train_speed,multivariate_time_series_test_speed[23:24,:,:],30,10,10,4)
S57_day=classification_pred_same_tris(multivariate_time_series_train_speed,multivariate_time_series_test_speed[23:24,:,:],30,10,10,5)
R170_day=classification_pred_same_tris(multivariate_time_series_train_speed,multivariate_time_series_test_speed[23:24,:,:],30,10,10,6)
S1707_day=classification_pred_same_tris(multivariate_time_series_train_speed,multivariate_time_series_test_speed[23:24,:,:],30,10,10,7)
S59_day=classification_pred_same_tris(multivariate_time_series_train_speed,multivariate_time_series_test_speed[23:24,:,:],30,10,10,8)
R130_day=classification_pred_same_tris(multivariate_time_series_train_speed,multivariate_time_series_test_speed[23:24,:,:],30,10,10,9)
R171_day=classification_pred_same_tris(multivariate_time_series_train_speed,multivariate_time_series_test_speed[23:24,:,:],30,10,10,10)
S60_day=classification_pred_same_tris(multivariate_time_series_train_speed,multivariate_time_series_test_speed[23:24,:,:],30,10,10,11)
S61_day=classification_pred_same_tris(multivariate_time_series_train_speed,multivariate_time_series_test_speed[23:24,:,:],30,10,10,12)
#cluster
#prediction
S54_day[1].shape
#ground_truth
S54_day[2].shape
columns = ['S54 speed (km/h)','S54 speed (km/h) ground truth','S1706 speed (km/h)','S1706 speed (km/h) ground truth', 'R169 speed (km/h)','R169 speed (km/h) ground truth','S56 speed (km/h)','S56 speed (km/h) ground truth','R129 speed (km/h)','R129 speed (km/h) ground truth', 'S57 speed (km/h)','S57 speed (km/h) ground truth','R170 speed (km/h)','R170 speed (km/h) ground truth','S1707 speed (km/h)','S1707 speed (km/h) ground truth', 'S59 speed (km/h)','S59 speed (km/h) ground truth','R130 speed (km/h)','R130 speed (km/h) ground truth','R171 speed (km/h)','R171 speed (km/h) ground truth', 'S60 speed (km/h)','S60 speed (km/h) ground truth','S61 speed (km/h)','S61 speed (km/h) ground truth']
index=pd.date_range("08:00", periods=10, freq="6min")
df_5= pd.DataFrame(index=index.time, columns=columns)
df_5

Y_pred_S54=series_train_S54_speed[1].inverse_transform(S54_day[1])
Y_test_S54=series_test_S54_speed[1].inverse_transform(S54_day[2].reshape(-1,1))
error_S54=math.sqrt(mean_squared_error(Y_test_S54,Y_pred_S54.reshape(-1,1)))
Y_pred_S1706=series_train_S1706_speed[1].inverse_transform(S1706_day[1])
Y_test_S1706=series_test_S1706_speed[1].inverse_transform(S1706_day[2].reshape(-1,1))
error_S1706=math.sqrt(mean_squared_error(Y_test_S1706,Y_pred_S1706.reshape(-1,1)))
Y_pred_R169=series_train_R169_speed[1].inverse_transform(R169_day[1])
Y_test_R169=series_test_R169_speed[1].inverse_transform(R169_day[2].reshape(-1,1))
error_R169=math.sqrt(mean_squared_error(Y_test_R169,Y_pred_R169.reshape(-1,1)))
Y_pred_S56=series_train_S56_speed[1].inverse_transform(S56_day[1])
Y_test_S56=series_test_S56_speed[1].inverse_transform(S56_day[2].reshape(-1,1))
error_S56=math.sqrt(mean_squared_error(Y_test_S56,Y_pred_S56.reshape(-1,1)))
Y_pred_R129=series_train_R129_speed[1].inverse_transform(R129_day[1])
Y_test_R129=series_test_R129_speed[1].inverse_transform(R129_day[2].reshape(-1,1))
error_R129=math.sqrt(mean_squared_error(Y_test_R129,Y_pred_R129.reshape(-1,1)))
Y_pred_S57=series_train_S57_speed[1].inverse_transform(S57_day[1])
Y_test_S57=series_test_S57_speed[1].inverse_transform(S57_day[2].reshape(-1,1))
error_S57=math.sqrt(mean_squared_error(Y_test_S57,Y_pred_S57.reshape(-1,1)))
Y_pred_R170=series_train_R170_speed[1].inverse_transform(R170_day[1])
Y_test_R170=series_test_R170_speed[1].inverse_transform(R170_day[2].reshape(-1,1))
error_R170=math.sqrt(mean_squared_error(Y_test_R170,Y_pred_R170.reshape(-1,1)))
Y_pred_S1707=series_train_S1707_speed[1].inverse_transform(S1707_day[1])
Y_test_S1707=series_test_S1707_speed[1].inverse_transform(S1707_day[2].reshape(-1,1))
error_S1707=math.sqrt(mean_squared_error(Y_test_S1707,Y_pred_S1707.reshape(-1,1)))
Y_pred_S59=series_train_S59_speed[1].inverse_transform(S59_day[1])
Y_test_S59=series_test_S59_speed[1].inverse_transform(S59_day[2].reshape(-1,1))
error_S59=math.sqrt(mean_squared_error(Y_test_S59,Y_pred_S59.reshape(-1,1)))
Y_pred_R130=series_train_R130_speed[1].inverse_transform(R130_day[1])
Y_test_R130=series_test_R130_speed[1].inverse_transform(R130_day[2].reshape(-1,1))
error_R130=math.sqrt(mean_squared_error(Y_test_R130,Y_pred_R130.reshape(-1,1)))
Y_pred_R171=series_train_R171_speed[1].inverse_transform(R171_day[1])
Y_test_R171=series_test_R171_speed[1].inverse_transform(R171_day[2].reshape(-1,1))
error_R171=math.sqrt(mean_squared_error(Y_test_R171,Y_pred_R171.reshape(-1,1)))
Y_pred_S60=series_train_S60_speed[1].inverse_transform(S60_day[1])
Y_test_S60=series_test_S60_speed[1].inverse_transform(S60_day[2].reshape(-1,1))
error_S60=math.sqrt(mean_squared_error(Y_test_S60,Y_pred_S60.reshape(-1,1)))
Y_pred_S61=series_train_S61_speed[1].inverse_transform(S61_day[1])
Y_test_S61=series_test_S61_speed[1].inverse_transform(S61_day[2].reshape(-1,1))
error_S61=math.sqrt(mean_squared_error(Y_test_S61,Y_pred_S61.reshape(-1,1)))




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
writer = pd.ExcelWriter('/Users/nronzoni/Desktop/single loop/Classification_prediction_speed_1HOUR.xlsx', engine='xlsxwriter')

# Write each dataframe to a different worksheet.
df_1.to_excel(writer, sheet_name='10-2-2014 morning')
df_2.to_excel(writer, sheet_name='12-2-2014 morning')
df_3.to_excel(writer, sheet_name='22-3-2014 afternoon')
df_4.to_excel(writer, sheet_name='15-8-2014 afternoon')
df_5.to_excel(writer, sheet_name='10-9-2014 morning')

# Close the Pandas Excel writer and output the Excel file.
writer.save()


# multistep SVR prediction 

first_day_S54=SVR_pred_d_speed_bis(multivariate_time_series_train_speed,multivariate_time_series_test_speed[2:3,:,:],30,10,10,0)
first_day_S1706=SVR_pred_d_speed_bis(multivariate_time_series_train_speed,multivariate_time_series_test_speed[2:3,:,:],30,10,10,1)
first_day_R169=SVR_pred_d_speed_bis(multivariate_time_series_train_speed,multivariate_time_series_test_speed[2:3:,:],30,10,10,2)
first_day_S56=SVR_pred_d_speed_bis(multivariate_time_series_train_speed,multivariate_time_series_test_speed[2:3,:,:],30,10,10,3)
first_day_R129=SVR_pred_d_speed_bis(multivariate_time_series_train_speed,multivariate_time_series_test_speed[2:3,:,:],30,10,10,4)
first_day_S57=SVR_pred_d_speed_bis(multivariate_time_series_train_speed,multivariate_time_series_test_speed[2:3,:,:],30,10,10,5)
first_day_R170=SVR_pred_d_speed_bis(multivariate_time_series_train_speed,multivariate_time_series_test_speed[2:3,:,:],30,10,10,6)
first_day_S1707=SVR_pred_d_speed_bis(multivariate_time_series_train_speed,multivariate_time_series_test_speed[2:3,:,:],30,10,10,7)
first_day_S59=SVR_pred_d_speed_bis(multivariate_time_series_train_speed,multivariate_time_series_test_speed[2:3,:,:],30,10,10,8)
first_day_R130=SVR_pred_d_speed_bis(multivariate_time_series_train_speed,multivariate_time_series_test_speed[2:3,:,:],30,10,10,9)
first_day_R171=SVR_pred_d_speed_bis(multivariate_time_series_train_speed,multivariate_time_series_test_speed[2:3,:,:],30,10,10,10)
first_day_S60=SVR_pred_d_speed_bis(multivariate_time_series_train_speed,multivariate_time_series_test_speed[2:3,:,:],30,10,10,11)
first_day_S61=SVR_pred_d_speed_bis(multivariate_time_series_train_speed,multivariate_time_series_test_speed[2:3,:,:],30,10,10,12)


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



columns = ['S54 speed (km/h)','S54 speed (km/h) ground truth','S1706 speed (km/h)','S1706 speed (km/h) ground truth', 'R169 speed (km/h)','R169 speed (km/h) ground truth','S56 speed (km/h)','S56 speed (km/h) ground truth','R129 speed (km/h)','R129 speed (km/h) ground truth', 'S57 speed (km/h)','S57 speed (km/h) ground truth','R170 speed (km/h)','R170 speed (km/h) ground truth','S1707 speed (km/h)','S1707 speed (km/h) ground truth', 'S59 speed (km/h)','S59 speed (km/h) ground truth','R130 speed (km/h)','R130 speed (km/h) ground truth','R171 speed (km/h)','R171 speed (km/h) ground truth', 'S60 speed (km/h)','S60 speed (km/h) ground truth','S61 speed (km/h)','S61 speed (km/h) ground truth']
index=pd.date_range("08:00", periods=10, freq="6min")
df_2= pd.DataFrame(index=index.time, columns=columns)
df_2
df_2['S54 speed (km/h)']=Y_pred_S54.reshape(-1,1)
df_2['S54 speed (km/h) ground truth']=Y_test_S54.reshape(-1,1)
df_2['S1706 speed (km/h)']=Y_pred_S1706.reshape(-1,1)
df_2['S1706 speed (km/h) ground truth']=Y_test_S1706.reshape(-1,1)
df_2['R169 speed (km/h)']=Y_pred_R169.reshape(-1,1)
df_2['R169 speed (km/h) ground truth']=Y_test_R169.reshape(-1,1)
df_2['S56 speed (km/h)']=Y_pred_S56.reshape(-1,1)
df_2['S56 speed (km/h) ground truth']=Y_test_S56.reshape(-1,1)
df_2['R129 speed (km/h)']=Y_pred_R129.reshape(-1,1)
df_2['R129 speed (km/h) ground truth']=Y_test_R129.reshape(-1,1)
df_2['S57 speed (km/h)']=Y_pred_S57.reshape(-1,1)
df_2['S57 speed (km/h) ground truth']=Y_test_S57.reshape(-1,1)
df_2['R170 speed (km/h)']=Y_pred_R170.reshape(-1,1)
df_2['R170 speed (km/h) ground truth']=Y_test_R170.reshape(-1,1)
df_2['S1707 speed (km/h)']=Y_pred_S1707.reshape(-1,1)
df_2['S1707 speed (km/h) ground truth']=Y_test_S1707.reshape(-1,1)
df_2['S59 speed (km/h)']=Y_pred_S59.reshape(-1,1)
df_2['S59 speed (km/h) ground truth']=Y_test_S59.reshape(-1,1)
df_2['R130 speed (km/h)']=Y_pred_R130.reshape(-1,1)
df_2['R130 speed (km/h) ground truth']=Y_test_R130.reshape(-1,1)
df_2['R171 speed (km/h)']=Y_pred_R171.reshape(-1,1)
df_2['R171 speed (km/h) ground truth']=Y_test_R171.reshape(-1,1)
df_2['S60 speed (km/h)']=Y_pred_S60.reshape(-1,1)
df_2['S60 speed (km/h) ground truth']=Y_test_S60.reshape(-1,1)
df_2['S61 speed (km/h)']=Y_pred_S61.reshape(-1,1)
df_2['S61 speed (km/h) ground truth']=Y_test_S61.reshape(-1,1)
df_2



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

writer = pd.ExcelWriter('/Users/nronzoni/Desktop/single loop/SupportVectorRegression_prediction_speed_1HOUR.xlsx', engine='xlsxwriter')

# Write each dataframe to a different worksheet.
df_1.to_excel(writer, sheet_name='10-2-2014 morning')
df_2.to_excel(writer, sheet_name='12-2-2014 morning')
df_3.to_excel(writer, sheet_name='22-3-2014 afternoon')
df_4.to_excel(writer, sheet_name='15-8-2014 afternoon')
df_5.to_excel(writer, sheet_name='10-9-2014 morning')

# Close the Pandas Excel writer and output the Excel file.
writer.save()


#walk forward validation prediction 


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


















