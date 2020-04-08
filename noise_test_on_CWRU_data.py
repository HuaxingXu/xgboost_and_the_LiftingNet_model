import tensorflow as tf
import numpy as np
import math
import glob
import matplotlib
import os

import matplotlib.pyplot as plt
from matplotlib.pyplot import plot,savefig
from matplotlib.colors import ListedColormap

import xgboost as xgb
from xgboost import plot_importance
from xgboost import XGBClassifier

import mylib as ml2
from mylib import LiftNet, create_LiftNet, create_Standard_LiftNet, Standard_LiftNet, create_Standard_LiftNet_CWRU
from dataset import *

from plotly.graph_objs import Scatter,Layout
import plotly
import plotly.offline as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

import pandas as pd

import sklearn as sk
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.decomposition import PCA

import seaborn as sea

#setting offilne
plotly.offline.init_notebook_mode(connected=True)
#------------------------default parameters-----------------------#
test_rate = 0.2
epochs = 1000

lr=0.015
momentum=0.8

decay=0.01

validation_split=0.2
steps_per_epoch=1
validation_steps=1
bunch_steps = 100
snapshot = 500
bunch_steps = 100
snapshot = 500


circle_num=1

whether_use_CWRU_data=1

label_name1 = ['NORMAL', 'IR', 'OR', 'BALL', 'JOINT']
label_name2 = ['NORMAL', 'BALL', 'IR', 'OR_3', 'OR_6', 'OR_12']

if whether_use_CWRU_data == 1:
    label_name = label_name2
    cutsize = 1024
    channel = 2
    save_map_end_name='_noise_in_CWRU_data.png'
    save_result_end_name='_noise_in_CWRU_data.txt'
    result_save_path = '/home/silver-bullet/newpaper/result/noise_test_result/CWRU_data'
else:
    label_name = label_name1
    cutsize = 256
    channel = 3
    save_map_end_name='_noise_in_our_data.png'
    save_result_end_name='_noise_in_our_data.txt'
    result_save_path = '/home/silver-bullet/newpaper/result/noise_test_result/our_data'
input_shape = (cutsize,channel)
class_num = len(label_name)
print('set default parameters')

#------------------------adjustable parameters-----------------------#

steps = 1000
noise_scales = 0.5
LiftingNet_noise_scale = 0


whether_expansion_train_data = 1
whether_append_expansion_train_data = 1
whether_append_expansion_test_data = 0

expansion_data_number = 500

"""
if whether_use_CWRU_data==1:
    noise_scales = LiftingNet_noise_scale
"""
#artificial_feature_method: 1 is 19 features, 2 is 9 features
artificial_feature_method = 2

pca_parameters = 18

#result_save_path = '/home/silver-bullet/newpaper/result/noise_test_result'

#-------------------------set read name-----------------------------#

model_path = ['/home/silver-bullet/newpaper/model/circle_1_to_6_in_our_data/', 
              '/home/silver-bullet/newpaper/model/circle_7_to_16_in_our_data_with_0.01_noise_data/',
              '/home/silver-bullet/newpaper/model/model_in_CWRU_data/']
model_head_name = ['Standard_LiftingNet_',
                   'Standard_LiftingNet_use_expansion_data__with_',
                   'Standard_expansion_CWRU_data_LiftingNet__with_',
                  'Standard_LiftingNet_use_expansion_data_',
                  'Standard_expansion_CWRU_data_LiftingNet_']

#data_path = ['/media/silverbullet/data_and_programing_file/newpaper/dataset/newdataset/','/media/silverbullet/data_and_programing_file/newpaper/dataset/CWRU/CWRUdataset']
data_path = ['/home/silver-bullet/newpaper/data/dataset/','/home/silver-bullet/newpaper/data/CWRUdataset/']


if whether_use_CWRU_data ==1:
    dataset, label = load_CWRU_data(data_path[1])
    read_model_name = model_path[2]+model_head_name[2]+str(LiftingNet_noise_scale)+'_noise_'+str(cutsize)+'_data_the_'+str(steps)+'th_snapshot_with_'+str(steps_per_epoch)+'_steps_per_epoch.h5'
    if LiftingNet_noise_scale == 0:
        read_model_name = model_path[2]+model_head_name[4]+str(cutsize)+'_data_the_'+str(steps)+'th_snapshot_with_'+str(steps_per_epoch)+'_steps_per_epoch.h5'
else:
    dataset, label = load_dataset(data_path[0],circle_num=circle_num)
    if circle_num<=6:
        model_path_index=0
        head_index = 0
        zero_noise_head_index=0
    else:
        model_path_index=1
        head_index = 1
        zero_noise_head_index=3
    read_model_name = model_path[model_path_index]+model_head_name[head_index]+str(LiftingNet_noise_scale)+'_noise_'+str(circle_num)+'_data_the_'+str(steps)+'th_snapshot_with_'+str(steps_per_epoch)+'_steps_per_epoch.h5'
    if LiftingNet_noise_scale == 0:
        read_model_name = model_path[model_path_index]+model_head_name[zero_noise_head_index]+str(circle_num)+'_data_the_'+str(steps)+'th_snapshot_with_'+str(steps_per_epoch)+'_steps_per_epoch.h5'



#------------------------data processing---------------------------------#
input_shape = (dataset.shape[1],dataset.shape[2])
channel = dataset.shape[2]
x_number = dataset.shape[0]

x_train = dataset
y_train = label
print('x_train.shape: ', x_train.shape)
print('y_train.shape: ', y_train.shape)



#------------------------expansion data parameters-----------------------#

if whether_expansion_train_data == 1:
    x_train, y_train = expansion_and_add_noise(x_train,y_train,exnumber=expansion_data_number, noise_scales=LiftingNet_noise_scale,whether_link_original_data=whether_append_expansion_train_data)
    print('data expansion')

x_test, y_test = expansion_and_add_noise(dataset, label,exnumber=expansion_data_number,noise_scales=noise_scales,whether_link_original_data=whether_append_expansion_test_data)

x_train_number = x_train.shape[0]
print('x_train_number: ',x_train_number)
x_test_number = x_test.shape[0]
print('x_test_number: ',x_test_number)


#------------------------------artificial features extraction-----------------------------#
if artificial_feature_method == 1:
    artificial_feature_of_train_data = feature_extractor(x_train)
    artificial_feature_of_test_data = feature_extractor(x_test)
else:
    artificial_feature_of_train_data = feature_extractor2(x_train)
    artificial_feature_of_test_data = feature_extractor2(x_test)
#artificial_feature_data = artificial_feature_data.reshape(x_number,-1)[:,:select_feature_numbers]
print('artificial_feature_of_train_data.shape: ', artificial_feature_of_train_data.shape)
print('artificial_feature_of_test_data.shape: ', artificial_feature_of_test_data.shape)

artificial_feature_of_train_data = artificial_feature_of_train_data.reshape(x_train_number,-1)
artificial_feature_of_test_data = artificial_feature_of_test_data.reshape(x_test_number,-1)
print('artificial_feature_of_train_data.shape: ', artificial_feature_of_train_data.shape)
print('artificial_feature_of_test_data.shape: ', artificial_feature_of_test_data.shape)

#--------------------------------load LiftingNet-------------------------------------------#

liftnet = create_Standard_LiftNet_CWRU(class_num = class_num, 
                                    channel = channel, 
                                    cut_size = cutsize, 
                                    input_shape = input_shape,
                                    lr=lr, 
                                    momentum = momentum, 
                                    decay=decay)

liftnet.load_weights(read_model_name)
print('load model')

result_txt_list = []
#--------------------------------LiftingNet Predict-------------------------------------------#
print('test LiftingNet model')
result_txt_list.append('LiftingNet result: \n')
LiftingNet_predict = liftnet.predict(x_test, steps=1)

LiftngNet_test_result = ml2.evaluate_model(LiftingNet_predict, y_test,whether_save_result=1, whether_use_CWRU_data_label=1)
result_txt_list.extend(LiftngNet_test_result)

LiftingNet_predict2 = ml2.pre_to_index(LiftingNet_predict)

ans = LiftingNet_predict2
cnt1 = 0
cnt2 = 0
for i in range(len(y_test)):
    if ans[i] == y_test[i]:
        cnt1 += 1
    else:
        cnt2 += 1
print("Accuracy: %.2f %% " % (100 * cnt1 / (cnt1 + cnt2)))
result_txt_list.append('Accuracy: '+str(100 * cnt1 / (cnt1 + cnt2))+'%\n')
result_txt_list.append('\n')

print('ans.shape: ', ans.shape)
print(ans.T)
print('y_test.shape: ', y_test.shape)
print(y_test.T)

confusion_matrix_map_name1 = result_save_path + '/LiftingNet_result_of_'+str(LiftingNet_noise_scale)+'_LiftingNet_noise_and_'+str(noise_scales)+'_noise_in_data.png'
plot_confusion_matrix(ans,y_test,label_name, save_name=confusion_matrix_map_name1)

#--------------------------------SVM with 9AF Train and Predict-------------------------------------------#
print('test svm model')
result_txt_list.append('svm 9AF result: \n')
sc = StandardScaler()
sc.fit(artificial_feature_of_train_data)

svm_x_train = sc.transform(artificial_feature_of_train_data)
svm_x_test = sc.transform(artificial_feature_of_test_data)

svm_y_train = y_train.reshape((y_train.shape[0],))
svm_y_test = y_test.reshape((y_test.shape[0],))

print('svm_y_train.shape: ', svm_y_train.shape)
print('svm_y_test.shape: ', svm_y_test.shape)

svm = SVC(kernel='linear',C=1.0,random_state= 0)

svm.fit(svm_x_train, svm_y_train)

svm_predict = svm.predict(svm_x_test)

svm_test_result = ml2.evaluate_model3(svm_predict, svm_y_test, whether_save_result=1, whether_use_CWRU_data_label=whether_use_CWRU_data)
result_txt_list.extend(svm_test_result)

ans = svm_predict
cnt1 = 0
cnt2 = 0
for i in range(len(y_test)):
    if ans[i] == y_test[i]:
        cnt1 += 1
    else:
        cnt2 += 1
print("Accuracy: %.2f %% " % (100 * cnt1 / (cnt1 + cnt2)))
result_txt_list.append('Accuracy: '+str(100 * cnt1 / (cnt1 + cnt2))+'%\n')
result_txt_list.append('\n')

print('ans.shape: ', ans.shape)
print('y_test.shape: ', y_test.shape)

confusion_matrix_map_name2 = result_save_path + '/SVM_9AF_result_of_'+str(LiftingNet_noise_scale)+'_LiftingNet_noise_and_'+str(noise_scales)+'_noise_in_data.png'
plot_confusion_matrix(ans,y_test,label_name, save_name=confusion_matrix_map_name2)

#--------------------------------XGBoost with 9AF Train and Predict-------------------------------------------#
print('test xgboost with 9AF model')
result_txt_list.append('xgboost 9AF result: \n')
# create XGBoost
xgboost_model = XGBClassifier(learning_rate=0.1,
                                n_estimators=1000,
                                max_depth=16,
                                min_child_weight = 1,
                                gamma=0.1,
                                subsample=0.8,
                                colsample_btree=0.8,
                                objective='multi:softmax',
                                scale_pos_weight=1,
                                )

xgboost_y_train = y_train.reshape((y_train.shape[0],))
xgboost_y_test = y_test.reshape((y_test.shape[0],))


# train xgboost
xgboost_model.fit(artificial_feature_of_train_data, xgboost_y_train)

xgboost_predict = xgboost_model.predict(artificial_feature_of_test_data)

xgboost_result = ml2.evaluate_model3(xgboost_predict, xgboost_y_test, whether_save_result=1, whether_use_CWRU_data_label=whether_use_CWRU_data)
result_txt_list.extend(xgboost_result)

ans = xgboost_predict
cnt1 = 0
cnt2 = 0
for i in range(len(y_test)):
    if ans[i] == y_test[i]:
        cnt1 += 1
    else:
        cnt2 += 1
print("Accuracy: %.2f %% " % (100 * cnt1 / (cnt1 + cnt2)))
result_txt_list.append('Accuracy: '+str(100 * cnt1 / (cnt1 + cnt2))+'%\n')
result_txt_list.append('\n')

print('ans.shape: ', ans.shape)
print('y_test.shape: ', y_test.shape)

confusion_matrix_map_name3 = result_save_path + '/XGBoost_with_9AF_result_of_'+str(LiftingNet_noise_scale)+'_LiftingNet_noise_and_'+str(noise_scales)+'_noise_in_data.png'
plot_confusion_matrix(ans,y_test,label_name, save_name=confusion_matrix_map_name3)

#--------------------------------XGBoost 9AF + LF Train and Predict-------------------------------------------#
print('test xgboost with 9AF and LF model')
#LiftingNet features
print('start extract feature')
feature_data_for_train = liftnet.feature_extractor(x_train)
feature_data_for_test = liftnet.feature_extractor(x_test)
print('extracted feature')
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    feature_data_for_train = feature_data_for_train.eval()
    feature_data_for_test = feature_data_for_test.eval()
print('feature_data_for_train.shape: ', feature_data_for_train.shape)
print('feature_data_for_test.shape: ', feature_data_for_test.shape)
"""
feature_data_for_train = np.reshape(feature_data_for_train,(feature_data_for_train[0],feature_data_for_train.shape[1]))
feature_data_for_test = np.reshape(feature_data_for_test,(feature_data_for_test[0],feature_data_for_test.shape[1]))

#feature_data = np.concatenate(dataset,axis=2)
print('feature_data_for_train.shape: ', feature_data_for_train.shape)
print('feature_data_for_test.shape: ', feature_data_for_test.shape)
"""
print('finished feature extract')
pca_feature_extractor = PCA(n_components=pca_parameters)
pca_feature_extractor.fit(feature_data_for_train)
pca_feature_for_train = pca_feature_extractor.transform(feature_data_for_train)
pca_feature_for_test = pca_feature_extractor.transform(feature_data_for_test)
print('pca_feature_for_train.shape: ', pca_feature_for_train.shape)
print('pca_feature_for_test.shape: ', pca_feature_for_test.shape)


#create xgboost train dataset
x_train_for_xgboost_AF_and_LF = pca_feature_for_train.reshape(x_train_number,-1)
x_test_for_xgboost_AF_and_LF = pca_feature_for_test.reshape(x_test_number,-1)

x_train_for_xgboost_AF_and_LF  = np.concatenate((x_train_for_xgboost_AF_and_LF ,artificial_feature_of_train_data),axis=1)
x_test_for_xgboost_AF_and_LF  = np.concatenate((x_test_for_xgboost_AF_and_LF ,artificial_feature_of_test_data),axis=1)

print('x_train_for_xgboost_AF_and_LF.shape: ', x_train_for_xgboost_AF_and_LF.shape)
print('x_test_for_xgboost_AF_and_LF.shape: ', x_test_for_xgboost_AF_and_LF.shape)

#create xgboost model
xgboost_model_for_AF_and_LF = XGBClassifier(learning_rate=0.1,
                                            n_estimators=1000,
                                            max_depth=16,
                                            min_child_weight = 1,
                                            gamma=0.1,
                                            subsample=0.8,
                                            colsample_btree=0.8,
                                            objective='multi:softmax',
                                            scale_pos_weight=1,
                                            )
print('create the model')
result_txt_list.append('xgboost 9AF and LF result: \n')

xgboost_aflf_y_train = y_train.reshape((y_train.shape[0],))
xgboost_aflf_y_test = y_test.reshape((y_test.shape[0],))

xgboost_model_for_AF_and_LF.fit(x_train_for_xgboost_AF_and_LF, xgboost_aflf_y_train)
print('fit the model')
xgboost_model_for_AF_and_LF_predict = xgboost_model_for_AF_and_LF.predict(x_test_for_xgboost_AF_and_LF)
print('finish predict')
xgboost_with_AF_LF_result = ml2.evaluate_model3(xgboost_model_for_AF_and_LF_predict, xgboost_aflf_y_test, whether_save_result=1, whether_use_CWRU_data_label=whether_use_CWRU_data)
result_txt_list.extend(xgboost_with_AF_LF_result)
print('save the result')
ans = xgboost_model_for_AF_and_LF_predict
cnt1 = 0
cnt2 = 0
for i in range(len(y_test)):
    if ans[i] == y_test[i]:
        cnt1 += 1
    else:
        cnt2 += 1
print("Accuracy: %.2f %% " % (100 * cnt1 / (cnt1 + cnt2)))
result_txt_list.append('Accuracy: '+str(100 * cnt1 / (cnt1 + cnt2))+'%\n')
result_txt_list.append('\n')

print('ans.shape: ', ans.shape)
print('y_test.shape: ', y_test.shape)

confusion_matrix_map_name4 = result_save_path + '/XGBoost_with_9AFandLF_result_of_'+str(LiftingNet_noise_scale)+'_LiftingNet_noise_and_'+str(noise_scales)+'_noise_in_data.png'
plot_confusion_matrix(ans,y_test,label_name, save_name=confusion_matrix_map_name4)

result_txt_name = result_save_path + '/result_of_' + str(LiftingNet_noise_scale)+'_LiftingNet_noise_and_'+str(noise_scales)+'_noise_in_data.txt'
if os.path.exists(result_txt_name):
    os.remove(result_txt_name)

f = open(result_txt_name,'w')
for info in result_txt_list:
    f.writelines(info)

