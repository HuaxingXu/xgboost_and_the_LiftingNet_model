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

result_save_path = '/media/silverbullet/data_and_programing_file/newpaper/result/transfer_learning_result'

print('set default parameters')
#------------------------adjustable parameters-----------------------#


circle_num = 2
steps = 1000
whether_use_CWRU_data = 0

whether_expansion_test_data = 1
whether_expansion_train_data = 1
expansion_data_number = 500
noise_scales = 0.01

LiftingNet_noise_scale = 0.01

whether_append_expansion_train_data = 1
whether_append_expansion_test_data = 1

if whether_use_CWRU_data==1:
    class_num = 6
else:
    class_num = 5
"""
if whether_use_CWRU_data==1:
    noise_scales = LiftingNet_noise_scale
"""
#artificial_feature_method: 1 is 19 features, 2 is 9 features
artificial_feature_method = 2

pca_parameters = 27

if whether_use_CWRU_data==1:
    result_files_last_name = 'CWRU_data_train_model_our_data_test.png'
    result_txt_last_name = 'CWRU_data_train_model_our_data_test.txt'
else:
    result_files_last_name = 'our_'+ str(circle_num)+'_circles_data_trains_model_CWRU_data_test.png'
    result_txt_last_name = 'our_'+ str(circle_num)+'_circles_data_trains_model_CWRU_data_test.txt'

label_name1 = ['NORMAL', 'IR', 'OR', 'BALL', 'JOINT']
label_name2 = ['NORMAL', 'BALL', 'IR', 'OR_3', 'OR_6', 'OR_12']

if whether_use_CWRU_data == 1:
    label_name = label_name2
else:
    label_name = label_name1

#-------------------------set read name-----------------------------#

model_path = ['/media/silverbullet/data_and_programing_file/newpaper/code/code1226/model/circle_1_to_6_in_our_data/', 
              '/media/silverbullet/data_and_programing_file/newpaper/code/code1226/model/circle_7_to_16_in_our_data_with_0.01_noise_data/',
              '/media/silverbullet/data_and_programing_file/newpaper/code/code1226/model/model_in_CWRU_data/']
model_head_name = ['Standard_LiftingNet_',
                   'Standard_LiftingNet_use_expansion_data__with_',
                   'Standard_expansion_CWRU_data_LiftingNet__with_',
                  'Standard_LiftingNet_use_expansion_data_',
                  'Standard_expansion_CWRU_data_LiftingNet_']

data_path = ['/media/silverbullet/data_and_programing_file/newpaper/dataset/newdataset/','/media/silverbullet/data_and_programing_file/newpaper/dataset/CWRU/CWRUdataset']

if whether_use_CWRU_data == 1:
    cutsize = 1024
    input_shape = (cutsize,2)
    channel = 2
    dataset, label = load_CWRU_data(data_path[1])
    read_model_name = model_path[2]+model_head_name[2]+str(LiftingNet_noise_scale)+'_noise_'+str(cutsize)+'_data_the_'+str(steps)+'th_snapshot_with_'+str(steps_per_epoch)+'_steps_per_epoch.h5'
    if LiftingNet_noise_scale == 0:
        read_model_name = model_path[2]+model_head_name[4]+str(cutsize)+'_data_the_'+str(steps)+'th_snapshot_with_'+str(steps_per_epoch)+'_steps_per_epoch.h5'
else:
    cutsize = 256
    input_shape = (640*circle_num,3)
    channel = 3
    dataset, label = load_dataset(data_path[0], circle_num=circle_num, cutsize=cutsize)
    if circle_num<7:
        read_model_name = model_path[0] + model_head_name[0]+str(circle_num)+'_data_the_'+str(steps)+'th_snapshot_with_'+str(steps_per_epoch)+'_steps_per_epoch.h5'
    else:
        read_model_name = model_path[1] + model_head_name[1]+str(LiftingNet_noise_scale)+'_noise_'+str(circle_num)+'_data_the_'+str(steps)+'th_snapshot_with_'+str(steps_per_epoch)+'_steps_per_epoch.h5'

if circle_num==8:
    read_model_name= model_path[1] + model_head_name[3] +  str(circle_num)+'_data_the_'+str(steps)+'th_snapshot_with_'+str(steps_per_epoch)+'_steps_per_epoch.h5'

print('set adjustable parameters')

#------------------------data processing---------------------------------#
input_shape = (dataset.shape[1],dataset.shape[2])
channel = dataset.shape[2]
x_number = dataset.shape[0]

x_train = dataset
y_train = label
print('x_train.shape: ', x_train.shape)
print('y_train.shape: ', y_train.shape)



CWRU_data_trans_to_us_path = '/media/silverbullet/data_and_programing_file/newpaper/dataset/CWRU_data_trans_to_our_data'
our_data_trans_to_CWRU_data_path = '/media/silverbullet/data_and_programing_file/newpaper/dataset/our_data_trans_to_CWRU_data/length_1024'


if whether_use_CWRU_data == 1:
    x_test_data_name = our_data_trans_to_CWRU_data_path+'/dataset_trans_to_CWRU_data_' + str(cutsize) + '.npy'
    y_test_data_name = our_data_trans_to_CWRU_data_path+'/label_trans_to_CWRU_data_' + str(cutsize)+'.npy'
else:
    x_test_data_name = CWRU_data_trans_to_us_path+'/CWRU_dataset_trans_to_'+str(circle_num)+'_circles_data.npy'
    y_test_data_name = CWRU_data_trans_to_us_path+'/CWRU_label_trans_to_'+str(circle_num)+'_circles_data.npy'

x_test = np.load(x_test_data_name)
y_test = np.load(y_test_data_name)

print('x_test.shape: ',x_test.shape)
print('y_test.shape: ', y_test.shape)




#------------------------expansion data parameters-----------------------#

if whether_expansion_train_data == 1:
    x_train, y_train = expansion_and_add_noise(x_train,y_train,exnumber=expansion_data_number, noise_scales=LiftingNet_noise_scale,whether_link_original_data=whether_append_expansion_train_data,is_label_sort=0)
    print('data expansion')

if whether_expansion_test_data == 1:
    x_test, y_test = expansion_and_add_noise(x_test, y_test,exnumber=expansion_data_number,noise_scales=noise_scales,whether_link_original_data=whether_append_expansion_test_data,is_label_sort=0)

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
if whether_use_CWRU_data == 1:
    liftnet = create_Standard_LiftNet_CWRU(class_num = class_num, 
                                       channel = channel, 
                                       cut_size = cutsize, 
                                       input_shape = input_shape,
                                       lr=lr, 
                                       momentum = momentum, 
                                       decay=decay)
else:
    liftnet = create_Standard_LiftNet(class_num = class_num, 
                                  channel = channel, 
                                  circle_num = circle_num, 
                                  input_shape=input_shape,
                                  lr=lr, 
                                  momentum=momentum, 
                                  decay=decay)

liftnet.load_weights(read_model_name)
print('load model')

result_txt_list = []
#--------------------------------LiftingNet Predict-------------------------------------------#
print('test LiftingNet model')
result_txt_list.append('LiftingNet result: \n')
LiftingNet_predict = liftnet.predict(x_test, steps=1)

LiftngNet_test_result = ml2.evaluate_model(LiftingNet_predict, y_test,whether_save_result=1, whether_use_CWRU_data_label=whether_use_CWRU_data)
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

confusion_matrix_map_name1 = result_save_path + '/LiftingNet_result_of_'+result_files_last_name
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

confusion_matrix_map_name2 = result_save_path + '/SVM_9AF_result_of_'+result_files_last_name
plot_confusion_matrix(ans,y_test,label_name, save_name=confusion_matrix_map_name2)


#--------------------------------XGBoost with 9AF Train and Predict-------------------------------------------#
print('test xgboost with 9AF model')
result_txt_list.append('xgboost 9AF result: \n')
# create XGBoost
xgboost_model = XGBClassifier(learning_rate=0.1,
                                n_estimators=1000,         # 树的个数--1000棵树建立xgboost
                                max_depth=16,               # 树的深度
                                min_child_weight = 1,      # 叶子节点最小权重
                                gamma=0.1,                  # 惩罚项中叶子结点个数前的参数
                                subsample=0.8,             # 随机选择80%样本建立决策树
                                colsample_btree=0.8,       # 随机选择80%特征建立决策树
                                objective='multi:softmax', # 指定损失函数
                                scale_pos_weight=1,        # 解决样本个数不平衡的问题
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

confusion_matrix_map_name3 = result_save_path + '/XGBoost_with_9AF_result_of_'+result_files_last_name
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
                                            n_estimators=1000,         # 树的个数--1000棵树建立xgboost
                                            max_depth=16,               # 树的深度
                                            min_child_weight = 1,      # 叶子节点最小权重
                                            gamma=0.1,                  # 惩罚项中叶子结点个数前的参数
                                            subsample=0.8,             # 随机选择80%样本建立决策树
                                            colsample_btree=0.8,       # 随机选择80%特征建立决策树
                                            objective='multi:softmax', # 指定损失函数
                                            scale_pos_weight=1,        # 解决样本个数不平衡的问题
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

confusion_matrix_map_name4 = result_save_path + '/XGBoost_with_9AFandLF_result_of_'+result_files_last_name
plot_confusion_matrix(ans,y_test,label_name, save_name=confusion_matrix_map_name4)

result_txt_name = result_save_path + '/result_of_' + result_txt_last_name
if os.path.exists(result_txt_name):
    os.remove(result_txt_name)

f = open(result_txt_name,'w')
for info in result_txt_list:
    f.writelines(info)

print('OK')