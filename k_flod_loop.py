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
epochs = 1

lr=0.015
momentum=0.8
decay=0.01

validation_split=0.2
steps_per_epoch=1
validation_steps=1
bunch_steps = 1
snapshot = 1
circle_num = 1

print('set default parameters')
#------------------------adjustable parameters-----------------------#
#k_flod_num = [2,3,4,5,6,7,8,9,10]
k_flod_num = [2,3,4]
our_class_num=5
CWRU_class_num=6

artificial_feature_method = 2

pca_parameters = 27

label_name1 = ['NORMAL', 'IR', 'OR', 'BALL', 'JOINT']
label_name2 = ['NORMAL', 'BALL', 'IR', 'OR_3', 'OR_6', 'OR_12']

result_save_path = '/home/silver-bullet/newpaper/code0115/k_flod_model'

table_title = ['LiftingNet', 'SVM 9AF', 'XGBoost 9AF', 'XGBoost 9AF & LF', 'SVM 19AF', 'XGBoost 19AF', 'XGBoost 19AF & LF']

#-------------------------set read name-----------------------------#

data_path = ['/home/silver-bullet/newpaper/data/dataset/','/home/silver-bullet/newpaper/data/CWRUdataset']

CWRU_cutsize = 1024
our_cutsize = 256

CWRU_input_shape = (CWRU_cutsize,2)
our_input_shape = (640*circle_num,3)

CWRU_channel = 2
our_channel = 3

CWRU_dataset, CWRU_label = load_CWRU_data(data_path[1])
our_dataset, our_label = load_dataset(data_path[0], circle_num=circle_num, cutsize=our_cutsize)

CWRU_dataset, CWRU_label = mix_data(CWRU_dataset, CWRU_label)
our_dataset, our_label = mix_data(our_dataset, our_label)

print('set adjustable parameters')



#---------------------------start the loop---------------------------#

for k in k_flod_num:
    print('\nstart ' + str(k) + 'th flod')
    CWRU_KF = KFold(n_splits=k)
    our_KF = KFold(n_splits=k)
    
    our_flod_snapshot_num = 1
    CWRU_flod_snapshot_num = 1

    model_save_head_name = result_save_path+'/k_flod_result/'
    cwru_result_excel_name = result_save_path + '/CWRU_all_model_'+str(k)+'_flod_result.xlsx'
    our_result_excel_name = result_save_path + '/our_all_model_'+str(k)+'_flod_result.xlsx'

    cwru_accuracy_table = []
    our_accuracy_table = []

    for CWRU_train_index,CWRU_test_index in CWRU_KF.split(CWRU_dataset):
        CWRU_x_train, CWRU_x_test = CWRU_dataset[CWRU_train_index], CWRU_dataset[CWRU_test_index]
        print(CWRU_x_train.shape)
        print(CWRU_x_test.shape)
        CWRU_y_train, CWRU_y_test = CWRU_label[CWRU_train_index], CWRU_label[CWRU_test_index]

        CWRU_model_snapshot_head_name = result_save_path + '/' + str(k) + '_flod_snapshot/CWRU_data_model_'+ str(k) + '_flod_' + str(CWRU_flod_snapshot_num) + 'th_'
        CWRU_model_save_name = model_save_head_name + 'CWRU_data_model_'+ str(k) + '_flod_' + str(CWRU_flod_snapshot_num) + 'th_model.h5'

        CWRU_model_loss_map_name = model_save_head_name + 'CWRU_data_model_'+ str(k) + '_flod_' + str(CWRU_flod_snapshot_num) + 'th_model_loss_map.png'
        CWRU_model_acc_map_name = model_save_head_name + 'CWRU_data_model_'+ str(k) + '_flod_' + str(CWRU_flod_snapshot_num) + 'th_model_acc.png'

        CWRU_model_confusion_matrix_name = model_save_head_name + 'CWRU_data_LiftingNet_model_'+ str(k) + '_flod_' + str(CWRU_flod_snapshot_num) + 'th_model_confusion_matrix.png'
        
        CWRU_svm9af_model_confusion_matrix_name = model_save_head_name + 'CWRU_data_svm9af_model_'+ str(k) + '_flod_' + str(CWRU_flod_snapshot_num) + 'th_model_confusion_matrix.png'
        CWRU_svm19af_model_confusion_matrix_name = model_save_head_name + 'CWRU_data_svm19af_model_'+ str(k) + '_flod_' + str(CWRU_flod_snapshot_num) + 'th_model_confusion_matrix.png'
        CWRU_xgb9af_model_confusion_matrix_name = model_save_head_name + 'CWRU_data_xgb9af_model_'+ str(k) + '_flod_' + str(CWRU_flod_snapshot_num) + 'th_model_confusion_matrix.png'
        CWRU_xgb19af_model_confusion_matrix_name = model_save_head_name + 'CWRU_data_xgb19af_model_'+ str(k) + '_flod_' + str(CWRU_flod_snapshot_num) + 'th_model_confusion_matrix.png'
        CWRU_xgb9af_and_lf_model_confusion_matrix_name = model_save_head_name + 'CWRU_data_xgb9af_and_lf_model_'+ str(k) + '_flod_' + str(CWRU_flod_snapshot_num) + 'th_model_confusion_matrix.png'
        CWRU_xgb19af_and_lf_model_confusion_matrix_name = model_save_head_name + 'CWRU_data_xgb19af_and_lf_model_'+ str(k) + '_flod_' + str(CWRU_flod_snapshot_num) + 'th_model_confusion_matrix.png'
        
        CWRU_liftnet = create_Standard_LiftNet_CWRU(class_num = CWRU_class_num, 
                                                    channel = CWRU_channel, 
                                                    cut_size = CWRU_cutsize, 
                                                    input_shape = CWRU_input_shape,
                                                    lr=lr, 
                                                    momentum = momentum, 
                                                    decay=decay)
        # train the LiftingNet
        loss, val_loss, acc, val_acc = CWRU_liftnet.train(CWRU_x_train,CWRU_y_train,
                                                    validation_split=validation_split,
                                                    epochs=epochs,
                                                    batch_size=bunch_steps,
                                                    steps_per_epoch=steps_per_epoch,
                                                    validation_steps=validation_steps,
                                                    snapshot = snapshot,
                                                    head_of_name = CWRU_model_snapshot_head_name,
                                                    circle_num = circle_num)
        ml2.plot_loss_map(loss, val_loss,loss_map_name=CWRU_model_loss_map_name)
        ml2.plot_acc_map(acc,val_acc,acc_map_name=CWRU_model_acc_map_name)
        # LiftingNet predict
        CWRU_LiftingNet_pred_y = CWRU_liftnet.predict(CWRU_x_test, steps=1)
        CWRU_LiftingNet_pred_y = ml2.pre_to_index(CWRU_LiftingNet_pred_y)
        CWRU_LiftingNet_accuracy = calculation_the_accuracy(CWRU_LiftingNet_pred_y,CWRU_y_test)
        plot_confusion_matrix(CWRU_LiftingNet_pred_y, CWRU_y_test, label_name= label_name2,save_name=CWRU_model_confusion_matrix_name)
        #Feature extraction
        af9_train = feature_extractor2(CWRU_x_train)
        af9_test = feature_extractor2(CWRU_x_test)
        af19_train = feature_extractor(CWRU_x_train)
        af19_test = feature_extractor(CWRU_x_test)
        
        
        train_num = CWRU_x_train.shape[0]
        test_num =CWRU_x_test.shape[0]
        print('train_num: ',train_num)
        print('test_num: ',test_num)

        af9_train = af9_train.reshape(train_num, -1)
        af9_test = af9_test.reshape(test_num, -1)
        af19_train = af19_train.reshape(train_num, -1)
        af19_test = af19_test.reshape(test_num, -1)
        print('af19_train.shape: ', af19_train.shape)
        print('af19_test.shape: ', af19_test.shape)

        CWRU_lf_train = CWRU_liftnet.feature_extractor(CWRU_x_train)
        CWRU_lf_test = CWRU_liftnet.feature_extractor(CWRU_x_test)
        print('extracted LF feature')
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            CWRU_lf_train = CWRU_lf_train.eval()
            CWRU_lf_test = CWRU_lf_test.eval()
        print('finished feature extract')
        CWRU_pca_feature_extractor = PCA(n_components=pca_parameters)
        CWRU_pca_feature_extractor.fit(CWRU_lf_train)
        CWRU_pca_train = CWRU_pca_feature_extractor.transform(CWRU_lf_train)
        CWRU_pca_test = CWRU_pca_feature_extractor.transform(CWRU_lf_test)
        print('CWRU_pca_train.shape: ', CWRU_pca_train.shape)
        print('CWRU_pca_test.shape: ', CWRU_pca_test.shape)

        CWRU_x_train_af19_and_lf = CWRU_pca_train.reshape(train_num, -1)
        CWRU_x_test_af19_and_lf = CWRU_pca_test.reshape(test_num, -1)
        CWRU_x_train_af19_and_lf = np.concatenate((CWRU_x_train_af19_and_lf, af19_train),axis=1)
        CWRU_x_test_af19_and_lf = np.concatenate((CWRU_x_test_af19_and_lf, af19_test), axis=1)

        CWRU_x_train_af9_and_lf = CWRU_pca_train.reshape(train_num, -1)
        CWRU_x_test_af9_and_lf = CWRU_pca_test.reshape(test_num, -1)
        CWRU_x_train_af9_and_lf = np.concatenate((CWRU_x_train_af9_and_lf, af9_train),axis=1)
        CWRU_x_test_af9_and_lf = np.concatenate((CWRU_x_test_af9_and_lf, af9_test), axis=1)

        #SVM 9 AF

        cwru_sc_9 = StandardScaler()
        cwru_sc_9.fit(af9_train)

        cwru_svm9_x_train = cwru_sc_9.transform(af9_train)
        cwru_svm9_x_test = cwru_sc_9.transform(af9_test)

        cwru_svm9_y_train = CWRU_y_train.reshape((CWRU_y_train.shape[0],))
        cwru_svm9_y_test = CWRU_y_test.reshape((CWRU_y_test.shape[0],))

        cwru_svm9 = SVC(kernel='linear',C=1.0,random_state= 0)
        cwru_svm9.fit(cwru_svm9_x_train, cwru_svm9_y_train)

        cwru_svm9_predict = cwru_svm9.predict(cwru_svm9_x_test)
        cwru_svm9_accuracy = calculation_the_accuracy(cwru_svm9_predict, CWRU_y_test)
        plot_confusion_matrix(cwru_svm9_predict, CWRU_y_test, label_name= label_name2, save_name=CWRU_svm9af_model_confusion_matrix_name)

        #SVM 19 AF

        cwru_sc_19 = StandardScaler()
        cwru_sc_19.fit(af19_train)

        cwru_svm19_x_train = cwru_sc_19.transform(af19_train)
        cwru_svm19_x_test = cwru_sc_19.transform(af19_test)

        cwru_svm19_y_train = CWRU_y_train.reshape((CWRU_y_train.shape[0],))
        cwru_svm19_y_test = CWRU_y_test.reshape((CWRU_y_test.shape[0],))

        cwru_svm19 = SVC(kernel='linear',C=1.0,random_state= 0)
        cwru_svm19.fit(cwru_svm19_x_train, cwru_svm19_y_train)

        cwru_svm19_predict = cwru_svm19.predict(cwru_svm19_x_test)
        cwru_svm19_accuracy = calculation_the_accuracy(cwru_svm19_predict, CWRU_y_test)
        plot_confusion_matrix(cwru_svm19_predict, CWRU_y_test, label_name= label_name2, save_name=CWRU_svm19af_model_confusion_matrix_name)

        #XGBoost 9 AF

        cwru_xgb9 = XGBClassifier(learning_rate=0.1,
                                n_estimators=1000,         # 树的个数--1000棵树建立xgboost
                                max_depth=16,               # 树的深度
                                min_child_weight = 1,      # 叶子节点最小权重
                                gamma=0.1,                  # 惩罚项中叶子结点个数前的参数
                                subsample=0.8,             # 随机选择80%样本建立决策树
                                colsample_btree=0.8,       # 随机选择80%特征建立决策树
                                objective='multi:softmax', # 指定损失函数
                                scale_pos_weight=1,        # 解决样本个数不平衡的问题
                                )
        cwru_xgb9_y_train = CWRU_y_train.reshape((CWRU_y_train.shape[0],))
        cwru_xgb9_y_test = CWRU_y_test.reshape((CWRU_y_test.shape[0],))

        cwru_xgb9.fit(af9_train, cwru_xgb9_y_train)
        cwru_xgb9_predict = cwru_xgb9.predict(af9_test)
        cwru_xgb9_accuracy = calculation_the_accuracy(cwru_xgb9_predict, CWRU_y_test)
        plot_confusion_matrix(cwru_xgb9_predict, CWRU_y_test, label_name= label_name2, save_name=CWRU_xgb9af_model_confusion_matrix_name)
        
        #XGBoost 19 AF

        cwru_xgb19 = XGBClassifier(learning_rate=0.1,
                                n_estimators=1000,         # 树的个数--1000棵树建立xgboost
                                max_depth=16,               # 树的深度
                                min_child_weight = 1,      # 叶子节点最小权重
                                gamma=0.1,                  # 惩罚项中叶子结点个数前的参数
                                subsample=0.8,             # 随机选择80%样本建立决策树
                                colsample_btree=0.8,       # 随机选择80%特征建立决策树
                                objective='multi:softmax', # 指定损失函数
                                scale_pos_weight=1,        # 解决样本个数不平衡的问题
                                )
        cwru_xgb19_y_train = CWRU_y_train.reshape((CWRU_y_train.shape[0],))
        cwru_xgb19_y_test = CWRU_y_test.reshape((CWRU_y_test.shape[0],))

        cwru_xgb19.fit(af19_train, cwru_xgb19_y_train)
        cwru_xgb19_predict = cwru_xgb19.predict(af19_test)
        cwru_xgb19_accuracy = calculation_the_accuracy(cwru_xgb19_predict, CWRU_y_test)
        plot_confusion_matrix(cwru_xgb19_predict, CWRU_y_test, label_name= label_name2, save_name=CWRU_xgb19af_model_confusion_matrix_name)

        #XGBoost 9AF+LF

        cwru_xgb9_lf = XGBClassifier(learning_rate=0.1,
                                n_estimators=1000,         # 树的个数--1000棵树建立xgboost
                                max_depth=16,               # 树的深度
                                min_child_weight = 1,      # 叶子节点最小权重
                                gamma=0.1,                  # 惩罚项中叶子结点个数前的参数
                                subsample=0.8,             # 随机选择80%样本建立决策树
                                colsample_btree=0.8,       # 随机选择80%特征建立决策树
                                objective='multi:softmax', # 指定损失函数
                                scale_pos_weight=1,        # 解决样本个数不平衡的问题
                                )
        cwru_xgb9_lf_y_train = CWRU_y_train.reshape((CWRU_y_train.shape[0],))
        cwru_xgb9_lf_y_test = CWRU_y_test.reshape((CWRU_y_test.shape[0],))

        cwru_xgb9_lf.fit(CWRU_x_train_af9_and_lf, cwru_xgb9_lf_y_train)
        cwru_xgb9_lf_predict = cwru_xgb9_lf.predict(CWRU_x_test_af9_and_lf)
        cwru_xgb9_lf_accuracy = calculation_the_accuracy(cwru_xgb9_lf_predict, CWRU_y_test)
        plot_confusion_matrix(cwru_xgb9_lf_predict, CWRU_y_test, label_name= label_name2, save_name=CWRU_xgb9af_and_lf_model_confusion_matrix_name)

        #XGBoost 19AF+LF
        cwru_xgb19_lf = XGBClassifier(learning_rate=0.1,
                                n_estimators=1000,         # 树的个数--1000棵树建立xgboost
                                max_depth=16,               # 树的深度
                                min_child_weight = 1,      # 叶子节点最小权重
                                gamma=0.1,                  # 惩罚项中叶子结点个数前的参数
                                subsample=0.8,             # 随机选择80%样本建立决策树
                                colsample_btree=0.8,       # 随机选择80%特征建立决策树
                                objective='multi:softmax', # 指定损失函数
                                scale_pos_weight=1,        # 解决样本个数不平衡的问题
                                )
        cwru_xgb19_lf_y_train = CWRU_y_train.reshape((CWRU_y_train.shape[0],))
        cwru_xgb19_lf_y_test = CWRU_y_test.reshape((CWRU_y_test.shape[0],))

        cwru_xgb19_lf.fit(CWRU_x_train_af19_and_lf, cwru_xgb19_lf_y_train)
        cwru_xgb19_lf_predict = cwru_xgb19_lf.predict(CWRU_x_test_af19_and_lf)
        cwru_xgb19_lf_accuracy = calculation_the_accuracy(cwru_xgb19_lf_predict, CWRU_y_test)
        plot_confusion_matrix(cwru_xgb19_lf_predict, CWRU_y_test, label_name= label_name2, save_name=CWRU_xgb19af_and_lf_model_confusion_matrix_name)


        cwru_accuracy = [CWRU_LiftingNet_accuracy, cwru_svm9_accuracy, cwru_xgb9_accuracy, cwru_xgb9_lf_accuracy, cwru_svm19_accuracy, cwru_xgb19_accuracy, cwru_xgb19_lf_accuracy]
        cwru_accuracy_table.append(cwru_accuracy)
    
        CWRU_flod_snapshot_num = CWRU_flod_snapshot_num + 1
    cwru_table = pd.DataFrame(cwru_accuracy_table,columns = table_title)
    cwru_table.to_excel(cwru_result_excel_name,index=True)
        
    
    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

    for our_train_index,our_test_index in our_KF.split(our_dataset):
        our_x_train, our_x_test = our_dataset[our_train_index], our_dataset[our_test_index]
        print(our_x_train.shape)
        print(our_x_test.shape)
        our_y_train, our_y_test = our_label[our_train_index], our_label[our_test_index]

        our_model_snapshot_head_name = result_save_path + '/' + str(k) + '_flod_snapshot/our_data_model_'+ str(k) + '_flod_' + str(our_flod_snapshot_num) + 'th_'

        our_model_save_name = model_save_head_name + 'our_data_model_'+ str(k) + '_flod_' + str(our_flod_snapshot_num) + 'th_model.h5'

        our_model_loss_map_name = model_save_head_name + 'our_data_model_'+ str(k) + '_flod_' + str(our_flod_snapshot_num) + 'th_model_loss_map.png'
        our_model_acc_map_name = model_save_head_name + 'our_data_model_'+ str(k) + '_flod_' + str(our_flod_snapshot_num) + 'th_model_acc.png'

        our_model_confusion_matrix_name = model_save_head_name + 'our_data_LiftingNet_model_'+ str(k) + '_flod_' + str(our_flod_snapshot_num) + 'th_model_confusion_matrix.png'
        
        our_svm9af_model_confusion_matrix_name = model_save_head_name + 'our_data_svm9af_model_'+ str(k) + '_flod_' + str(our_flod_snapshot_num) + 'th_model_confusion_matrix.png'
        our_svm19af_model_confusion_matrix_name = model_save_head_name + 'our_data_svm19af_model_'+ str(k) + '_flod_' + str(our_flod_snapshot_num) + 'th_model_confusion_matrix.png'
        our_xgb9af_model_confusion_matrix_name = model_save_head_name + 'our_data_xgb9af_model_'+ str(k) + '_flod_' + str(our_flod_snapshot_num) + 'th_model_confusion_matrix.png'
        our_xgb19af_model_confusion_matrix_name = model_save_head_name + 'our_data_xgb19af_model_'+ str(k) + '_flod_' + str(our_flod_snapshot_num) + 'th_model_confusion_matrix.png'
        our_xgb9af_and_lf_model_confusion_matrix_name = model_save_head_name + 'our_data_xgb9af_and_lf_model_'+ str(k) + '_flod_' + str(our_flod_snapshot_num) + 'th_model_confusion_matrix.png'
        our_xgb19af_and_lf_model_confusion_matrix_name = model_save_head_name + 'our_data_xgb19af_and_lf_model_'+ str(k) + '_flod_' + str(our_flod_snapshot_num) + 'th_model_confusion_matrix.png'

        our_liftnet = create_Standard_LiftNet(class_num = our_class_num, 
                                            channel = our_channel, 
                                            circle_num =circle_num, 
                                            input_shape = our_input_shape,
                                            lr=lr, 
                                            momentum = momentum, 
                                            decay=decay)
# train the LiftingNet
        loss, val_loss, acc, val_acc = our_liftnet.train(our_x_train,our_y_train,
                                                    validation_split=validation_split,
                                                    epochs=epochs,
                                                    batch_size=bunch_steps,
                                                    steps_per_epoch=steps_per_epoch,
                                                    validation_steps=validation_steps,
                                                    snapshot = snapshot,
                                                    head_of_name = CWRU_model_snapshot_head_name,
                                                    circle_num = circle_num)
        ml2.plot_loss_map(loss, val_loss,loss_map_name=our_model_loss_map_name)
        ml2.plot_acc_map(acc,val_acc,acc_map_name=our_model_acc_map_name)

        # LiftingNet predict
        our_LiftingNet_pred_y = our_liftnet.predict(our_x_test, steps=1)
        our_LiftingNet_pred_y = ml2.pre_to_index(our_LiftingNet_pred_y)
        our_LiftingNet_accuracy = calculation_the_accuracy(our_LiftingNet_pred_y, our_y_test)
        plot_confusion_matrix(our_LiftingNet_pred_y, our_y_test, label_name= label_name1,save_name=our_model_confusion_matrix_name)

        #Feature extraction
        af9_train = feature_extractor2(our_x_train)
        af9_test = feature_extractor2(our_x_test)
        af19_train = feature_extractor(our_x_train)
        af19_test = feature_extractor(our_x_test)
        
        train_num = our_x_train.shape[0]
        test_num =our_x_test.shape[0]
        print('train_num: ',train_num)
        print('test_num: ',test_num)

        af9_train = af9_train.reshape(train_num, -1)
        af9_test = af9_test.reshape(test_num, -1)
        af19_train = af19_train.reshape(train_num, -1)
        af19_test = af19_test.reshape(test_num, -1)

        our_lf_train = our_liftnet.feature_extractor(our_x_train)
        our_lf_test = our_liftnet.feature_extractor(our_x_test)
        print('extracted LF feature')
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            our_lf_train = our_lf_train.eval()
            our_lf_test = our_lf_test.eval()
        print('finished feature extract')
        our_pca_feature_extractor = PCA(n_components=pca_parameters)
        our_pca_feature_extractor.fit(our_lf_train)
        our_pca_train = our_pca_feature_extractor.transform(our_lf_train)
        our_pca_test = our_pca_feature_extractor.transform(our_lf_test)
        print('our_pca_train.shape: ', our_pca_train.shape)
        print('our_pca_test.shape: ', our_pca_test.shape)

        our_x_train_af19_and_lf = our_pca_train.reshape(train_num, -1)
        our_x_test_af19_and_lf = our_pca_test.reshape(test_num, -1)
        our_x_train_af19_and_lf = np.concatenate((our_x_train_af19_and_lf, af19_train),axis=1)
        our_x_test_af19_and_lf = np.concatenate((our_x_test_af19_and_lf, af19_test), axis=1)

        our_x_train_af9_and_lf = our_pca_train.reshape(train_num, -1)
        our_x_test_af9_and_lf = our_pca_test.reshape(test_num, -1)
        our_x_train_af9_and_lf = np.concatenate((our_x_train_af9_and_lf, af9_train),axis=1)
        our_x_test_af9_and_lf = np.concatenate((our_x_test_af9_and_lf, af9_test), axis=1)

        #SVM 9 AF

        our_sc_9 = StandardScaler()
        our_sc_9.fit(af9_train)

        our_svm9_x_train = our_sc_9.transform(af9_train)
        our_svm9_x_test = our_sc_9.transform(af9_test)

        our_svm9_y_train = our_y_train.reshape((our_y_train.shape[0],))
        our_svm9_y_test = our_y_test.reshape((our_y_test.shape[0],))

        our_svm9 = SVC(kernel='linear',C=1.0,random_state= 0)
        our_svm9.fit(our_svm9_x_train, our_svm9_y_train)

        our_svm9_predict = our_svm9.predict(our_svm9_x_test)
        our_svm9_accuracy = calculation_the_accuracy(our_svm9_predict, our_y_test)
        plot_confusion_matrix(our_svm9_predict, our_y_test, label_name= label_name1,save_name=our_svm9af_model_confusion_matrix_name)


        #SVM 19 AF

        our_sc_19 = StandardScaler()
        our_sc_19.fit(af19_train)

        our_svm19_x_train = our_sc_19.transform(af19_train)
        our_svm19_x_test = our_sc_19.transform(af19_test)

        our_svm19_y_train = our_y_train.reshape((our_y_train.shape[0],))
        our_svm19_y_test = our_y_test.reshape((our_y_test.shape[0],))

        our_svm19 = SVC(kernel='linear',C=1.0,random_state= 0)
        our_svm19.fit(our_svm19_x_train, our_svm19_y_train)

        our_svm19_predict = our_svm19.predict(our_svm19_x_test)
        our_svm19_accuracy = calculation_the_accuracy(our_svm19_predict, our_y_test)
        plot_confusion_matrix(our_svm19_predict, our_y_test, label_name= label_name1,save_name=our_svm19af_model_confusion_matrix_name)

        #XGBoost 9 AF

        our_xgb9 = XGBClassifier(learning_rate=0.1,
                                n_estimators=1000,         # 树的个数--1000棵树建立xgboost
                                max_depth=16,               # 树的深度
                                min_child_weight = 1,      # 叶子节点最小权重
                                gamma=0.1,                  # 惩罚项中叶子结点个数前的参数
                                subsample=0.8,             # 随机选择80%样本建立决策树
                                colsample_btree=0.8,       # 随机选择80%特征建立决策树
                                objective='multi:softmax', # 指定损失函数
                                scale_pos_weight=1,        # 解决样本个数不平衡的问题
                                )
        our_xgb9_y_train = our_y_train.reshape((our_y_train.shape[0],))
        our_xgb9_y_test = our_y_test.reshape((our_y_test.shape[0],))

        our_xgb9.fit(af9_train, our_xgb9_y_train)
        our_xgb9_predict = our_xgb9.predict(af9_test)
        our_xgb9_accuracy = calculation_the_accuracy(our_xgb9_predict, our_y_test)
        plot_confusion_matrix(our_xgb9_predict, our_y_test, label_name= label_name1,save_name=our_xgb9af_model_confusion_matrix_name)

        
        #XGBoost 19 AF

        our_xgb19 = XGBClassifier(learning_rate=0.1,
                                n_estimators=1000,         # 树的个数--1000棵树建立xgboost
                                max_depth=16,               # 树的深度
                                min_child_weight = 1,      # 叶子节点最小权重
                                gamma=0.1,                  # 惩罚项中叶子结点个数前的参数
                                subsample=0.8,             # 随机选择80%样本建立决策树
                                colsample_btree=0.8,       # 随机选择80%特征建立决策树
                                objective='multi:softmax', # 指定损失函数
                                scale_pos_weight=1,        # 解决样本个数不平衡的问题
                                )
        our_xgb19_y_train = our_y_train.reshape((our_y_train.shape[0],))
        our_xgb19_y_test = our_y_test.reshape((our_y_test.shape[0],))

        our_xgb19.fit(af19_train, our_xgb19_y_train)
        our_xgb19_predict = our_xgb19.predict(af19_test)
        our_xgb19_accuracy = calculation_the_accuracy(our_xgb19_predict, our_y_test)
        plot_confusion_matrix(our_xgb19_predict, our_y_test, label_name= label_name1,save_name=our_xgb19af_model_confusion_matrix_name)

        #XGBoost 9AF+LF

        our_xgb9_lf = XGBClassifier(learning_rate=0.1,
                                n_estimators=1000,         # 树的个数--1000棵树建立xgboost
                                max_depth=16,               # 树的深度
                                min_child_weight = 1,      # 叶子节点最小权重
                                gamma=0.1,                  # 惩罚项中叶子结点个数前的参数
                                subsample=0.8,             # 随机选择80%样本建立决策树
                                colsample_btree=0.8,       # 随机选择80%特征建立决策树
                                objective='multi:softmax', # 指定损失函数
                                scale_pos_weight=1,        # 解决样本个数不平衡的问题
                                )
        our_xgb9_lf_y_train = our_y_train.reshape((our_y_train.shape[0],))
        our_xgb9_lf_y_test = our_y_test.reshape((our_y_test.shape[0],))

        our_xgb9_lf.fit(our_x_train_af9_and_lf, our_xgb9_lf_y_train)
        our_xgb9_lf_predict = our_xgb9_lf.predict(our_x_test_af9_and_lf)
        our_xgb9_lf_accuracy = calculation_the_accuracy(our_xgb9_lf_predict, our_y_test)
        plot_confusion_matrix(our_xgb9_lf_predict, our_y_test, label_name= label_name1,save_name=our_xgb9af_and_lf_model_confusion_matrix_name)


        #XGBoost 19AF+LF
        our_xgb19_lf = XGBClassifier(learning_rate=0.1,
                                n_estimators=1000,         # 树的个数--1000棵树建立xgboost
                                max_depth=16,               # 树的深度
                                min_child_weight = 1,      # 叶子节点最小权重
                                gamma=0.1,                  # 惩罚项中叶子结点个数前的参数
                                subsample=0.8,             # 随机选择80%样本建立决策树
                                colsample_btree=0.8,       # 随机选择80%特征建立决策树
                                objective='multi:softmax', # 指定损失函数
                                scale_pos_weight=1,        # 解决样本个数不平衡的问题
                                )
        our_xgb19_lf_y_train = our_y_train.reshape((our_y_train.shape[0],))
        our_xgb19_lf_y_test = our_y_test.reshape((our_y_test.shape[0],))

        our_xgb19_lf.fit(our_x_train_af19_and_lf, our_xgb19_lf_y_train)
        our_xgb19_lf_predict = our_xgb19_lf.predict(our_x_test_af19_and_lf)
        our_xgb19_lf_accuracy = calculation_the_accuracy(our_xgb19_lf_predict, our_y_test)
        plot_confusion_matrix(our_xgb19_lf_predict, our_y_test, label_name= label_name1,save_name=our_xgb19af_and_lf_model_confusion_matrix_name)


        our_accuracy = [our_LiftingNet_accuracy, our_svm9_accuracy, our_xgb9_accuracy, our_xgb9_lf_accuracy, our_svm19_accuracy, our_xgb19_accuracy, our_xgb19_lf_accuracy]
        our_accuracy_table.append(our_accuracy)

        our_flod_snapshot_num = our_flod_snapshot_num + 1
    our_table = pd.DataFrame(our_accuracy_table,columns = table_title)
    our_table.to_excel(our_result_excel_name,index=True)




print('ok')







