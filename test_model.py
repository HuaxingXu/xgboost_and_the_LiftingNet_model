import tensorflow as tf
import numpy as np
import math
import glob
import matplotlib

import matplotlib.pyplot as plt
from matplotlib.pyplot import plot,savefig
from dataset import *

from sklearn.datasets import load_iris
import xgboost as xgb
import sklearn as sk
from xgboost import plot_importance
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.decomposition import PCA
import mylib as ml2
from mylib import LiftNet, create_LiftNet, create_Standard_LiftNet, Standard_LiftNet, create_Standard_LiftNet_CWRU

from plotly.graph_objs import Scatter,Layout
import plotly
import plotly.offline as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly.subplots import make_subplots
import pandas as pd
import plotly.figure_factory as ff

from xgboost import XGBClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix

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
print('set default parameters')
#------------------------adjustable parameters-----------------------#
model_path = ['/media/silverbullet/data_and_programing_file/newpaper/code/code1226/model/circle_1_to_6_in_our_data/', 
              '/media/silverbullet/data_and_programing_file/newpaper/code/code1226/model/circle_7_to_16_in_our_data_with_0.01_noise_data/',
              '/media/silverbullet/data_and_programing_file/newpaper/code/code1226/model/model_in_CWRU_data/']
model_head_name = ['Standard_LiftingNet_',
                   'Standard_LiftingNet_use_expansion_data__with_',
                   'Standard_expansion_CWRU_data_LiftingNet__with_',
                  'Standard_LiftingNet_use_expansion_data_',
                  'Standard_expansion_CWRU_data_LiftingNet_']

data_path = ['/media/silverbullet/data_and_programing_file/newpaper/dataset/newdataset/','/media/silverbullet/data_and_programing_file/newpaper/dataset/CWRU/CWRUdataset']

class_num = 5
circle_num = 1
cutsize = 256
steps = 5000
whether_use_CWRU_data = 0
LiftingNet_noise_scale = 2
whether_expansion_data = 0
expansion_data_number = 500
noise_scales = 0.01
if whether_use_CWRU_data==1:
    noise_scales = LiftingNet_noise_scale

#artificial_feature_method: 1 is 19 features, 2 is 9 features
artificial_feature_method = 2

pca_parameters = 27

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
#------------------------expansion data parameters-----------------------#

if whether_expansion_data == 1:
    dataset, label = expansion_and_add_noise(dataset,label,exnumber=expansion_data_number, noise_scales=noise_scales)
    print('data expansion')

#------------------------data processing---------------------------------#

input_shape = (dataset.shape[1],dataset.shape[2])
channel = dataset.shape[2]

x_number = dataset.shape[0]

x = dataset
print('x.shape: ',x.shape)
y = label
print('y.shape: ', y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_rate, random_state=615)
print('x_train.shape: ', x_train.shape)
print('x_test.shape: ',x_test.shape)
print('y_train.shape: ', y_train.shape)
print('y_test.shape: ', y_test.shape)

x_train_number = x_train.shape[0]
x_test_number = x_test.shape[0]
print('x_train_number: ',x_train_number)
print('x_test_number: ',x_test_number)

#------------------------------artificial features extraction-----------------------------#
if artificial_feature_method == 1:
    artificial_feature_data = feature_extractor(dataset)
else:
    artificial_feature_data = feature_extractor2(dataset)
#artificial_feature_data = artificial_feature_data.reshape(x_number,-1)[:,:select_feature_numbers]
artificial_feature_data = artificial_feature_data.reshape(x_number,-1)
print('artificial_feature_data.shape: ', artificial_feature_data.shape)

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

#----------------------------------LiftingNet features---------------------------------#
print('start extract feature')
feature_data = liftnet.feature_extractor(dataset)
print('extracted feature')
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    feature_data = feature_data.eval()
#feature_data = tf.Session().run(feature_data)
#sess = tf.InteractiveSession()
#feature_data = feature_data.eval()
#print(feature_data)
print('feature_data.shape: ', feature_data.shape)
feature_data = np.reshape(feature_data,(feature_data.shape[0],feature_data.shape[1]))
#feature_data = np.concatenate(dataset,axis=2)
print('feature_data.shape: ', feature_data.shape)
print('finished feature extract')
pca_feature_extractor = PCA(n_components=pca_parameters)
pca_feature_extractor.fit(feature_data)
pca_feature = pca_feature_extractor.transform(feature_data)
print('pca_feature.shape: ', pca_feature.shape)

#--------------------------create xgboost train dataset---------------------------#
x = pca_feature.reshape(x_number,-1)

#x = feature_data.reshape(x_number,-1)
x = np.concatenate((x,artificial_feature_data),axis=1)
print('x.shape: ',x.shape)

y = label.reshape(x_number,)
print('y.shape: ', y.shape)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_rate, random_state=99)

print('X_train.shape: ', X_train.shape)
print('X_test.shape: ',X_test.shape)
print('y_train.shape: ', y_train.shape)
print('y_test.shape: ', y_test.shape)

#---------------------------XGBoost------------------------------#
model = XGBClassifier(learning_rate=0.1,
                        n_estimators=1000,         # 树的个数--1000棵树建立xgboost
                        max_depth=16,               # 树的深度
                        min_child_weight = 1,      # 叶子节点最小权重
                        gamma=0.1,                  # 惩罚项中叶子结点个数前的参数
                        subsample=0.8,             # 随机选择80%样本建立决策树
                        colsample_btree=0.8,       # 随机选择80%特征建立决策树
                        objective='multi:softmax', # 指定损失函数
                        scale_pos_weight=1,        # 解决样本个数不平衡的问题
                        )

model.fit(X_train,y_train)
# 对测试集进行预测
#dtest = xgb.DMatrix(X_test)
#ans = model.predict(dtest)
ans = model.predict(X_test)
print('ans.shape: ', ans.shape)
ml2.evaluate_model2(ans, y_test)
# 计算准确率
cnt1 = 0
cnt2 = 0

for i in range(len(y_test)):
    if ans[i] == y_test[i]:
        cnt1 += 1
    else:
        cnt2 += 1
print("Accuracy: %.2f %% " % (100 * cnt1 / (cnt1 + cnt2)))

#-------------------------------plot confusion matrix------------------------#

label_name1 = ['NORMAL', 'IR', 'OR', 'BALL', 'JOINT']
label_name2 = ['NORMAL', 'BALL', 'IR', 'OR_3', 'OR_6', 'OR_12']

if whether_use_CWRU_data == 1:
    label_name = label_name2
else:
    label_name = label_name1

if whether_use_CWRU_data == 1:
    save_fig_name = 'CWRU_data_model_in_' + str(LiftingNet_noise_scale) + '_noise_to_test_' + str(noise_scales) + '_noise_scale_data_confusion_matrix.png'
else:
    save_fig_name = 'Our_data_model_in_' + str(circle_num) + '_circles_model_to_test_' + str(noise_scales) + '_noise_expansion_data_confusion_matrix.png'


plot_confusion_matrix(ans,y_test,label_name, save_name=save_fig_name)
"""
cm = confusion_matrix(y_test, ans)
cm = cm.astype('int32')

cmap = sea.cubehelix_palette(start = 1.5, rot = 3, gamma=0.8, as_cmap = True)
#f, (ax1,ax2) = plt.subplots(figsize = (10,10),nrows=2)
f= plt.figure(figsize = (16,9))
ax1 = f.add_subplot(1,1,1)

#sea.heatmap(cm,annot=True,ax= ax1, linewidths=0.05, cmap='rainbow',fmt='d')
sea.heatmap(cm,annot=True,ax= ax1, linewidths=0.05, cmap=cmap,fmt='d')
ax1.set_title("Confusion Matrix")
ax1.set_xlabel('Predict')
ax1.set_ylabel('True')
ax1.set_xticklabels(label_name)
ax1.set_yticklabels(label_name)

#plot_confusion_matrix(cm, label_name, "Confusion Matrix")
print(cm)
plt.savefig('Confusion Matrix.png', format='png')


#---------------------------K Fold-------------------------------------------#
scoring = ['precision_macro', 'recall_macro']
scores = cross_validate(model, X_train, y_train, scoring=scoring,cv=10, return_train_score=True)
sorted(scores.keys())
print('test result:')
print(scores)
scores_df = pd.DataFrame(scores)

#table_header = ['test_recall_macro', 'train_recall_macro','fit_time', 'train_precision_macro','test_precision_macro']

scores_df
"""


