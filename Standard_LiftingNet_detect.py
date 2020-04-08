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
from xgboost import plot_importance
from sklearn.model_selection import train_test_split
import mylib as ml2
from keras.callbacks import TensorBoard

from mylib import LiftNet, create_LiftNet, Standard_LiftNet, create_Standard_LiftNet

from plotly.graph_objs import Scatter,Layout
import plotly
import plotly.offline as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly.subplots import make_subplots
import pandas as pd
import plotly.figure_factory as ff

#setting offilne
plotly.offline.init_notebook_mode(connected=True)

test_rate = 0.2
epochs = 10000
lr=0.015
momentum=0.8
decay=0.01
validation_split=0.2
steps_per_epoch=1
validation_steps=1
circle_num =1
cutsize = 256
#head_of_name = './snapshot4/networks_liftnet_1111_'
#head_of_name = './snapshot/Standard_LiftingNet_'
#head_of_name = '/home/silver-bullet/newpaper/code1216/snapshot/snapshot_LiftingNet_with_expansion_data/Standard_LiftingNet_use_expansion_data_'
head_of_name = '/home/silver-bullet/newpaper/code1216/snapshot/snapshot/Standard_LiftingNet_'
#head_of_name = './snapshot_standard_liftingnet_4/networks_liftnet_1111_'
class_num = 5
bunch_steps = 100
snapshot = 500
channel = 3
circle_num =12
input_shape = (640*circle_num,3)

whether_expansion_data = 1
expansion_data_number = 500
noise_scales = 0.01

steps = 1000
save_model_name = head_of_name + str(circle_num) + '_data_the_' + str(steps)+ 'th_snapshot_with_' + str(steps_per_epoch) + '_steps_per_epoch.h5' 

#save_model_name = head_of_name + str(circle_num) + '_' + str(epochs)+ '_' + str(steps_per_epoch) + '.h5'
#save_model_name = head_of_name + str(circle_num) + '_data_the_' + str(epochs)+ 'th_snapshot_with_' + str(steps_per_epoch) + '_steps_per_epoch.h5'
#loss_map_name = head_of_name + str(circle_num) + '_' + str(epochs)+ '_' + str(steps_per_epoch) +'_loss.jpg'
#acc_map_name = head_of_name + str(circle_num) + '_' + str(epochs)+ '_' + str(steps_per_epoch) +'_acc.jpg'
data_path = '/home/silver-bullet/newpaper/data/dataset/'
dataset, label = load_dataset(data_path, circle_num=circle_num, cutsize=cutsize) 

if whether_expansion_data == 1:
    dataset, label = expansion_and_add_noise(dataset,label,exnumber=expansion_data_number, noise_scales=noise_scales)

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

bunch_size = math.floor(x_train_number / bunch_steps)
print('bunch_size: ', bunch_size)

x_train_truth_length = bunch_size * bunch_steps
x_train = x_train[:x_train_truth_length, :, :]
y_train = y_train[:x_train_truth_length,:]
print('x_train.shape: ', x_train.shape)


#liftnet = create_LiftNet(class_num = class_num, channel = channel, circle_num = circle_num, input_shape=input_shape,lr=lr, momentum=momentum, decay=decay)
liftnet = create_Standard_LiftNet(class_num = class_num, channel = channel, circle_num = circle_num, input_shape=input_shape,lr=lr, momentum=momentum, decay=decay)
#liftnet.summary()
"""
tbCallBack = TensorBoard(log_dir='./logs',  
                 histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
#                  batch_size=32,     # 用多大量的数据计算直方图
                 write_graph=True,  # 是否存储网络结构图
                 write_grads=True, # 是否可视化梯度直方图
                 write_images=True,# 是否可视化参数
                 embeddings_freq=0, 
                 embeddings_layer_names=None, 
                 embeddings_metadata=None)
"""

#history = liftnet.fit(x_train,y_train,validation_split=validation_split,epochs=epochs,batch_size=bunch_size,steps_per_epoch=steps_per_epoch,validation_steps=validation_steps,callbacks=[tbCallBack])
#history = liftnet.fit(x_train,y_train,validation_split=validation_split,epochs=epochs,batch_size=bunch_steps,steps_per_epoch=steps_per_epoch,validation_steps=validation_steps)
#loss, val_loss, acc, val_acc = liftnet.train(x_train,y_train,validation_split=validation_split,epochs=epochs,batch_size=bunch_steps,steps_per_epoch=steps_per_epoch,validation_steps=validation_steps,snapshot = snapshot,head_of_name = head_of_name,circle_num = circle_num)

liftnet.load_weights(save_model_name)
print('load model')

print('------------------------------------------------------------')
print('first predict: ')

#test
pre_y = liftnet.predict(x_test, steps=1)
#print(pre_y)
ml2.evaluate_model(pre_y,y_test)

pre_y2 = ml2.pre_to_index(pre_y)
# 计算准确率

cnt1 = 0
cnt2 = 0

for i in range(len(y_test)):
    if pre_y2[i] == y_test[i]:
        cnt1 += 1
    else:
        cnt2 += 1

print("Accuracy: %.2f %% " % (100 * cnt1 / (cnt1 + cnt2)))

del liftnet


print('save and delete model')
print('------------------------------------------------------------')
print('program end')