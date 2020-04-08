#import tensorflow as tf
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
"""
from plotly.graph_objs import Scatter,Layout
import plotly
import plotly.offline as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly.subplots import make_subplots
import pandas as pd
import plotly.figure_factory as ff
"""
#setting offilne

#head_of_name = './snapshot4/networks_liftnet_1111_'
#head_of_name = './snapshot/Standard_LiftingNet_'
#head_of_name = './snapshot_standard_liftingnet_4/networks_liftnet_1111_'
#save_model_name = head_of_name + str(circle_num) + '_' + str(epochs)+ '_' + str(steps_per_epoch) + '.h5'
#save_model_name = head_of_name + str(circle_num) + '_data_the_' + str(epochs)+ 'th_snapshot_with_' + str(steps_per_epoch) + '_steps_per_epoch.h5'
#loss_map_name = head_of_name + str(circle_num) + '_' + str(epochs)+ '_' + str(steps_per_epoch) +'_loss.jpg'
#acc_map_name = head_of_name + str(circle_num) + '_' + str(epochs)+ '_' + str(steps_per_epoch) +'_acc.jpg'


#plotly.offline.init_notebook_mode(connected=True)


circle_num =1
cutsize = 256
class_num = 5
input_shape = (640*circle_num,3)

data_path = '/home/silver-bullet/newpaper/data/dataset/'
dataset_file_name = data_path + 'dataset_' + str(circle_num) + '_' + str(cutsize) + '.npy'
label_file_name = data_path + 'label_' + str(circle_num) + '_' + str(cutsize) + '.npy'

dataset = np.load(dataset_file_name)
label = np.load(label_file_name)
print('dataset shape : ',dataset.shape)
print('label shape : ', label.shape)

noise_scale = 0.01
noise = np.random.randn(dataset.shape[0],dataset.shape[1],dataset.shape[2])
data_max_abs_x = np.max(np.abs(dataset),axis=1)[:,np.newaxis,:]

print('data_max_abs_x.shape: ',data_max_abs_x.shape)
noise = noise_scale*data_max_abs_x*noise
print('noise.shape: ', noise.shape)


#------------data expansion------------------#
exnumber = 50
weight = np.random.rand(exnumber,2)
weight = weight/np.sum(weight,axis=1)[:,np.newaxis]
index = np.random.randint(0,dataset.shape[0],[exnumber,2])
#--------------------------------------------------
whether_can_merge = np.zeros(exnumber)
for i in range(exnumber):
    if label[index[i,0],0]==label[index[i,1],0]:
        whether_can_merge[i] = 1
#print(whether_can_merge)
#print(weight)
#print(index)
#-------------------------------------------------
label_num = []
k=0
for i in range(label.shape[0]):
    if label[i,0]==k:
        label_num.append(i)
        k=k+1
print(k)
label_num.append(label.shape[0])
print(len(label_num))

e,l = data_expension(dataset,label)
print(l.T)
"""
dataset_0 = dataset[:label_num[0],:,:]
dataset_1 = dataset[label_num[0]:label_num[1],:,:]
dataset_2 = dataset[label_num[1]:label_num[2],:,:]
dataset_3 = dataset[label_num[2]:label_num[3],:,:]
dataset_4 = dataset[label_num[3]:,:,:]

new_data = [dataset_0,dataset_1,dataset_2,dataset_3,dataset_4]

print(dataset_0.shape)
print(dataset_1.shape)
print(dataset_2.shape)
print(dataset_3.shape)
output_data = []
output_label = []
for i in range(len(new_data)):
    
    if i == 0:
        expansion_data = dataset[:label_num[i],:,:]
    elif i ==len(label_num):
        expansion_data = dataset[label_num[i]:,:,:]
    else:
        expansion_data = dataset[label_num[i]:label_num[i+1],:,:]
    
    expansion_data = new_data[i]
    print(expansion_data.shape)
    sample_num = expansion_data.shape[0]
    weight = np.random.rand(exnumber,2)
    weight = weight/np.sum(weight,axis=1)[:,np.newaxis]
    index = np.random.randint(0,sample_num,[exnumber,2])
    merge_data = np.zeros([exnumber,expansion_data.shape[1],expansion_data.shape[2]])
    merge_label = np.ones([exnumber,1])*i
    for j in range(exnumber):
        merge_data[j,:,:] = expansion_data[index[j,0]]*weight[j,0] + expansion_data[index[j,1]]*weight[j,1]
    print(merge_data.shape)
    print(merge_label.shape)
    if output_data==[]:
        output_data = merge_data
        output_label = merge_label
        continue
    output_data = np.concatenate((output_data,merge_data),axis=0)
    output_label = np.concatenate((output_label,merge_label),axis=0)
print(output_data.shape)
print(output_label.shape)
print(output_label.T)





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
"""