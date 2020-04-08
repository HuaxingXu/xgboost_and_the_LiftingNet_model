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

import plotly
import plotly.offline as pltoff
from plotly.graph_objs import Scatter, Layout
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd

cutsize = 256
head_of_name = 'networks7_1030'
circle_num =4

input_shape = (640*circle_num,3)

dataset_file_name = 'dataset_' + str(circle_num) + '_' + str(cutsize) + '.npy'
label_file_name = 'label_' + str(circle_num) + '_' + str(cutsize) + '.npy'

dataset = np.load(dataset_file_name)
label = np.load(label_file_name)
print('dataset shape : ',dataset.shape)
print('label shape : ', label.shape)
x_number = dataset.shape[0]


feature_data = feature_extractor(dataset)
print('feature_data.shape: ', feature_data.shape)

x = feature_data.reshape(x_number,-1)
print('x.shape: ',x.shape)
y = label.reshape(x_number,-1)
print('y.shape: ', y.shape)

all_attrs1 = ['ch1_XP', 'ch1_XPP', 'ch1_sum_square', 'ch1_RMS', 'ch1_data_mean', 'ch1_data_std', 'ch1_Variance', 'ch1_Skewness', 'ch1_Kurtosis', 'ch1_SF', 'ch1_CF', 'ch1_IF', 'ch1_MF', 'ch1_mobility','ch1_complexity', 'ch1_FC', 'ch1_MSF', 'ch1_RMSF', 'ch1_RVF']
all_attrs2 = ['ch2_XP', 'ch2_XPP', 'ch2_sum_square', 'ch2_RMS', 'ch2_data_mean', 'ch2_data_std', 'ch2_Variance', 'ch2_Skewness', 'ch2_Kurtosis', 'ch2_SF', 'ch2_CF', 'ch2_IF', 'ch2_MF', 'ch2_mobility','ch2_complexity', 'ch2_FC', 'ch2_MSF', 'ch2_RMSF', 'ch2_RVF']
all_attrs3 = ['ch3_XP', 'ch3_XPP', 'ch3_sum_square', 'ch3_RMS', 'ch3_data_mean', 'ch3_data_std', 'ch3_Variance', 'ch3_Skewness', 'ch3_Kurtosis', 'ch3_SF', 'ch3_CF', 'ch3_IF', 'ch3_MF', 'ch3_mobility','ch3_complexity', 'ch3_FC', 'ch3_MSF', 'ch3_RMSF', 'ch3_RVF']
attrs_group = [all_attrs1,all_attrs2,all_attrs3]
total_all_attrs = []
for i in range(len(all_attrs1)):
    total_all_attrs.append(all_attrs1[i])
    total_all_attrs.append(all_attrs2[i])
    total_all_attrs.append(all_attrs3[i])
print(total_all_attrs)

for c in [1,2,3]:
    channel = c
    feature_number = 19
    save_fig_name = 'channel_' + str(channel) + '_feature_use_' + str(circle_num) + '_circles_data.html'
    feature_index = []
    flag = 0 + channel - 1
    for i in range(feature_number):
        feature_index.append(flag)
        flag = flag + 3

    newx = x[:,feature_index]
    all_attrs = attrs_group[c-1]
    feature_data_and_label = np.concatenate((newx,y),axis=1)
    print('feature_data_and_label.shape: ',feature_data_and_label.shape)
    all_attrs.append('label')
    print(all_attrs)

    data_feature = {}
    for i in range(len(all_attrs)):
        data_feature[all_attrs[i]] = feature_data_and_label[:, i]
    #print(data_feature)
    data_feature = pd.DataFrame(data_feature)

    figure_label = dict(zip(all_attrs,all_attrs))

    fig = px.parallel_coordinates(data_feature, color="label", labels=figure_label,
                        color_continuous_scale=px.colors.diverging.Tealrose, color_continuous_midpoint=2)

    pltoff.plot(fig, filename=save_fig_name)

    fig.show()
"""
feature43 = x[:,42]
feature43 = feature43.reshape(x_number,-1)
print('feature43.shape: ', feature43.shape)
ch1_complexity = feature_data[:,14,0]
ch1_complexity = ch1_complexity.reshape(x_number,-1)
print('ch1_complexity.shape: ',ch1_complexity.shape)
ch3_mean = feature_data[:,4,2]
ch3_mean = ch3_mean.reshape(x_number,-1)
print('ch3_mean.shape: ',ch3_mean.shape)

d_feature43_ch1 = np.sum(feature43 - ch1_complexity)
d_feature43_ch3 = np.sum(feature43 - ch3_mean)

print('feature43 - ch1_complexity = ', d_feature43_ch1)
print('feature43 - ch3_mean = ', d_feature43_ch3)
"""





