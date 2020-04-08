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
from sklearn.model_selection import train_test_split,cross_val_score, cross_validate
import mylib as ml2

from sklearn.svm import SVC
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
 
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

#------------------------read data----------------------------#
cutsize = 256
circle_num =7
expansion_data_number = 500
noise_scales = 0.01
"""
dataset_file_name = 'dataset_' + str(circle_num) + '_' + str(cutsize) + '.npy'
label_file_name = 'label_' + str(circle_num) + '_' + str(cutsize) + '.npy'
"""
data_path = '/home/silver-bullet/newpaper/data/dataset/'
dataset, label = load_dataset(data_path, circle_num=circle_num, cutsize=cutsize) 

if circle_num>=7:
    dataset, label = expansion_and_add_noise(dataset,label,exnumber=expansion_data_number, noise_scales=noise_scales)


x_number = dataset.shape[0]

feature_data = feature_extractor(dataset)
print('feature_data.shape: ', feature_data.shape)

x = feature_data.reshape(x_number,-1)
print('x.shape: ',x.shape)
y = label.reshape(x_number,)
print('y.shape: ', y.shape)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=56)

print('X_train.shape: ', X_train.shape)
print('X_test.shape: ',X_test.shape)
print('y_train.shape: ', y_train.shape)
print('y_test.shape: ', y_test.shape)

#------------------------create and train svm----------------------------#
sc = StandardScaler()
sc.fit(X_train)

X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

X_combined_std = np.vstack((X_train_std,X_test_std))
y_combined_std = np.hstack((y_train,y_test))
print('X_train_std.shape: ',X_train_std.shape)

svm = SVC(kernel='linear',C=1.0,random_state= 0)
svm.fit(X_train_std,y_train)

predict_y = svm.predict(X_test_std)

print('predict_x.shape: ', predict_y.shape)

ml2.evaluate_model2(predict_y, y_test)

# 计算准确率
cnt1 = 0
cnt2 = 0
for i in range(len(y_test)):
    if predict_y[i] == y_test[i]:
        cnt1 += 1
    else:
        cnt2 += 1

print("Total Accuracy: %.2f %% " % (100 * cnt1 / (cnt1 + cnt2)))

#-----------k fold----------------#

#scoring = ['accuracy','recalling']
scoring = ['precision_macro', 'recall_macro']
#scores = cross_validate(svm, X_test_std, y_test, scoring=scoring,cv=10, return_train_score=True)

scores = cross_validate(svm, X_train_std, y_train, scoring=scoring,cv=10, return_train_score=True)
sorted(scores.keys())
print('test result:')
print(scores)


#plot
init_notebook_mode(connected=True)
scores_df = pd.DataFrame(scores)
scores_df


table_header = ['test_recall_macro', 'train_recall_macro','fit_time', 'train_precision_macro','test_precision_macro']



