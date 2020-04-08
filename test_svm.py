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

cutsize = 256
head_of_name = 'networks7_1030'
circle_num =8

input_shape = (640*circle_num,3)
"""
dataset_file_name = 'dataset_' + str(circle_num) + '_' + str(cutsize) + '.npy'
label_file_name = 'label_' + str(circle_num) + '_' + str(cutsize) + '.npy'
"""
data_path = '/home/silver-bullet/newpaper/data/dataset/'
dataset_file_name = data_path + 'dataset_' + str(circle_num) + '_' + str(cutsize) + '.npy'
label_file_name = data_path + 'label_' + str(circle_num) + '_' + str(cutsize) + '.npy'

dataset = np.load(dataset_file_name)
label = np.load(label_file_name)
print('dataset shape : ',dataset.shape)
print('label shape : ', label.shape)
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

table_header = ['test_recall_macro', 'train_recall_macro','fit_time', 'train_precision_macro','test_precision_macro']

scores_df



"""
def plot_decision_regions(X, y, classifier,test_idx=None, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'cyan','gray')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    # plot all samples
    #X_test, y_test = X[test_idx, :], y[test_idx]
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)
    # highlight test samples
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:,0], X_test[:,1], c='',
                alpha=1.0, linewidth=1, marker='o',
                s=55, label='test set')
 
plot_decision_regions(X_combined_std,y_combined_std, classifier=svm,test_idx=range(105,150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()

"""
