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
from mylib import LiftNet, create_LiftNet, create_Standard_LiftNet, Standard_LiftNet

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

#---------liftnet parameter------#
test_rate = 0.1
epochs = 10000
lr=0.015
momentum=0.8
decay=0.01
validation_split=0.2
steps_per_epoch=1
validation_steps=1

cutsize = 256
#head_of_name = './snapshot/networks_liftnet_1111_'
#head_of_name = './snapshot_standard_liftingnet_1/networks_liftnet_1111_'
#head_of_name = './snapshot/Standard_LiftingNet_'
head_of_name = '/home/silver-bullet/newpaper/code1216/snapshot/snapshot_LiftingNet_with_expansion_data/Standard_LiftingNet_use_expansion_data_'

class_num = 5
bunch_steps = 100
snapshot = 500
channel = 3
circle_num =10
input_shape = (640*circle_num,3)

steps = 1000
noise_scales = 0.01

#save_model_name = head_of_name + str(circle_num) + '_' + str(epochs)+ '_' + str(steps_per_epoch) + '.h5'
#save_model_name = head_of_name + str(circle_num) + '_data_the_' + str(steps)+ 'th_snapshot_with_' + str(steps_per_epoch) + '_steps_per_epoch.h5' 
save_model_name = head_of_name + '_with_' + str(noise_scales) + '_noise_'+str(circle_num) + '_data_the_' + str(steps)+ 'th_snapshot_with_' + str(steps_per_epoch) + '_steps_per_epoch.h5' 
# loss_map_name = head_of_name + str(circle_num) + '_' + str(epochs)+ '_' + str(steps_per_epoch) +'_loss.jpg'
#acc_map_name = head_of_name + str(circle_num) + '_' + str(epochs)+ '_' + str(steps_per_epoch) +'_acc.jpg'
data_path = '/home/silver-bullet/newpaper/data/dataset/'
dataset, label = load_dataset(data_path, circle_num=circle_num, cutsize=cutsize) 

"""
cutsize = 256
head_of_name = 'networks7_1030'
circle_num =2
"""


x_number = dataset.shape[0]
print(x_number)


dataset = dataset.astype(np.float32)
print(dataset.dtype)
print('dataset shape : ',dataset.shape)
print('label shape : ', label.shape)

select_feature_numbers = 30

artificial_feature_data = feature_extractor(dataset)
artificial_feature_data = artificial_feature_data.reshape(x_number,-1)[:,:select_feature_numbers]
print(artificial_feature_data.shape)
"""
feature_data = feature_extractor(dataset)
print('feature_data.shape: ', feature_data.shape)
"""

#create liftnet
#liftnet = create_LiftNet(class_num = class_num, channel = channel, circle_num = circle_num, input_shape=input_shape,lr=lr, momentum=momentum, decay=decay)
liftnet = create_Standard_LiftNet(class_num = class_num, channel = channel, circle_num = circle_num, input_shape=input_shape,lr=lr, momentum=momentum, decay=decay)

liftnet.load_weights(save_model_name)
print('load model')

feature_data = liftnet.feature_extractor(dataset)
#del liftnet
#tf.reset_default_graph() 
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    feature_data = feature_data.eval()
#feature_data = tf.Session().run(feature_data)
#sess = tf.InteractiveSession()
#feature_data = feature_data.eval()

print(feature_data)
print(feature_data.shape)
feature_data = np.reshape(feature_data,(feature_data.shape[0],feature_data.shape[1]))

#feature_data = np.concatenate(dataset,axis=2)
print(feature_data.shape)
print('finished feature extract')

pca_feature_extractor = PCA(n_components=30)
pca_feature_extractor.fit(feature_data)
pca_feature = pca_feature_extractor.transform(feature_data)
print(pca_feature_extractor.n_components_)
print(pca_feature_extractor.explained_variance_ratio_)

print(pca_feature.shape)



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

#---------xgboost parameter-------#

params = {
    'booster': 'gbtree',
    'objective': 'multi:softmax',
    'num_class': 5,
    'gamma': 0.1,
    'max_depth': 16,
    'lambda': 2,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'min_child_weight': 3,
    'silent': 1,
    'eta': 0.1,
    'seed': 1000,
    'nthread': 4,
}

plst = params.items()


dtrain = xgb.DMatrix(X_train, y_train)
num_rounds = 1000
model = xgb.train(plst, dtrain, num_rounds)


dtest = xgb.DMatrix(X_test)
ans = model.predict(dtest)

print('ans.shape: ', ans.shape)

ml2.evaluate_model2(ans, y_test)


cnt1 = 0
cnt2 = 0

for i in range(len(y_test)):
    if ans[i] == y_test[i]:
        cnt1 += 1
    else:
        cnt2 += 1

print("Accuracy: %.2f %% " % (100 * cnt1 / (cnt1 + cnt2)))

plot_importance(model)
plt.show()

