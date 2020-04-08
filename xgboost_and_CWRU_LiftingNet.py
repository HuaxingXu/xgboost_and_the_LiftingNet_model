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
from sklearn.model_selection import cross_val_score, cross_validate
import pandas as pd


test_rate = 0.1
lr=0.015
momentum=0.8
decay=0.01
validation_split=0.2
validation_steps=1
epochs = 10000

LiftingNet_noise_scale = 0.01
train_steps = 800
steps_per_epoch=1
cutsize = 1024

class_num = 6
bunch_steps = 100
snapshot = 500

circle_num =1


whether_expansion_data = 1
expansion_data_number = 500
noise_scales = 0.01


model_head_name = '/home/silver-bullet/newpaper/snapshot/snapshot_CWRU_LiftingNet/Standard_expansion_CWRU_data_LiftingNet_'
model_name = model_head_name + str(LiftingNet_noise_scale) + '_data_the_' + str(train_steps) + 'th_snapshot_with_' + str(steps_per_epoch) + '_steps_per_epoch.h5'

data_path = '/home/silver-bullet/newpaper/data/CWRUdataset'
dataset, label = load_CWRU_data(data_path)

if whether_expansion_data == 1:
    dataset, label = expansion_and_add_noise(dataset,label,exnumber=expansion_data_number, noise_scales=noise_scales)

x_number = dataset.shape[0]
print(x_number)

dataset = dataset.astype(np.float32)
print(dataset.dtype)
print('dataset shape : ',dataset.shape)
print('label shape : ', label.shape)

input_shape = (dataset.shape[1],dataset.shape[2])
channel = dataset.shape[2]

select_feature_numbers = 30

artificial_feature_data = feature_extractor2(dataset)
#artificial_feature_data = artificial_feature_data.reshape(x_number,-1)[:,:select_feature_numbers]
artificial_feature_data = artificial_feature_data.reshape(x_number,-1)
print(artificial_feature_data.shape)

liftnet = create_Standard_LiftNet_CWRU(class_num = class_num, channel = channel, cut_size = cutsize, input_shape=input_shape,lr=lr, momentum=momentum, decay=decay)

liftnet.load_weights(model_name)
print('load model')

feature_data = liftnet.feature_extractor(dataset)
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
print(feature_data.shape)
print('finished feature extract')

pca_feature_extractor = PCA(n_components=25)
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



"""
#---------------------------K Fold-------------------------------------------#
scoring = ['precision_macro', 'recall_macro']
scores = cross_validate(model, X_train, y_train, scoring=scoring,cv=10, return_train_score=True)
sorted(scores.keys())
print('test result:')
print(scores)
scores_df = pd.DataFrame(scores)

table_header = ['test_recall_macro', 'train_recall_macro','fit_time', 'train_precision_macro','test_precision_macro']

scores_df




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
"""