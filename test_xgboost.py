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
from xgboost import XGBClassifier

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score, cross_validate
import pandas as pd

"""
test_rate = 0.2
epochs = 20000
lr=0.01
momentum=0.8
decay=0.01
validation_split=0.2
steps_per_epoch=1
validation_steps=1
circle_num =1
cutsize = 256

head_of_name = 'networks7_1030'
input_shape = (640*circle_num,3)

save_model_name = head_of_name + str(circle_num) + '_' + str(epochs)+ '_' + str(steps_per_epoch) + '.h5'
loss_map_name = head_of_name + str(circle_num) + '_' + str(epochs)+ '_' + str(steps_per_epoch) +'.jpg'

dataset_file_name = 'dataset_' + str(circle_num) + '_' + str(cutsize) + '.npy'
label_file_name = 'label_' + str(circle_num) + '_' + str(cutsize) + '.npy'
dataset = np.load(dataset_file_name)
label = np.load(label_file_name)
print('dataset shape : ',dataset.shape)
print('label shape : ', label.shape)
"""
#------------------------read data----------------------------#
cutsize = 256
circle_num =8
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

feature_data = feature_extractor2(dataset)
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

#---------------------XGBoost----------------------------#
"""
params = {
    'booster': 'gbtree',
    'objective': 'multi:softmax',
    'num_class': 5,
    'gamma': 0.1,
    'max_depth': 8,
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
"""
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

#---------------------------K Fold-------------------------------------------#
scoring = ['precision_macro', 'recall_macro']
scores = cross_validate(model, X_train, y_train, scoring=scoring,cv=10, return_train_score=True)
sorted(scores.keys())
print('test result:')
print(scores)
scores_df = pd.DataFrame(scores)

table_header = ['test_recall_macro', 'train_recall_macro','fit_time', 'train_precision_macro','test_precision_macro']

scores_df


"""
kfold = KFold(n_splits=10)
results = cross_val_score(model, x, y, cv=kfold)
print('result:')
print(results)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

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

print(model.feature_importances_.shape)
print(model.feature_importances_)
print(np.argsort(-model.feature_importances_))




# 显示重要特征
plot_importance(model)
plt.show()
"""




