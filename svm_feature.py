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

from sklearn.svm import SVC, LinearSVC
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
from xgboost import XGBClassifier
#from sklearn.cross_validation import cross_val_score, ShuffleSplit

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
circle_num =2

input_shape = (640*circle_num,3)

data_path = '/home/silver-bullet/newpaper/data/dataset/'

dataset, label = load_dataset(data_path, circle_num, cutsize)


x_number = dataset.shape[0]


feature_data = feature_extractor(dataset)
print('feature_data.shape: ', feature_data.shape)

x = feature_data.reshape(x_number,-1)
all_feature_data = x
print('x.shape: ',x.shape)
y = label.reshape(x_number,)
print('y.shape: ', y.shape)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=56)

print('X_train.shape: ', X_train.shape)
print('X_test.shape: ',X_test.shape)
print('y_train.shape: ', y_train.shape)
print('y_test.shape: ', y_test.shape)

#----------------------------------SVM----------------------------#

sc = StandardScaler()
sc.fit(X_train)

X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

X_combined_std = np.vstack((X_train_std,X_test_std))
y_combined_std = np.hstack((y_train,y_test))
print('X_train_std.shape: ',X_train_std.shape)

#svm = SVC(kernel='linear',C=1.0,random_state= 0)
svm = LinearSVC(max_iter=10000)
"""
scores = []

for i in range(X_train.shape[1]):
    score = cross_val_score(svm, X_train_std[:, i:i+1],y_train,scoring=)
"""
svm.fit(X_train_std,y_train)

predict_y = svm.predict(X_test_std)
print('SVM result: ')
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

#feature_ranker = SelectKBest(score_func=mutual_info_classif, k=20).fit(x,y)

print(svm.coef_.shape)
#print(svm.coef_)
print(np.argsort(-svm.coef_,axis=1))

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
                        max_depth=6,               # 树的深度
                        min_child_weight = 1,      # 叶子节点最小权重
                        gamma=0.1,                  # 惩罚项中叶子结点个数前的参数
                        subsample=0.8,             # 随机选择80%样本建立决策树
                        colsample_btree=0.8,       # 随机选择80%特征建立决策树
                        objective='multi:softmax', # 指定损失函数
                        scale_pos_weight=1,        # 解决样本个数不平衡的问题
                        random_state=27)            # 随机数
model.fit(X_train,y_train)
# 对测试集进行预测
#dtest = xgb.DMatrix(X_test)
#ans = model.predict(dtest)
ans = model.predict(X_test)
print('XGBoost result: ')
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

"""
# 显示重要特征
plot_importance(model)
plt.show()
"""
#--------------------------select feature-------------------------------#
print('start select common feature')
select_feature_num = 20
svm_feature_index = np.argsort(-svm.coef_,axis=1)[:,:select_feature_num]
xgb_feature_index = np.argsort(-model.feature_importances_)[:select_feature_num].T
print(xgb_feature_index.shape)
select_method = 1
common_feature = select_svm_xgboost_common_feature(svm_feature_index,xgb_feature_index,select_feature_num,select_method)
print(common_feature)
class_name = ['normal', 'inner ring', 'outer ring', 'ball', 'joint']
if select_method==1:
    for i in range(len(common_feature)):
        print(class_name[i]+' result: ')
        select_feature = common_feature[i]
        select_feature_data = all_feature_data[:,select_feature]
        print(select_feature_data.shape)
        x = select_feature_data
        print('x.shape: ',x.shape)
        y = label.reshape(x_number,)
        print('y.shape: ', y.shape)
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
        print('X_train.shape: ', X_train.shape)
        print('X_test.shape: ',X_test.shape)
        print('y_train.shape: ', y_train.shape)
        print('y_test.shape: ', y_test.shape)
        
        sc1 = StandardScaler()
        sc1.fit(X_train)

        X_train_std = sc1.transform(X_train)
        X_test_std = sc1.transform(X_test)

        X_combined_std = np.vstack((X_train_std,X_test_std))
        y_combined_std = np.hstack((y_train,y_test))
        print('X_train_std.shape: ',X_train_std.shape)

        #svm = SVC(kernel='linear',C=1.0,random_state= 0)
        svm = LinearSVC(max_iter=10000)
        svm.fit(X_train_std,y_train)

        predict_y = svm.predict(X_test_std)
        print('SVM result: ')
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
        scores_svm = cross_validate(svm, X_train_std, y_train, scoring=scoring,cv=10, return_train_score=True)
        sorted(scores_svm.keys())
        print('test result:')
        print(scores_svm)
        #plot
        init_notebook_mode(connected=True)
        scores_df_svm = pd.DataFrame(scores_svm)
        scores_df_svm
        
        xgboost_model = XGBClassifier(learning_rate=0.1,
                        n_estimators=1000,         # 树的个数--1000棵树建立xgboost
                        max_depth=6,               # 树的深度
                        min_child_weight = 1,      # 叶子节点最小权重
                        gamma=0.1,                  # 惩罚项中叶子结点个数前的参数
                        subsample=0.8,             # 随机选择80%样本建立决策树
                        colsample_btree=0.8,       # 随机选择80%特征建立决策树
                        objective='multi:softmax', # 指定损失函数
                        scale_pos_weight=1,        # 解决样本个数不平衡的问题
                        random_state=27)            # 随机数
        xgboost_model.fit(X_train,y_train)
        # 对测试集进行预测
        #dtest = xgb.DMatrix(X_test)
        #ans = model.predict(dtest)
        ans = xgboost_model.predict(X_test)
        print('XGBoost result: ')
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
        
        scores_xgb = cross_validate(xgboost_model, X_train, y_train, scoring=scoring,cv=10, return_train_score=True)
        sorted(scores_xgb.keys())
        print('test result:')
        print(scores_xgb)
        scores_df_xgb = pd.DataFrame(scores_xgb)
        scores_df_xgb
        
else:
    select_feature = common_feature
    select_feature_data = all_feature_data[:,select_feature]
    print(select_feature_data.shape)
    x = select_feature_data
    print('x.shape: ',x.shape)
    y = label.reshape(x_number,)
    print('y.shape: ', y.shape)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    print('X_train.shape: ', X_train.shape)
    print('X_test.shape: ',X_test.shape)
    print('y_train.shape: ', y_train.shape)
    print('y_test.shape: ', y_test.shape)
    
    sc2 = StandardScaler()
    sc2.fit(X_train)

    X_train_std = sc2.transform(X_train)
    X_test_std = sc2.transform(X_test)

    X_combined_std = np.vstack((X_train_std,X_test_std))
    y_combined_std = np.hstack((y_train,y_test))
    print('X_train_std.shape: ',X_train_std.shape)

    #svm = SVC(kernel='linear',C=1.0,random_state= 0)
    svm = LinearSVC(max_iter=10000)
    svm.fit(X_train_std,y_train)

    predict_y = svm.predict(X_test_std)
    print('SVM result: ')
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
    scores_svm = cross_validate(svm, X_train_std, y_train, scoring=scoring,cv=10, return_train_score=True)
    sorted(scores_svm.keys())
    print('test result:')
    print(scores_svm)
    #plot
    init_notebook_mode(connected=True)
    scores_df_svm = pd.DataFrame(scores_svm)
    scores_df_svm
    
    xgboost_model = XGBClassifier(learning_rate=0.1,
                    n_estimators=1000,         # 树的个数--1000棵树建立xgboost
                    max_depth=6,               # 树的深度
                    min_child_weight = 1,      # 叶子节点最小权重
                    gamma=0.1,                  # 惩罚项中叶子结点个数前的参数
                    subsample=0.8,             # 随机选择80%样本建立决策树
                    colsample_btree=0.8,       # 随机选择80%特征建立决策树
                    objective='multi:softmax', # 指定损失函数
                    scale_pos_weight=1,        # 解决样本个数不平衡的问题
                    random_state=27)            # 随机数
    xgboost_model.fit(X_train,y_train)
    # 对测试集进行预测
    #dtest = xgb.DMatrix(X_test)
    #ans = model.predict(dtest)
    ans = xgboost_model.predict(X_test)
    print('XGBoost result: ')
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
    
    scores_xgb = cross_validate(xgboost_model, X_train, y_train, scoring=scoring,cv=10, return_train_score=True)
    sorted(scores_xgb.keys())
    print('test result:')
    print(scores_xgb)
    scores_df_xgb = pd.DataFrame(scores_xgb)
    scores_df_xgb

                        


"""
class_number = svm_feature_index.shape[0]
print(class_number)

svm_common_feature = svm_feature_index[0,:]
for i in range(class_number-1):
    next_feature_index = svm_feature_index[i+1,:]
    common_feature = list(set(svm_common_feature).intersection(set(next_feature_index)))
    svm_common_feature = common_feature

print(svm_common_feature)

svm_common_feature = []
#common_feature = []
for i in range(class_number-1):
    now_feature_index = svm_feature_index[i,:]
    next_feature_index = svm_feature_index[i+1,:]
    common_feature = list(set(now_feature_index).intersection(set(next_feature_index)))
    print('the '+str(i)+'th:')
    if len(svm_common_feature)==0:
        svm_common_feature = common_feature
        print(svm_common_feature)
        continue
    svm_common_feature = list(set(svm_common_feature).union(set(common_feature)))
    print(svm_common_feature)
print('the final svm feature')
print(svm_common_feature)
print(len(svm_common_feature))


select_feature =  list(set(svm_common_feature).intersection(set(xgb_feature_index)))
print(select_feature)
print(len(select_feature))

select_feature_data = x[:,select_feature]
print(select_feature_data.shape)

common_feature = []
for i in range(class_number):
    svm_i_class_feature_index = svm_feature_index[i,:]
    #next_feature_index = svm_feature_index[i+1,:]
    common_feature.append(list(set(svm_i_class_feature_index).intersection(set(xgb_feature_index))))


print('the final common feature')
print(common_feature)

"""



"""
#-----------------------new train--------------------------------------#
x = select_feature_data
print('x.shape: ',x.shape)
y = label.reshape(x_number,)
print('y.shape: ', y.shape)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=56)

print('X_train.shape: ', X_train.shape)
print('X_test.shape: ',X_test.shape)
print('y_train.shape: ', y_train.shape)
print('y_test.shape: ', y_test.shape)




#----------------------------------SVM----------------------------#

sc = StandardScaler()
sc.fit(X_train)

X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

X_combined_std = np.vstack((X_train_std,X_test_std))
y_combined_std = np.hstack((y_train,y_test))
print('X_train_std.shape: ',X_train_std.shape)

#svm = SVC(kernel='linear',C=1.0,random_state= 0)
svm = LinearSVC()

scores = []

for i in range(X_train.shape[1]):
    score = cross_val_score(svm, X_train_std[:, i:i+1],y_train,scoring=)

svm.fit(X_train_std,y_train)

predict_y = svm.predict(X_test_std)
print('SVM result: ')
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

#feature_ranker = SelectKBest(score_func=mutual_info_classif, k=20).fit(x,y)

print(svm.coef_.shape)
#print(svm.coef_)
print(np.argsort(-svm.coef_,axis=1))

#---------------------XGBoost----------------------------#

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

model = XGBClassifier(learning_rate=0.1,
                        n_estimators=1000,         # 树的个数--1000棵树建立xgboost
                        max_depth=6,               # 树的深度
                        min_child_weight = 1,      # 叶子节点最小权重
                        gamma=0.1,                  # 惩罚项中叶子结点个数前的参数
                        subsample=0.8,             # 随机选择80%样本建立决策树
                        colsample_btree=0.8,       # 随机选择80%特征建立决策树
                        objective='multi:softmax', # 指定损失函数
                        scale_pos_weight=1,        # 解决样本个数不平衡的问题
                        random_state=27)            # 随机数
model.fit(X_train,y_train)
# 对测试集进行预测
#dtest = xgb.DMatrix(X_test)
#ans = model.predict(dtest)
ans = model.predict(X_test)
print('XGBoost result: ')
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