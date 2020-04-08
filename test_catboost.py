import pandas as pd
import numpy as np
import lightgbm as lgb
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.cluster import KMeans
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from dataset import *
import mylib as ml2

cutsize = 256
circle_num =4
save_log_path = './catboost_log/'

input_shape = (640*circle_num,3)

dataset_file_name = 'dataset_' + str(circle_num) + '_' + str(cutsize) + '.npy'
label_file_name = 'label_' + str(circle_num) + '_' + str(cutsize) + '.npy'

dataset = np.load(dataset_file_name)
label = np.load(label_file_name)
print('dataset shape : ',dataset.shape)
print('label shape : ', label.shape)
x_number = dataset.shape[0]

feature_data = feature_extractor(dataset)
print('feature.shape: ', feature_data.shape)

x = feature_data.reshape(x_number,-1)
print('x.shape: ',x.shape)
y = label.reshape(x_number,)
print('y.shape: ', y.shape)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=56)

"""

X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)
y_train = pd.DataFrame(y_train)
y_test = pd.DataFrame(y_test)
"""
all_attrs1 = ['ch1_XP', 'ch1_XPP', 'ch1_sum_square', 'ch1_RMS', 'ch1_data_mean', 'ch1_data_std', 'ch1_Variance', 'ch1_Skewness', 'ch1_Kurtosis', 'ch1_SF', 'ch1_CF', 'ch1_IF', 'ch1_MF', 'ch1_mobility','ch1_complexity', 'ch1_FC', 'ch1_MSF', 'ch1_RMSF', 'ch1_RVF']
all_attrs2 = ['ch2_XP', 'ch2_XPP', 'ch2_sum_square', 'ch2_RMS', 'ch2_data_mean', 'ch2_data_std', 'ch2_Variance', 'ch2_Skewness', 'ch2_Kurtosis', 'ch2_SF', 'ch2_CF', 'ch2_IF', 'ch2_MF', 'ch2_mobility','ch2_complexity', 'ch2_FC', 'ch2_MSF', 'ch2_RMSF', 'ch2_RVF']
all_attrs3 = ['ch3_XP', 'ch3_XPP', 'ch3_sum_square', 'ch3_RMS', 'ch3_data_mean', 'ch3_data_std', 'ch3_Variance', 'ch3_Skewness', 'ch3_Kurtosis', 'ch3_SF', 'ch3_CF', 'ch3_IF', 'ch3_MF', 'ch3_mobility','ch3_complexity', 'ch3_FC', 'ch3_MSF', 'ch3_RMSF', 'ch3_RVF']
all_attrs = []
for i in range(len(all_attrs1)):
    all_attrs.append(all_attrs1[i])
    all_attrs.append(all_attrs2[i])
    all_attrs.append(all_attrs3[i])

print(all_attrs)
"""
X_train.columns = all_attrs
X_test.columns = all_attrs
"""
print('X_train.shape: ', X_train.shape)
print('X_test.shape: ',X_test.shape)
print('y_train.shape: ', y_train.shape)
print('y_test.shape: ', y_test.shape)


"""
print('X_train: ', X_train)
print('X_test: ',X_test)
print('y_train: ', y_train)
print('y_test: ', y_test)

'''
数据预处理
'''
all_attrs = ['Parameter1', 'Parameter2', 'Parameter3', 'Parameter4', 'Parameter5', 'Parameter6', 'Parameter7',
             'Parameter8', 'Parameter9', 'Parameter10', 'Attribute1', 'Attribute2', 'Attribute3', 'Attribute4',
             'Attribute5', 'Attribute6', 'Attribute7', 'Attribute8', 'Attribute9', 'Attribute10', 'Quality_label']
unused_attrs = ['Attribute1', 'Attribute2', 'Attribute3', 'Attribute4', 'Attribute5', 'Attribute6', 'Attribute7',
                'Attribute8', 'Attribute9', 'Attribute10']
cat_attrs = ['Parameter5', 'Parameter6', 'Parameter7', 'Parameter8', 'Parameter9', 'Parameter10']
dataset = dataset.drop(unused_attrs, axis=1)
quality_mapping = {
           'Excellent': 1,
           'Good': 2,
           'Pass': 3,
           'Fail': 4}
dataset['Quality_label'] = dataset['Quality_label'].map(quality_mapping)
 
 
X = dataset.drop('Quality_label', axis=1)
print('X  : ', X.shape)
y = dataset['Quality_label']

"""

'''
模型构建
'''
catboost_model = CatBoostClassifier(
    iterations=200,
    od_type='Iter',
    od_wait=50,
    max_depth=8,
    learning_rate=0.5,
    l2_leaf_reg=3,
    random_seed=2019,
    metric_period=50,
    fold_len_multiplier=1.1,
    used_ram_limit = '2gb',
    gpu_ram_part = '2gb',
    thread_count = 4,
    loss_function='MultiClass',
    logging_level='Verbose',
    classes_count = 5,
    verbose=True,
    train_dir=save_log_path,
    allow_writing_files=True,
    save_snapshot=True,
    plot=True
    
    )
 
catboost_model.fit(X_train, y_train)

y_pred = catboost_model.predict(X_test)
print(y_pred.shape)
ml2.evaluate_model2(y_pred,y_test)
# 模型评价
f1 = f1_score(y_test, y_pred, average='macro')
acc = accuracy_score(y_test, y_pred)
print('f1 : ', f1)
print('accuracy : ', acc)

catboost_model.get_feature_importance
print('finished')
