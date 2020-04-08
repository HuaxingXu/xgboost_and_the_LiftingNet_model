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

from mylib import LiftNet, create_LiftNet









"""
test_rate = 0.9
epochs = 10000
lr=0.01
momentum=0.8
decay=0.01
validation_split=0.2
steps_per_epoch=1
validation_steps=1
circle_num =1
cutsize = 256
head_of_name = './snapshot/networks_liftnet_1111_'

bunch_steps = 100
snapshot = 7000

circle_num =1
input_shape = (640*circle_num,3)

save_model_name = head_of_name + str(circle_num) + '_data_the_' + str(snapshot)+ 'th_snapshot_with_' + str(steps_per_epoch) + '_steps_per_epoch.h5'


dataset_file_name = 'dataset_' + str(circle_num) + '_' + str(cutsize) + '.npy'
label_file_name = 'label_' + str(circle_num) + '_' + str(cutsize) + '.npy'

dataset = np.load(dataset_file_name)
label = np.load(label_file_name)
print('dataset shape : ',dataset.shape)
print('label shape : ', label.shape)
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


liftnet = create_LiftNet()
liftnet.load_weights(save_model_name)
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









class LiftNet(tf.keras.Model):
    def __init__(self, class_num = 5, input_shape=(640,3), channel = 3, netname='LiftNet'):
        super(LiftNet,self).__init__(name=netname)
        self.input_layer = tf.keras.layers.InputLayer(input_shape=input_shape,name = 'input')
        self.convinputs = tf.keras.layers.Conv1D(filters=channel,kernel_size=1,strides=1,padding='valid',name='convinputs')
        #split layer 320
        self.convsplit1_d = tf.keras.layers.Conv1D(filters=channel,kernel_size=2,strides=2,padding='valid',name='split1_d')
        self.convsplit1_s = tf.keras.layers.Conv1D(filters=channel,kernel_size=2,strides=2,padding='valid',name='split1_s')
        #predict layer
        self.convpredict1_1 = tf.keras.layers.Conv1D(filters=channel,kernel_size=3,strides=1,padding='same',name='predict1_1')
        self.convpredict1_2 = tf.keras.layers.Conv1D(filters=channel*5,kernel_size=1,strides=1,padding='valid',name='predict1_2',use_bias=True,bias_initializer=tf.keras.initializers.RandomNormal(),activation=tf.nn.relu)
        #update layer
        self.convupdate1_1 = tf.keras.layers.Conv1D(filters=channel,kernel_size=3,strides=1,padding='same',name='update1_1')
        self.convupdate1_2 = tf.keras.layers.Conv1D(filters=channel*5,kernel_size=1,strides=1,padding='valid',name='update1_2',use_bias=True,bias_initializer=tf.keras.initializers.RandomNormal(),activation=tf.nn.relu)

        #split layer 160
        self.convsplit2_d = tf.keras.layers.Conv1D(filters=channel*5,kernel_size=2,strides=2,padding='valid',name='split2_d')
        self.convsplit2_s = tf.keras.layers.Conv1D(filters=channel*5,kernel_size=2,strides=2,padding='valid',name='split2_s')
        #predict layer
        self.convpredict2_1 = tf.keras.layers.Conv1D(filters=channel*5,kernel_size=3,strides=1,padding='same',name='predict2_1')
        self.convpredict2_2 = tf.keras.layers.Conv1D(filters=channel*25,kernel_size=1,strides=1,padding='valid',name='predict2_2',use_bias=True,bias_initializer=tf.keras.initializers.RandomNormal(),activation=tf.nn.relu)
        #update layer
        self.convupdate2_1 = tf.keras.layers.Conv1D(filters=channel*5,kernel_size=3,strides=1,padding='same',name='update2_1')
        self.convupdate2_2 = tf.keras.layers.Conv1D(filters=channel*25,kernel_size=1,strides=1,padding='valid',name='update2_2',use_bias=True,bias_initializer=tf.keras.initializers.RandomNormal(),activation=tf.nn.relu)

        #split layer 80
        self.convsplit3_d = tf.keras.layers.Conv1D(filters=channel*25,kernel_size=2,strides=2,padding='valid',name='split3_d')
        self.convsplit3_s = tf.keras.layers.Conv1D(filters=channel*25,kernel_size=2,strides=2,padding='valid',name='split3_s')
        #predict layer
        self.convpredict3_1 = tf.keras.layers.Conv1D(filters=channel*25,kernel_size=3,strides=1,padding='same',name='predict3_1')
        self.convpredict3_2 = tf.keras.layers.Conv1D(filters=channel*125,kernel_size=1,strides=1,padding='valid',name='predict3_2',use_bias=True,bias_initializer=tf.keras.initializers.RandomNormal(),activation=tf.nn.relu)
        #update layer
        self.convupdate3_1 = tf.keras.layers.Conv1D(filters=channel*25,kernel_size=3,strides=1,padding='same',name='update3_1')
        self.convupdate3_2 = tf.keras.layers.Conv1D(filters=channel*125,kernel_size=1,strides=1,padding='valid',name='update3_2',use_bias=True,bias_initializer=tf.keras.initializers.RandomNormal(),activation=tf.nn.relu)
        # 40
        self.pooling = tf.keras.layers.GlobalMaxPool1D()
        self.flatten = tf.keras.layers.Flatten()
        self.FC = tf.keras.layers.Dense(class_num,activation=tf.keras.activations.sigmoid,name='predict')
    
    def call(self,input_tensor):
        x = self.input_layer(input_tensor)
        x = self.convinputs(x)
        #1
        x_d = self.convsplit1_d(x)
        x_s = self.convsplit1_s(x)

        x_p = self.convpredict1_1(x_s)
        x_p = tf.math.subtract(x_d, x_p)
        x = self.convpredict1_2(x_p)

        x_u = self.convupdate1_1(x)
        x_u = tf.math.add(x_u, x_s)
        x = self.convupdate1_2(x_u)
        #2
        x_d = self.convsplit2_d(x)
        x_s = self.convsplit2_s(x)

        x_p = self.convpredict2_1(x_s)
        x_p = tf.math.subtract(x_d, x_p)
        x = self.convpredict2_2(x_p)

        x_u = self.convupdate2_1(x)
        x_u = tf.math.add(x_u, x_s)
        x = self.convupdate2_2(x_u)
        #3
        x_d = self.convsplit3_d(x)
        x_s = self.convsplit3_s(x)

        x_p = self.convpredict3_1(x_s)
        x_p = tf.math.subtract(x_d, x_p)
        x = self.convpredict3_2(x_p)

        x_u = self.convupdate3_1(x)
        x_u = tf.math.add(x_u, x_s)
        x = self.convupdate3_2(x_u)
        #pooling
        x = self.pooling(x)
        x = self.flatten(x)
        x = self.FC(x)

        return x


def create_LiftNet(class_num = 5, channel = 3, netname='LiftNet', input_shape=(640,3),lr=0.01, momentum=0.8, decay=0.01):
    model = LiftNet(class_num=class_num,input_shape=input_shape,channel=channel,netname=netname)
    sgd = tf.keras.optimizers.SGD(lr=lr,momentum=momentum,decay=decay)
    model.compile(optimizer=sgd,loss=tf.keras.losses.SparseCategoricalCrossentropy(),metrics=['accuracy'])
    model.build((None, 640, 3))
    model.summary()
    return model


test_rate = 0.2
epochs = 10
lr=0.01
momentum=0.8
decay=0.01
validation_split=0.2
steps_per_epoch=1
validation_steps=1
circle_num =1
cutsize = 256
head_of_name = 'networks_liftnet_1110'

bunch_steps = 100


circle_num =1
input_shape = (640*circle_num,3)

save_model_name = head_of_name + str(circle_num) + '_' + str(epochs)+ '_' + str(steps_per_epoch) + '.h5'
loss_map_name = head_of_name + str(circle_num) + '_' + str(epochs)+ '_' + str(steps_per_epoch) +'.jpg'
dataset_file_name = 'dataset_' + str(circle_num) + '_' + str(cutsize) + '.npy'
label_file_name = 'label_' + str(circle_num) + '_' + str(cutsize) + '.npy'

dataset = np.load(dataset_file_name)
label = np.load(label_file_name)
print('dataset shape : ',dataset.shape)
print('label shape : ', label.shape)
x_number = dataset.shape[0]

x = dataset
print('x.shape: ',x.shape)
y = label
print('y.shape: ', y.shape)




x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_rate, random_state=56)

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


liftnet = create_LiftNet()
#liftnet.summary()

tbCallBack = TensorBoard(log_dir='./logs',  
                 histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
#                  batch_size=32,     # 用多大量的数据计算直方图
                 write_graph=True,  # 是否存储网络结构图
                 write_grads=True, # 是否可视化梯度直方图
                 write_images=True,# 是否可视化参数
                 embeddings_freq=0, 
                 embeddings_layer_names=None, 
                 embeddings_metadata=None)

#history = liftnet.fit(x_train,y_train,validation_split=validation_split,epochs=epochs,batch_size=bunch_size,steps_per_epoch=steps_per_epoch,validation_steps=validation_steps,callbacks=[tbCallBack])
history = liftnet.fit(x_train,y_train,validation_split=validation_split,epochs=epochs,batch_size=bunch_size,steps_per_epoch=steps_per_epoch,validation_steps=validation_steps)

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(num=0)
newepochs = range(1,len(loss)+1)
plt.plot(newepochs,loss,'b',label='train loss')
plt.plot(newepochs,val_loss,'r',label='val_loss')
plt.title('train and val loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
#plt.savefig(loss_map_name)

print('------------------------------------------------------------')
print('first predict: ')

#test
pre_y = liftnet.predict(x_test, steps=1)
#print(pre_y)
ml2.evaluate_model(pre_y,y_test)




liftnet.save_weights(save_model_name)
del liftnet
print('save and delete model')
print('------------------------------------------------------------')


print('------------------------------------------------------------')
print('create a new model and load model to predict:')


liftnet2 = create_LiftNet()
liftnet2.load_weights(save_model_name)

pre_y2 = liftnet2.predict(x_test, steps=1)
#print(pre_y)
ml2.evaluate_model(pre_y2,y_test)









#---------------------------------------------------------------------------------
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
#--------------------------------------------------------------------------------------------
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
#------------------------RMS-------------------------------#
square_data = dataset * dataset
x_number = dataset.shape[0]
print('x_number: ', x_number)
N = dataset.shape[1]
print('N:',N)
sum_square = np.sum(square_data,axis=1)/N
print('sum_square.shape(): ',sum_square.shape)
RMS = np.sqrt(sum_square)
print('RMS.shape(): ',RMS.shape)

#----------------------Variance-----------------------------#
data_mean = np.mean(dataset,axis=1)
print('data_mean.shape(): ',data_mean.shape)
data_std = np.std(dataset,axis=1)
print('data_std.shape(): ', data_std.shape)

new_data_mean = data_mean[:,np.newaxis,:]
padding_data_mean = []
for i in range(N):
    if len(padding_data_mean)==0:
        padding_data_mean = new_data_mean
        continue
    padding_data_mean = np.concatenate([padding_data_mean,new_data_mean],axis=1)
print('padding_data_mean.shape: ', padding_data_mean.shape)

d_data_mean = dataset - padding_data_mean
print('d_data_mean.shape: ',d_data_mean.shape)

Variance = np.sum(np.power(d_data_mean, 2), axis=1)/((N-1) * np.power(data_std, 2))
print('Variance.shape: ',Variance.shape)

#--------------------Skewness----------------------------#
Skewness = np.sum(np.power(d_data_mean, 3), axis=1)/((N-1) * np.power(data_std, 3))
print('Skewness.shape: ',Skewness.shape)

#--------------------Kurtosis----------------------------#
Kurtosis = np.sum(np.power(d_data_mean, 4), axis=1)/((N-1) * np.power(data_std, 4))
print('Kurtosis.shape: ', Kurtosis.shape)

#--------------------Shape factor------------------------#
SF = np.sqrt(np.sum(np.power(dataset, 2),axis=1)/N)/(np.sum(np.abs(dataset),axis=1)/N)
print('SF.shape: ',SF.shape)

#---------------------Crest factor----------------------#
CF = np.max(np.abs(dataset),axis=1)/RMS
print('CF.shape: ',CF.shape)

#---------------------impulse factor--------------------#
IF = np.max(np.abs(dataset),axis=1)/(np.sum(np.abs(dataset),axis=1)/N)
print('IF.shape: ',IF.shape)

#---------------------Margin factor---------------------#
MF = np.max(np.abs(dataset),axis=1)/np.power((np.sum(np.abs(dataset),axis=1)/N),2)
print('MF.shape: ',MF.shape)

#---------------------mobility-------------------------#
diff1_data = dataset[:,1:,:] - dataset[:,:-1,:]
print('diff1_data.shape: ',diff1_data.shape)
diff2_data = diff1_data[:,1:,:] - diff1_data[:,:-1,:]
print('diff2_data.shape: ', diff2_data.shape)

std_diff1 = np.std(diff1_data,axis=1)
std_diff2 = np.std(diff2_data,axis=1)

print('std_diff1.shape: ', std_diff1.shape)
print('std_diff2.shape: ', std_diff2.shape)

mobility = std_diff1 / data_std
print('mobility.shape: ', mobility.shape)

#----------------------complexity----------------------#
complexity = (std_diff2/std_diff1)/mobility
print('complexity.shape: ',complexity.shape)

#----------------------frequency centre----------------#
pi = math.pi
FC = np.sum(diff1_data * dataset[:,1:,:],axis=1)/(2*pi*np.sum(square_data,axis=1))
print('FC.shape: ',FC.shape)

#--------------------mean square frequency-------------#
MSF = np.sum(np.power(diff1_data,2),axis=1)/(4*pi*pi*np.sum(square_data,axis=1))
print('MSF.shape: ', MSF.shape)

#---------------root mean square frequency--------------#
RMSF = np.sqrt(MSF)
print('RMSF.shape: ', RMSF.shape)

#---------------root variance frequency----------------#
RVF = np.sqrt((MSF - np.power(FC,2)))
print('RVF.shape: ', RVF.shape)
#---------------------Peak amplitude-----------------------#
XP = np.max(np.abs(dataset),axis=1)
print('XP.shape: ', XP.shape)
#-----------------Peak to Peak amplitude-------------------#
XPP = np.max(dataset,axis=1) - np.min(dataset,axis=1)

#------------------organize feature--------------------#
XP = XP[:, np.newaxis, :]
XPP = XPP[:, np.newaxis, :]
sum_square = sum_square[:, np.newaxis, :]
RMS = RMS[:, np.newaxis, :]
data_mean = data_mean[:, np.newaxis, :]
data_std = data_std[:, np.newaxis, :]
Variance = Variance[:, np.newaxis, :]
Skewness = Skewness[:, np.newaxis, :]
Kurtosis = Kurtosis[:, np.newaxis, :]
SF = SF[:, np.newaxis, :]
CF = CF[:, np.newaxis, :]
IF = IF[:, np.newaxis, :]
MF = MF[:, np.newaxis, :]
mobility = mobility[:, np.newaxis, :]
complexity = complexity[:, np.newaxis, :]
FC = FC[:, np.newaxis, :]
MSF = MSF[:, np.newaxis, :]
RMSF = RMSF[:, np.newaxis, :]
RVF = RVF[:, np.newaxis, :]

feature_data = np.concatenate([XP,XPP,sum_square,RMS,data_mean,data_std,Variance,Skewness,Kurtosis,SF,CF,IF,MF,mobility,complexity,FC,MSF,RMSF,RVF],axis=1)
print('feature_data.shape: ', feature_data.shape)

x = feature_data.reshape(x_number,-1)
print('x.shape: ',x.shape)
y = label.reshape(x_number,)
print('y.shape: ', y.shape)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

print('X_train.shape: ', X_train.shape)
print('X_test.shape: ',X_test.shape)
print('y_train.shape: ', y_train.shape)
print('y_test.shape: ', y_test.shape)

params = {
    'booster': 'gbtree',
    'objective': 'multi:softmax',
    'num_class': 5,
    'gamma': 0.1,
    'max_depth': 38,
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

# 对测试集进行预测
dtest = xgb.DMatrix(X_test)
ans = model.predict(dtest)

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

# 显示重要特征
plot_importance(model)
plt.show()

#--------------------------------------------------------------------------------

dataset1 = np.loadtxt('dataset.csv',dtype=float,delimiter=',')
print('dataset shape : ')
print(dataset1.shape)
n = dataset1.shape[0]
l1 = dataset1.shape[1]
l1 = np.array(range(l1))
print(l1.shape)
print('n= ',n)

print('diff1')
diff11 = dataset1[:,1:]
diff12 = dataset1[:,:-1]
print(diff11.shape)
print(diff12.shape)
#diff = diff11 - diff12
diff = np.abs(np.abs(diff11) - np.abs(diff12))
print(diff.shape)
l2 = diff.shape[1]
l2 = np.array(range(l2))
print(l2.shape)

print('diff2:')
diff21 = diff[:,1:]
diff22 = diff[:,:-1]
print(diff21.shape)
print(diff22.shape)
#diff2 = diff21 - diff22
diff2 = np.abs(np.abs(diff21) - np.abs(diff22))
print(diff2.shape)
l3 = diff2.shape[1]
l3 = np.array(range(l3))
print(l3.shape)




for i in range(n):
    fig = plt.figure(num=i)
    ax1 = fig.add_subplot(3,1,1)
    ax2 = fig.add_subplot(3,1,2)
    ax3 = fig.add_subplot(3,1,3)
    ax1.plot(l1,dataset1[i,:])
    ax2.plot(l2,diff[i,:])
    ax3.plot(l3,diff2[i,:])
    save_name = './photo/data' + str(i) +'.jpg'
    savefig(save_name)

print('finished plot')




"""
#import mylib as ml2
"""
test_rate = 0.2
epochs = 1500
lr=0.01
momentum=0.8
decay=0.01
input_shape = 256000
validation_split=0.2
steps_per_epoch=1
validation_steps=1

dataset_file_name = 'dataset.csv'
label_file_name = 'label.csv'
dataset = ml2.read_data(dataset_file_name)
label = ml2.read_data(label_file_name)[:,np.newaxis]

#train_data, train_label, test_data, test_label, dataset, label = ml2.diliver_test_and_train_data(dataset,label,test_rate=test_rate)
_, _, _, _, dataset, label = ml2.diliver_test_and_train_data(dataset,label,test_rate=test_rate)
#x_train,y_train = train_data, train_label
#x_test, y_test = test_data, test_label

x_test, y_test = dataset, label

#print(x_train.shape)
#print(y_train.shape)

train_loss = []
validation_loss = []
train_step = 0

label_min=0
label_range=1

pre_data_min = 0
pre_data_range = 1

model = ml2.create_networks(input_shape=input_shape,lr=lr,momentum=momentum,decay=decay)

model.load_weights('network1_1021.h5')
sgd = tf.keras.optimizers.SGD(lr=lr,momentum=momentum,decay=decay)
model.compile(optimizer=sgd,loss=tf.keras.losses.SparseCategoricalCrossentropy(),metrics=['accuracy'])
model.summary()

#history = model.fit(x_train,y_train,validation_split=validation_split,epochs=epochs,steps_per_epoch=steps_per_epoch,validation_steps=validation_steps)
#loss = history.history['loss']
#val_loss = history.history['val_loss']
#model.save('model_with_conv1d_final.h5')  

pre_y=model.predict(x_test,steps=1)
pre_y = ml2.pre_to_index(pre_y)

y_test = y_test.astype(np.uint8)
pre_y = pre_y.astype(np.uint8)
print('truth:')
print(y_test.T)
print('predict:')
print(pre_y.T)

for i in range(5):
    tp, tn, fp, fn = ml2.count_tptnfpfn(y_test,pre_y,i)
    label_name = ['normal', 'inner_ring', 'outer_ring', 'roller', 'joint']
    print(label_name[i] + ' result:')
    print('tp:',tp,' tn:',tn,' fp:',fp,' fn:',fn)
    print('tp/(tp+tn): ', tp/(tp+tn))
    print('tp/(tp+fp): ', tp/(tp+fp))

plt.figure(num=0)
epochs = range(1,len(loss)+1)
plt.plot(epochs,loss,'b',label='train loss')
plt.plot(epochs,val_loss,'r',label='val_loss')
plt.title('train and val loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

file_name = '/media/silverbullet/data_and_programing_file/newpaper/dataset/ER16k/joint/REC2490_ch1.txt'
#file_name = '/media/silverbullet/data_and_programing_file/newpaper/dataset/ER16k'
#save_path = '/media/silverbullet/data_and_programing_file/newpaper/dataset/photo'

data1 = np.loadtxt(file_name,skiprows=16)
data2 = data1[:256000,:].T
print(data1.shape)
print(data2.shape)

def read_and_savefig(loadpath,savepath,fig_size=(20,2)):
    label_name = ['/normal/', '/inner_ring/', '/outer_ring/', '/roller/', '/joint/']
    fig_num = 0
    for name in label_name:
        fnames = glob.glob(loadpath + name + '*.txt')
        for file_name in fnames:
            read_data = np.loadtxt(file_name,skiprows=16)
            f1 = plt.figure(num=fig_num,figsize=fig_size)
            plot(read_data[:,0],read_data[:,1])
            savefig_path = file_name.replace(loadpath,save_path)
            savefig_path = savefig_path.replace('.txt','.jpg')
            savefig(savefig_path)
            fig_num += 1

def read_and_savedataset(loadpath,savepath):
    label_name = ['/normal/', '/inner_ring/', '/outer_ring/', '/roller/', '/joint/']
    label = []
    dataset = []
    for i in range(len(label_name)):
        name = label_name[i]
        print(name,i)
        fnames = glob.glob(loadpath + name + '*.txt')
        for file_name in fnames:
            read_data = np.loadtxt(file_name,skiprows=16)
            read_data = read_data[:256000,1].T
            read_data = read_data[np.newaxis,:]
            if len(dataset)==0:
                dataset = read_data
                label.append(i)
                continue
            dataset = np.concatenate((dataset,read_data),axis = 0)
            label.append(i)
    label = np.array(label)
    label = label[:,np.newaxis]
    print(label.shape)
    print(dataset.shape)
    total_dataset = np.concatenate((label,dataset),axis=1)
    print(total_dataset.shape)
    np.savetxt(savepath + '/label.csv', label, delimiter = ',')
    np.savetxt(savepath + '/dataset.csv', dataset, delimiter = ',')
    np.savetxt(savepath + '/total_dataset.csv', total_dataset, delimiter = ',')
    print('dataset saved')

"""
"""           
def read_and_plot2(filename,savepath,cutsize=1000,fig_size=(20,2)):
    read_data = np.loadtxt(filename,skiprows=16)[:256000,:]
    split_data = np.vsplit(read_data,cutsize)
    for i in range(len(split_data)):
        plt.figure(num=i,figsize=fig_size)
        plot(split_data[i][:,0],split_data[i][:,1])
        savefig_path = savepath + str(i) + '.jpg'
        savefig(savefig_path)

file_name1 = '/media/silverbullet/data_and_programing_file/newpaper/dataset/ER16k/normal/REC2449_ch2.txt'
file_name2 = '/media/silverbullet/data_and_programing_file/newpaper/dataset/ER16k/normal/REC2449_ch3.txt'
file_name3 = '/media/silverbullet/data_and_programing_file/newpaper/dataset/ER16k/normal/REC2449_ch4.txt'

"""
load_path = '/home/silver-bullet/newpaper/data/ER16k'
save_path = '/home/silver-bullet/newpaper/data/dataset'
cutsize = 256
num_circle = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
for num in num_circle:
    print('start ' + str(num) + 'th data read')
    read_split_and_savedataset(load_path,save_path,num_circle=num)

"""


data1 =  np.loadtxt(file_name1,skiprows=16)[:256000,1]
data2 =  np.loadtxt(file_name2,skiprows=16)[:256000,1]
data3 =  np.loadtxt(file_name3,skiprows=16)[:256000,1]
data = np.vstack((data1,data2))
data = np.vstack((data,data3)).T
print(data.shape)
print(data.shape[1])
print(len(data.shape))
split_data, num = split_dataset(data,num_circle=num_circle,cutsize=cutsize)



#fig_size=(20,2)



#read_and_plot(file_name,save_path,cutsize)


read_data = np.loadtxt(file_name,skiprows=16)[:256000,:]
split_data = np.vsplit(read_data,cutsize)
index = []
max_min = 0.2
for i in range(len(split_data)):
    if np.max(split_data[i][:,1]) >= max_min:
        index.append(1)
        continue
    if np.min(split_data[i][:,1]) <= -max_min:
        index.append(1)
        continue
    index.append(0)
print(index)
start = 1
end = 1

first_one = 0
final_one = len(index)-1
setp1 = 1
setp2 = 1
while start:
    if index[first_one] == 1:
        break
    first_one += 1
    setp1 += 1
    if setp1 >= len(index):
        print('cannot find the one')
        break

while end:
    if index[final_one] == 1:
        break
    final_one = final_one - 1
    setp2 += 1
    if setp1 >= len(index):
        print('cannot find the one')
        break
print('first: ' + str(first_one) + ' final: ' + str(final_one))

stack_data = []
for i in range(first_one,final_one):
    if len(stack_data)==0:
        stack_data = split_data[i]
        continue
    stack_data = np.vstack((stack_data,split_data[i]))
print('end vstack')
print(stack_data.shape)

plt.figure(num=i,figsize=fig_size)
plot(stack_data[:,0],stack_data[:,1])
savefig('test.jpg')
print('end plot')



print('end')





file_path = '/media/silverbullet/data_and_programing_file/newpaper/dataset/ER16k'
save_path = '/media/silverbullet/data_and_programing_file/newpaper/dataset'
read_and_savedataset(file_path,save_path)

#path = '/media/silverbullet/data_and_programing_file/newpaper/dataset/ER16k/joint/'
#path = '/media/silverbullet/data_and_programing_file/newpaper/dataset/ER16k/normal/'
path = '/media/silverbullet/data_and_programing_file/newpaper/dataset/ER16k/inner_ring/'
#path = '/media/silverbullet/data_and_programing_file/newpaper/dataset/ER16k/outer_ring/'
#path = '/media/silverbullet/data_and_programing_file/newpaper/dataset/ER16k/roller/'
fnames = glob.glob(path+'*.txt')
for file_name in fnames:
    data = np.loadtxt(file_name,skiprows=16)
    print(data.shape)
    if data.shape[0] >256000:
        print(file_name)
    if data.shape[0] <256000:
        print(file_name)
    
"""



