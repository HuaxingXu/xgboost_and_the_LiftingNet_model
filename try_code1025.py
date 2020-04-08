import tensorflow as tf
import numpy as np
import math
import glob
import matplotlib

import matplotlib.pyplot as plt
from matplotlib.pyplot import plot,savefig
import mylib as ml2
from keras.callbacks import TensorBoard

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



train_data, train_label, test_data, test_label, dataset, label = ml2.diliver_test_and_train_newdata(dataset,label,test_rate=test_rate)
print('train_data shape : ',train_data.shape)
print('train_label shape : ', train_label.shape)
print('test_data shape : ',test_data.shape)
print('test_label shape : ', test_label.shape)
print('dataset shape : ',dataset.shape)
print('label shape : ', label.shape)

x_train,y_train = train_data, train_label
x_test, y_test = test_data, test_label
#x_test, y_test = dataset, label

print(x_train.shape)
print(y_train.shape)

tbCallBack = TensorBoard(log_dir='./logs',  
                 histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
#                  batch_size=32,     # 用多大量的数据计算直方图
                 write_graph=True,  # 是否存储网络结构图
                 write_grads=True, # 是否可视化梯度直方图
                 write_images=True,# 是否可视化参数
                 embeddings_freq=0, 
                 embeddings_layer_names=None, 
                 embeddings_metadata=None)
#create model
model = ml2.create_networks7(input_shape=input_shape,lr=lr,momentum=momentum,decay=decay)

#train and plot
#model = ml2.train_and_plot(model,x_train,y_train,validation_split=validation_split,epochs=epochs,steps_per_epoch=steps_per_epoch,validation_steps=validation_steps,save_model_name=save_model_name)

history = model.fit(x_train,y_train,validation_split=validation_split,epochs=epochs,steps_per_epoch=steps_per_epoch,validation_steps=validation_steps,callbacks=[tbCallBack])
loss = history.history['loss']
val_loss = history.history['val_loss']
model.save(save_model_name)

plt.figure(num=0)
newepochs = range(1,len(loss)+1)
plt.plot(newepochs,loss,'b',label='train loss')
plt.plot(newepochs,val_loss,'r',label='val_loss')
plt.title('train and val loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.savefig(loss_map_name)

#test
pre_y = model.predict(x_test, steps=1)

ml2.evaluate_model(pre_y,y_test)


