import tensorflow as tf
import numpy as np
import math
import glob
import matplotlib

import matplotlib.pyplot as plt
from matplotlib.pyplot import plot,savefig
import mylib as ml2

test_rate = 0.2
epochs = 2000
lr=0.01
momentum=0.8
decay=0.01
input_shape = 256000
validation_split=0.2
steps_per_epoch=1
validation_steps=1
save_model_name = 'network1_1021.h5'

dataset_file_name = 'dataset.csv'
label_file_name = 'label.csv'
dataset = ml2.read_data(dataset_file_name)
label = ml2.read_data(label_file_name)[:,np.newaxis]

train_data, train_label, test_data, test_label, dataset, label = ml2.diliver_test_and_train_data(dataset,label,test_rate=test_rate)
print('train_data shape : ',train_data.shape)
print('train_label shape : ', train_label.shape)
print('test_data shape : ',test_data.shape)
print('test_label shape : ', test_label.shape)
print('dataset shape : ',dataset.shape)
print('label shape : ', label.shape)
"""
x_train,y_train = train_data, train_label
x_test, y_test = test_data, test_label
#x_test, y_test = dataset, label

print(x_train.shape)
print(y_train.shape)

#create model
model = ml2.create_networks(input_shape=input_shape,lr=lr,momentum=momentum,decay=decay)

#train and plot
#model = ml2.train_and_plot(model,x_train,y_train,validation_split=validation_split,epochs=epochs,steps_per_epoch=steps_per_epoch,validation_steps=validation_steps,save_model_name=save_model_name)

history = model.fit(x_train,y_train,validation_split=validation_split,epochs=epochs,steps_per_epoch=steps_per_epoch,validation_steps=validation_steps)
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
plt.savefig('network1_loss_1021.jpg')

#test
pre_y = model.predict(x_test, steps=1)

ml2.evaluate_model(pre_y,y_test)
"""