import tensorflow as tf
import numpy as np
import math
import glob
import matplotlib.pyplot as plt
import mylib as ml2
import random

test_rate = 0.2
epochs = 50
lr=0.01
momentum=0.8
decay=0.01
input_shape = 19
if input_shape == 11:
    add_orgin_data = False
else:
    add_orgin_data = True

batch = 200
test_batch = 100

validation_split=0.2
steps_per_epoch=5
validation_steps=1

file_path = '/media/silverbullet/data_and_programing_file/GMCM/data/train_set/'


#file_path = '/media/silverbullet/data_and_programing_file/GMCM/data2/'

files = glob.glob(file_path+'*.csv')
num_files = len(files)


test_num = round(test_rate * num_files)
train_num = num_files - test_num
train_index = random.sample(range(0,num_files-1),train_num)
print('train_num:',train_num)
#print(train_index)

file_index = np.zeros([num_files,1],dtype=np.int)
#file_index = range(0,0,num_files)
file_index[train_index] = 1
file_index=file_index.tolist()

#print(file_index)
train_files = []
test_files = []
for i in range(len(file_index)):
    if file_index[i][0] == 1:
        train_files.append(files[i])
    else:
        test_files.append(files[i])

#train_files = files[file_index == 1]
#test_files = files[file_index == 0]
train_loss = []
validation_loss = []
train_step = 0

label_min=0
label_range=1

pre_data_min = 0
pre_data_range = 1

batch_step = round((train_num / batch))
print(batch_step)
for i in range(batch_step):
    data = []
    for j in range(batch):
        files_num = i*batch+j
        file_name = train_files[files_num]
        if len(data)==0:
            data = ml2.read_data(file_name)
        else:
            read_datas = ml2.read_data(file_name)
            data = np.vstack((data,read_datas))
    
    #data = ml2.read_data(fname)
    pre_data, label = ml2.pre_deal_data(data,add_orgin_data=add_orgin_data)
    #pre_data = ml2.add_new_axis(pre_data)

    x, x_min, x_range = ml2.normalization(pre_data)
    y, y_min, y_range = ml2.normalization(label)
    print('x: ',x.shape,'y: ',y.shape)
    x_train ,y_train, x_test, y_test = x, y, x, y

    if train_step == 0:
        model = ml2.create_networks2(input_shape=input_shape,lr=lr,momentum=momentum,decay=decay)
        pre_data_min, pre_data_range, label_min, label_range = x_min, x_range, y_min, y_range

    else:
        model = ml2.create_networks2(input_shape=input_shape,lr=lr,momentum=momentum,decay=decay)
        model.load_weights('model_with_conv1d_final.h5')
        sgd = tf.keras.optimizers.SGD(lr=lr,momentum=momentum,decay=decay)
        model.compile(optimizer=sgd,loss='mse',metrics=['accuracy'])
        model.summary()
        pre_data_min, pre_data_range = ml2.new_min_range(pre_data_min, x_min, pre_data_range, x_range)
        label_min, label_range = ml2.new_min_range(label_min, y_min, label_range, y_range)
        """
        new_label_min = label_min
        new_label_range = label_range
        if y_min<label_min:
            new_label_min = y_min
        if label_min + label_range<y_min+y_range:
            new_label_max =y_min+y_range
            new_label_range = new_label_max - new_label_min

        label_min = new_label_min
        label_range = new_label_range
        """
    
    train_step = train_step+1

    history = model.fit(x_train,y_train,validation_split=validation_split,epochs=epochs,steps_per_epoch=steps_per_epoch,validation_steps=validation_steps)

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_loss.extend(loss)
    validation_loss.extend(val_loss)
    model.save('model_with_conv1d_final.h5')  

    del model

test_step = 0
TP,TN,FP,FN = 0, 0, 0, 0
test_batch_step = round((test_num / test_batch))



for i in range(test_batch_step):
    data = []
    for j in range(test_batch):
        files_num = i*test_batch+j
        file_name = train_files[files_num]
        if len(data)==0:
            data = ml2.read_data(file_name)
        else:
            read_datas = ml2.read_data(file_name)
            data = np.vstack((data,read_datas))
    pre_data, truth_y = ml2.pre_deal_data(data,add_orgin_data=add_orgin_data)
    #pre_data = ml2.add_new_axis(pre_data)

    x_test = ml2.renormalization(pre_data,pre_data_min, pre_data_range)
    #y, y_mean, y_var = ml2.standardization(label)

    model = ml2.create_networks2(input_shape=input_shape,lr=lr,momentum=momentum,decay=decay)
    model.load_weights('model_with_conv1d_final.h5')
    sgd = tf.keras.optimizers.SGD(lr=lr,momentum=momentum,decay=decay)
    model.compile(optimizer=sgd,loss='mse',metrics=['accuracy'])
    model.summary()

    pre_y=model.predict(x_test,steps=1)
    pre_y=pre_y * label_range + label_min

    tp, tn, fp, fn = ml2.count_tptnfpfn(truth_y,pre_y)
    TP = TP+tp
    TN = TN+tn
    FP = FP+fp
    FN = FN+fn

    if test_step == 0:
        predict_y = pre_y
        y_truth = truth_y
    else:
        predict_y = np.vstack((predict_y,pre_y))
        y_truth = np.vstack((y_truth,truth_y))



np.savetxt('pre_data_min.csv',pre_data_min,delimiter=',')
np.savetxt('pre_data_range.csv',pre_data_range,delimiter=',')
np.savetxt('label_min.csv',label_min,delimiter=',')
np.savetxt('label_range.csv',label_range,delimiter=',')

np.savetxt('train_loss.csv',train_loss,delimiter=',')
np.savetxt('validation_loss.csv',validation_loss,delimiter=',')

np.savetxt('predict_y.csv',predict_y,delimiter=',')
np.savetxt('truth_y.csv',y_truth,delimiter=',')
#pcrr=ml2.count_pcrr(y_truth,predict_y)
print('tp:',TP)
print('tn:',TN)
print('fp:',FP)
print('fn:',FN)

"""
p = TP/(TP+FP)
r = TP/(TP+FN)
prcc = 2*p*r/(p+r)
print('precision: ',p)
print('recall: ',r)
print('pcrr: ',pcrr)
"""

plt.figure(num=0)
epochs = range(1,len(train_loss)+1)
plt.plot(epochs,train_loss,'b',label='train loss')
plt.plot(epochs,validation_loss,'r',label='val_loss')
plt.title('train and val loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()


plt.figure(num=2)
num_y= len(predict_y)
epochs2 = range(1,len(predict_y)+1)
plt.plot(epochs2,predict_y,'r',label='predict y')
plt.plot(epochs2,y_truth,'b',label='truth y')
plt.title('predict and truth y')
plt.xlabel('epochs')
plt.ylabel('y')
plt.legend()
plt.show()