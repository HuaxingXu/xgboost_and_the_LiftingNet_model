from __future__ import  absolute_import
from __future__ import  division
import numpy as np
from scipy.io import loadmat
from dataset import *
import glob

datapath = '/home/silver-bullet/newpaper/data/CaseWesternReserveUniversityData-master/12K_Drive_End/1730/7/ball/'
datapath2 = '/home/silver-bullet/newpaper/data/CaseWesternReserveUniversityData-master/12K_Drive_End/1730/14/ball/'
data_file_name = datapath + '12k_Drive_End_B007_3_121.mat'
data_file_name2 = datapath2 + '12k_Drive_End_B014_3_188.mat'

m = loadmat(data_file_name)
m2 = loadmat(data_file_name2)

#print(m.items())
print('m.keys: ', m.keys())
print(m['X121_DE_time'].shape)
print(m['X121_FE_time'].shape)
print(m['X121RPM'])
print(m['X121_BA_time'].shape)

#print('m2.keys: ', m2.keys())

items_key = list(m.keys())
print(items_key)
#print(items_key[1])
#print(items_key[2].find('RPM'))
for i in range(len(items_key)):
    print(items_key[i].find('RPM'))


def find_item_name(items_list, item_key):
    for i in range(len(items_list)):
        if items_list[i].find(item_key)>=0:
            return items_list[i]
    return -1



origin_data = loadmat(data_file_name)

data_header = ['DE_time', 'FE_time', 'BA_time']
items_list = list(origin_data.keys())

read_data = []

for i in range(len(data_header)):
    current_item = find_item_name(items_list, data_header[i])
    current_data = origin_data[current_item][np.newaxis, :, :]
    if read_data==[]:
        read_data = current_data
        continue
    read_data = np.concatenate([read_data,current_data],axis=2)
print(read_data.shape)




data_path = '/home/silver-bullet/newpaper/data/CaseWesternReserveUniversityData-master'
save_path = '/home/silver-bullet/newpaper/data/CWRUdataset'
origin_index = 0
first_index = 0
second_index = 0
cut_size = 1024

dataset, label = read_and_save_CWRU_data(data_path,save_path)

dataset, label = expansion_and_add_noise(dataset,label,exnumber=500,noise_scales=0.1)
print('dataset.shape: ', dataset.shape)
print('label.shape: ', label.shape)

"""
origin_level_document = ['/12K_Drive_End', '/48K_Drive_End', '/Fan_End']

first_level_document = ['/1797', '/1772', '/1750', '/1730']

second_level_document = ['/7', '/14', '/21', '/28']

third_level_document = ['/normal/', '/ball/', '/inner_ring/', '/outer_ring_3/', '/outer_ring_6/', '/outer_ring_12/']


read_category = data_path + origin_level_document[origin_index] + first_level_document[first_index] + second_level_document[second_index]

print(read_category)
dataset = []
label = []
for i in range(len(third_level_document)):
    read_path = read_category + third_level_document[i]
    fnames = glob.glob(read_path + '*.mat')
    for file_name in fnames:
        read_data = read_CWRU_data(file_name)
        read_data, read_label = cut_CWRU_data(read_data, i, cut_size=cut_size)
        if dataset == []:
            dataset = read_data
            label.extend(read_label)
            continue
        dataset = np.concatenate((dataset, read_data), axis=0)
        label.extend(read_label)
label = np.array(label)
label = label[:,np.newaxis]
print(dataset.shape)
print(label.shape)
        

"""

"""

#a = np.array([[[1],[2],[3],[4],[5],[6]],[[7],[8],[9],[10],[11],[12]]])
a = np.array([[[1,2],[3,4],[5,6],[7,8],[9,10],[11,12]]])

print('a.shape: ', a.shape)
print(a)
print(np.reshape(a,(-1,3,2)).shape)
print(np.reshape(a,(-1,3,2)))

b = [1 for _ in range(10)]
print(b*2)
"""