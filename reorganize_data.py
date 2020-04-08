import numpy as np
from dataset import *

"""
data_read_path = '/media/silverbullet/data_and_programing_file/newpaper/dataset/ER16k'
trans_data_save_path = '/media/silverbullet/data_and_programing_file/newpaper/dataset/our_data_trans_to_CWRU_data/length_1024'

CWRU_data_long = 1024

read_split_and_savedataset(data_read_path,trans_data_save_path,whether_trans_to_CWRU_data=1,CWRU_data_length=CWRU_data_long)

"""

data_path = '/media/silverbullet/data_and_programing_file/newpaper/dataset/CaseWesternReserveUniversityData-master'
save_path = '/media/silverbullet/data_and_programing_file/newpaper/dataset/CWRU_data_trans_to_our_data'
trans_label = [3, 1, 2, 2, 2]
circles = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

for circle in circles:
    dataset, label = read_and_save_CWRU_data(data_path, save_path=save_path,whether_trans_to_our_data=1,trans_length=circle,trans_label=trans_label)
    print('trans '+ str(circle)+' circles data')
    print('dataset.shape: ', dataset.shape)
    print('label.shape: ', label.shape)
    print(label.T)
    expansion_data, expansion_label = data_expension2(dataset,label,exnumber=10)
    print(expansion_label.T)
    