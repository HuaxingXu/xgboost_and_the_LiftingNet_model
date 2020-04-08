"""
# MIT Copyright 2019
# zhaoyang
# 2019 10 19
"""
from __future__ import  absolute_import
from __future__ import  division
import tensorflow as tf
import numpy as np
import math
import glob
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot,savefig
import random

def create_networks(input_shape=256000,lr=0.01, momentum=0.8, decay=0.01):
    net = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(input_shape,1),name='input'),
        tf.keras.layers.Conv1D(filters=3,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d1_1'),
        tf.keras.layers.MaxPool1D(pool_size=2,strides=2,name='maxpool_1'),
        #128000 * 3
        tf.keras.layers.Conv1D(filters=8,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d2_1'),
        tf.keras.layers.Conv1D(filters=8,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d2_2'),
        tf.keras.layers.MaxPool1D(pool_size=2,strides=2,name='maxpool_2'),
        #64000 * 8
        tf.keras.layers.Conv1D(filters=16,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d3_1'),
        tf.keras.layers.Conv1D(filters=16,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d3_2'),
        tf.keras.layers.MaxPool1D(pool_size=2,strides=2,name='maxpool_3'),
        #32000 * 16
        tf.keras.layers.Conv1D(filters=32,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d4_1'),
        tf.keras.layers.Conv1D(filters=32,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d4_2'),
        tf.keras.layers.MaxPool1D(pool_size=2,strides=2,name='maxpool_4'),
        #16000 * 32
        tf.keras.layers.Conv1D(filters=64,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d5_1'),
        tf.keras.layers.Conv1D(filters=64,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d5_2'),
        tf.keras.layers.MaxPool1D(pool_size=2,strides=2,name='maxpool_5'),
        #8000 * 64
        tf.keras.layers.Conv1D(filters=128,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d6_1'),
        tf.keras.layers.Conv1D(filters=128,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d6_2'),
        tf.keras.layers.MaxPool1D(pool_size=2,strides=2,name='maxpool_6'),
        #4000 * 128
        tf.keras.layers.Conv1D(filters=128,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d7_1'),
        tf.keras.layers.Conv1D(filters=128,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d7_2'),
        tf.keras.layers.MaxPool1D(pool_size=2,strides=2,name='maxpool_7'),
        #2000 * 128
        tf.keras.layers.Conv1D(filters=64,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d8_1'),
        tf.keras.layers.Conv1D(filters=64,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d8_2'),
        tf.keras.layers.MaxPool1D(pool_size=2,strides=2,name='maxpool_8'),
        #1000 * 64
        tf.keras.layers.Conv1D(filters=32,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d9_1'),
        tf.keras.layers.Conv1D(filters=32,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d9_2'),
        tf.keras.layers.MaxPool1D(pool_size=2,strides=2,name='maxpool_9'),
        #500 * 32
        tf.keras.layers.Conv1D(filters=16,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d10_1'),
        tf.keras.layers.Conv1D(filters=16,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d10_2'),
        tf.keras.layers.MaxPool1D(pool_size=2,strides=2,name='maxpool_10'),
        #250 * 16
        tf.keras.layers.Conv1D(filters=8,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d11_1'),
        tf.keras.layers.Conv1D(filters=8,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d11_2'),
        tf.keras.layers.MaxPool1D(pool_size=2,strides=2,name='maxpool_11'),
        #125 * 8
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(500,activation=tf.nn.relu,name='dense_1'),
        tf.keras.layers.Dense(500,activation=tf.nn.relu,name='dense_2'),
        tf.keras.layers.Dense(250,activation=tf.nn.relu,name='dense_3'),
        tf.keras.layers.Dense(250,activation=tf.nn.relu,name='dense_4'),
        tf.keras.layers.Dense(125,activation=tf.nn.relu,name='dense_5'),
        tf.keras.layers.Dense(25,activation=tf.nn.relu,name='dense_6'),
        #tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(5,activation=tf.nn.softmax,name='dense_final')
    ])
    sgd = tf.keras.optimizers.SGD(lr=lr,momentum=momentum,decay=decay)
    net.compile(optimizer=sgd,loss=tf.keras.losses.SparseCategoricalCrossentropy(),metrics=['accuracy'])
    net.summary()
    return net

def create_networks2(input_shape=256000,lr=0.01, momentum=0.8, decay=0.01):
    net = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(input_shape,1),name='input'),
        tf.keras.layers.Conv1D(filters=4,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d1_1'),
        tf.keras.layers.MaxPool1D(pool_size=5,strides=5,name='maxpool_1'),
        #25600 * 3
        tf.keras.layers.Conv1D(filters=8,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d2_1'),
        tf.keras.layers.Conv1D(filters=8,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d2_2'),
        tf.keras.layers.MaxPool1D(pool_size=2,strides=2,name='maxpool_2'),
        #12800 * 8
        tf.keras.layers.Conv1D(filters=16,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d3_1'),
        tf.keras.layers.Conv1D(filters=16,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d3_2'),
        tf.keras.layers.MaxPool1D(pool_size=2,strides=2,name='maxpool_3'),
        #6400 * 16
        tf.keras.layers.Conv1D(filters=32,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d4_1'),
        tf.keras.layers.Conv1D(filters=32,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d4_2'),
        tf.keras.layers.MaxPool1D(pool_size=2,strides=2,name='maxpool_4'),
        #3200 * 32
        tf.keras.layers.Conv1D(filters=64,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d5_1'),
        tf.keras.layers.Conv1D(filters=64,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d5_2'),
        tf.keras.layers.MaxPool1D(pool_size=2,strides=2,name='maxpool_5'),
        #1600 * 64
        tf.keras.layers.Conv1D(filters=128,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d6_1'),
        tf.keras.layers.Conv1D(filters=128,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d6_2'),
        tf.keras.layers.MaxPool1D(pool_size=2,strides=2,name='maxpool_6'),
        #800 * 128
        tf.keras.layers.Conv1D(filters=128,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d7_1'),
        tf.keras.layers.Conv1D(filters=128,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d7_2'),
        tf.keras.layers.MaxPool1D(pool_size=2,strides=2,name='maxpool_7'),
        #400 * 128
        tf.keras.layers.Conv1D(filters=64,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d8_1'),
        tf.keras.layers.Conv1D(filters=64,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d8_2'),
        tf.keras.layers.MaxPool1D(pool_size=2,strides=2,name='maxpool_8'),
        #200 * 64
        tf.keras.layers.Conv1D(filters=32,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d9_1'),
        tf.keras.layers.Conv1D(filters=32,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d9_2'),
        tf.keras.layers.MaxPool1D(pool_size=2,strides=2,name='maxpool_9'),
        #100 * 32
        tf.keras.layers.Conv1D(filters=16,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d10_1'),
        tf.keras.layers.Conv1D(filters=16,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d10_2'),
        tf.keras.layers.MaxPool1D(pool_size=2,strides=2,name='maxpool_10'),
        #50 * 16
        tf.keras.layers.Conv1D(filters=8,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d11_1'),
        tf.keras.layers.Conv1D(filters=8,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d11_2'),
        tf.keras.layers.MaxPool1D(pool_size=2,strides=2,name='maxpool_11'),
        #25 * 8
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(100,activation=tf.nn.relu,name='dense_1'),
        tf.keras.layers.Dense(50,activation=tf.nn.relu,name='dense_2'),
        tf.keras.layers.Dense(50,activation=tf.nn.relu,name='dense_3'),
        tf.keras.layers.Dense(25,activation=tf.nn.relu,name='dense_4'),
        tf.keras.layers.Dense(25,activation=tf.nn.relu,name='dense_5'),
        #tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(5,activation=tf.nn.softmax,name='dense_final')
    ])
    sgd = tf.keras.optimizers.SGD(lr=lr,momentum=momentum,decay=decay)
    net.compile(optimizer=sgd,loss=tf.keras.losses.SparseCategoricalCrossentropy(),metrics=['accuracy'])
    net.summary()
    return net

def create_networks3(input_shape=256000,lr=0.01, momentum=0.8, decay=0.01):
    net = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(input_shape,1),name='input'),
        tf.keras.layers.Conv1D(filters=4,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d1_1'),
        tf.keras.layers.MaxPool1D(pool_size=2,strides=2,name='maxpool_1'),
        #128000 * 3
        tf.keras.layers.Conv1D(filters=8,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d2_1'),
        tf.keras.layers.Conv1D(filters=8,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d2_2'),
        tf.keras.layers.MaxPool1D(pool_size=2,strides=2,name='maxpool_2'),
        #64000 * 8
        tf.keras.layers.Conv1D(filters=16,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d3_1'),
        tf.keras.layers.Conv1D(filters=16,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d3_2'),
        tf.keras.layers.MaxPool1D(pool_size=2,strides=2,name='maxpool_3'),
        #32000 * 16
        tf.keras.layers.Conv1D(filters=32,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d4_1'),
        tf.keras.layers.Conv1D(filters=32,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d4_2'),
        tf.keras.layers.MaxPool1D(pool_size=2,strides=2,name='maxpool_4'),
        #16000 * 32
        tf.keras.layers.Conv1D(filters=64,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d5_1'),
        tf.keras.layers.Conv1D(filters=64,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d5_2'),
        tf.keras.layers.MaxPool1D(pool_size=2,strides=2,name='maxpool_5'),
        #8000 * 64
        tf.keras.layers.Conv1D(filters=128,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d6_1'),
        tf.keras.layers.Conv1D(filters=128,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d6_2'),
        tf.keras.layers.MaxPool1D(pool_size=2,strides=2,name='maxpool_6'),
        #4000 * 128
        tf.keras.layers.Conv1D(filters=128,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d7_1'),
        tf.keras.layers.Conv1D(filters=128,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d7_2'),
        tf.keras.layers.MaxPool1D(pool_size=2,strides=2,name='maxpool_7'),
        #2000 * 128
        tf.keras.layers.Conv1D(filters=64,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d8_1'),
        tf.keras.layers.Conv1D(filters=64,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d8_2'),
        tf.keras.layers.MaxPool1D(pool_size=2,strides=2,name='maxpool_8'),
        #1000 * 64
        tf.keras.layers.Conv1D(filters=32,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d9_1'),
        tf.keras.layers.Conv1D(filters=32,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d9_2'),
        tf.keras.layers.MaxPool1D(pool_size=2,strides=2,name='maxpool_9'),
        #500 * 32
        tf.keras.layers.Conv1D(filters=16,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d10_1'),
        tf.keras.layers.Conv1D(filters=16,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d10_2'),
        tf.keras.layers.MaxPool1D(pool_size=2,strides=2,name='maxpool_10'),
        #250 * 16
        tf.keras.layers.Conv1D(filters=8,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d11_1'),
        tf.keras.layers.Conv1D(filters=8,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d11_2'),
        tf.keras.layers.MaxPool1D(pool_size=2,strides=2,name='maxpool_11'),
        #125 * 8

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(500,activation=tf.nn.relu,name='dense_1'),
        tf.keras.layers.Dense(250,activation=tf.nn.relu,name='dense_2'),
        tf.keras.layers.Dense(250,activation=tf.nn.relu,name='dense_3'),
        tf.keras.layers.Dense(125,activation=tf.nn.relu,name='dense_4'),
        tf.keras.layers.Dense(125,activation=tf.nn.relu,name='dense_5'),
        tf.keras.layers.Dense(30,activation=tf.nn.relu,name='dense_6'),
        #tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(5,activation=tf.nn.softmax,name='dense_final')
    ])
    sgd = tf.keras.optimizers.SGD(lr=lr,momentum=momentum,decay=decay)
    net.compile(optimizer=sgd,loss=tf.keras.losses.SparseCategoricalCrossentropy(),metrics=['accuracy'])
    net.summary()
    return net

def create_networks4(input_shape=256000,lr=0.01, momentum=0.8, decay=0.01):
    net = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(input_shape,1),name='input'),
        tf.keras.layers.Conv1D(filters=3,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d1_1'),
        tf.keras.layers.Conv1D(filters=8,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d1_2'),
        tf.keras.layers.MaxPool1D(pool_size=2,strides=2,name='maxpool_1'),
        #128000 * 8 = 409600
        tf.keras.layers.Conv1D(filters=8,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d2_1'),
        tf.keras.layers.Conv1D(filters=16,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d2_2'),
        tf.keras.layers.MaxPool1D(pool_size=2,strides=2,name='maxpool_2'),
        #64000 * 16 = 163840
        tf.keras.layers.Conv1D(filters=16,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d3_1'),
        tf.keras.layers.Conv1D(filters=32,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d3_2'),
        tf.keras.layers.MaxPool1D(pool_size=2,strides=2,name='maxpool_3'),
        #32000 * 32 = 65536
        tf.keras.layers.Conv1D(filters=32,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d4_1'),
        tf.keras.layers.Conv1D(filters=64,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d4_2'),
        tf.keras.layers.MaxPool1D(pool_size=2,strides=2,name='maxpool_4'),
        #16000 * 64 = 65536
        tf.keras.layers.Conv1D(filters=64,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d5_1'),
        tf.keras.layers.Conv1D(filters=128,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d5_2'),
        tf.keras.layers.MaxPool1D(pool_size=2,strides=2,name='maxpool_5'),
        #8000 * 128 = 65536
        tf.keras.layers.Conv1D(filters=128,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d6_1'),
        tf.keras.layers.Conv1D(filters=192,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d6_2'),
        tf.keras.layers.MaxPool1D(pool_size=2,strides=2,name='maxpool_6'),
        #4000 * 192 = 49152
        tf.keras.layers.Conv1D(filters=192,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d7_1'),
        tf.keras.layers.Conv1D(filters=256,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d7_2'),
        tf.keras.layers.MaxPool1D(pool_size=2,strides=2,name='maxpool_7'),
        #2000 * 256 = 32768
        tf.keras.layers.Conv1D(filters=256,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d8_1'),
        tf.keras.layers.Conv1D(filters=512,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d8_2'),
        tf.keras.layers.MaxPool1D(pool_size=2,strides=2,name='maxpool_8'),
        #1000 * 256 = 16384
        tf.keras.layers.Conv1D(filters=512,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d9_1'),
        tf.keras.layers.Conv1D(filters=1024,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d9_2'),
        tf.keras.layers.AveragePooling1D(pool_size=5,strides=5,name='maxpool_9'),
        #200 * 256 = 8192
        tf.keras.layers.Conv1D(filters=1024,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d10_1'),
        tf.keras.layers.Conv1D(filters=1024,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d10_2'),
        tf.keras.layers.AveragePooling1D(pool_size=5,strides=5,name='maxpool_10'),
        #40 * 256 = 4096
        tf.keras.layers.Conv1D(filters=1024,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d11_1'),
        tf.keras.layers.Conv1D(filters=1024,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d11_2'),
        tf.keras.layers.AveragePooling1D(pool_size=5,strides=5,name='maxpool_11'),
        #8 * 256 = 2048
        tf.keras.layers.Conv1D(filters=1024,kernel_size=1,padding = 'same',activation=tf.nn.relu,name='conv1d12_1'),
        #tf.keras.layers.Conv1D(filters=1024,kernel_size=1,padding = 'same',activation=tf.nn.relu,name='conv1d12_2'),
        #tf.keras.layers.MaxPool1D(pool_size=2,strides=2,name='maxpool_12'),
        #4 * 256 = 1024
        tf.keras.layers.Flatten(),
        #tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(4096,activation=tf.nn.relu,name='dense_1'),
        tf.keras.layers.Dense(2048,activation=tf.nn.relu,name='dense_2'),
        tf.keras.layers.Dense(1024,activation=tf.nn.relu,name='dense_3'),
        tf.keras.layers.Dense(512,activation=tf.nn.relu,name='dense_4'),
        tf.keras.layers.Dense(256,activation=tf.nn.relu,name='dense_5'),
        tf.keras.layers.Dense(128,activation=tf.nn.relu,name='dense_6'),
        tf.keras.layers.Dense(64,activation=tf.nn.relu,name='dense_7'),
        tf.keras.layers.Dense(32,activation=tf.nn.relu,name='dense_8'),
        tf.keras.layers.Dense(16,activation=tf.nn.relu,name='dense_9'),
        #tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(5,activation=tf.nn.softmax,name='dense_final')
    ])
    sgd = tf.keras.optimizers.SGD(lr=lr,momentum=momentum,decay=decay)
    net.compile(optimizer=sgd,loss=tf.keras.losses.SparseCategoricalCrossentropy(),metrics=['accuracy'])
    net.summary()
    return net

def create_networks5(input_shape=256000,lr=0.01, momentum=0.8, decay=0.01):
    net = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(input_shape,1),name='input'),
        tf.keras.layers.Conv1D(filters=3,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d1_1'),
        tf.keras.layers.MaxPool1D(pool_size=2,strides=2,name='maxpool_1'),
        #128000 * 3
        tf.keras.layers.Conv1D(filters=8,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d2_1'),
        tf.keras.layers.Conv1D(filters=16,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d2_2'),
        tf.keras.layers.MaxPool1D(pool_size=2,strides=2,name='maxpool_2'),
        #64000 * 8
        tf.keras.layers.Conv1D(filters=16,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d3_1'),
        tf.keras.layers.Conv1D(filters=32,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d3_2'),
        tf.keras.layers.MaxPool1D(pool_size=2,strides=2,name='maxpool_3'),
        #32000 * 16
        tf.keras.layers.Conv1D(filters=32,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d4_1'),
        tf.keras.layers.Conv1D(filters=64,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d4_2'),
        tf.keras.layers.MaxPool1D(pool_size=2,strides=2,name='maxpool_4'),
        #16000 * 32
        tf.keras.layers.Conv1D(filters=64,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d5_1'),
        tf.keras.layers.Conv1D(filters=128,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d5_2'),
        tf.keras.layers.MaxPool1D(pool_size=2,strides=2,name='maxpool_5'),
        #8000 * 64
        tf.keras.layers.Conv1D(filters=128,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d6_1'),
        tf.keras.layers.Conv1D(filters=128,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d6_2'),
        tf.keras.layers.MaxPool1D(pool_size=2,strides=2,name='maxpool_6'),
        #4000 * 128
        tf.keras.layers.Conv1D(filters=128,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d7_1'),
        tf.keras.layers.Conv1D(filters=64,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d7_2'),
        tf.keras.layers.MaxPool1D(pool_size=2,strides=2,name='maxpool_7'),
        #2000 * 128
        tf.keras.layers.Conv1D(filters=64,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d8_1'),
        tf.keras.layers.Conv1D(filters=32,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d8_2'),
        tf.keras.layers.MaxPool1D(pool_size=2,strides=2,name='maxpool_8'),
        #1000 * 64
        tf.keras.layers.Conv1D(filters=32,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d9_1'),
        tf.keras.layers.Conv1D(filters=16,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d9_2'),
        tf.keras.layers.MaxPool1D(pool_size=2,strides=2,name='maxpool_9'),
        #500 * 32
        tf.keras.layers.Conv1D(filters=16,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d10_1'),
        tf.keras.layers.Conv1D(filters=8,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d10_2'),
        tf.keras.layers.MaxPool1D(pool_size=2,strides=2,name='maxpool_10'),
        #250 * 16
        tf.keras.layers.Conv1D(filters=8,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d11_1'),
        tf.keras.layers.MaxPool1D(pool_size=2,strides=2,name='maxpool_11'),
        tf.keras.layers.Conv1D(filters=8,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d12_1'),
        
        #125 * 8
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(500,activation=tf.nn.relu,name='dense_1'),
        tf.keras.layers.Dense(500,activation=tf.nn.relu,name='dense_2'),
        tf.keras.layers.Dense(250,activation=tf.nn.relu,name='dense_3'),
        tf.keras.layers.Dense(250,activation=tf.nn.relu,name='dense_4'),
        tf.keras.layers.Dense(125,activation=tf.nn.relu,name='dense_5'),
        tf.keras.layers.Dense(25,activation=tf.nn.relu,name='dense_6'),
        #tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(5,activation=tf.nn.softmax,name='dense_final')
    ])
    sgd = tf.keras.optimizers.SGD(lr=lr,momentum=momentum,decay=decay)
    net.compile(optimizer=sgd,loss=tf.keras.losses.SparseCategoricalCrossentropy(),metrics=['accuracy'])
    net.summary()
    return net

def create_networks6(input_shape=(640,3),lr=0.01, momentum=0.8, decay=0.01):
    net = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape,name='input'),
        tf.keras.layers.Conv1D(filters=8,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d1_1'),
        tf.keras.layers.MaxPool1D(pool_size=2,strides=2,name='maxpool_1'),
        #320 * 8
        tf.keras.layers.Conv1D(filters=8,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d2_1'),
        tf.keras.layers.Conv1D(filters=16,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d2_2'),
        tf.keras.layers.MaxPool1D(pool_size=2,strides=2,name='maxpool_2'),
        #160 * 16
        tf.keras.layers.Conv1D(filters=16,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d3_1'),
        tf.keras.layers.Conv1D(filters=32,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d3_2'),
        tf.keras.layers.MaxPool1D(pool_size=2,strides=2,name='maxpool_3'),
        #80 * 32
        tf.keras.layers.Conv1D(filters=32,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d4_1'),
        tf.keras.layers.Conv1D(filters=64,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d4_2'),
        tf.keras.layers.MaxPool1D(pool_size=2,strides=2,name='maxpool_4'),
        #40 * 64
        tf.keras.layers.Conv1D(filters=64,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d5_1'),
        tf.keras.layers.Conv1D(filters=64,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d5_2'),
        tf.keras.layers.MaxPool1D(pool_size=2,strides=2,name='maxpool_5'),
        #20 * 64
        tf.keras.layers.Conv1D(filters=64,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d6_1'),
        tf.keras.layers.Conv1D(filters=64,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d6_2'),
        tf.keras.layers.MaxPool1D(pool_size=2,strides=2,name='maxpool_6'),
        #10 * 64
        tf.keras.layers.Conv1D(filters=64,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d7_1'),
        tf.keras.layers.Conv1D(filters=64,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d7_2'),
        #10 * 64
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(320,activation=tf.nn.relu,name='dense_1'),
        tf.keras.layers.Dense(160,activation=tf.nn.relu,name='dense_2'),
        tf.keras.layers.Dense(80,activation=tf.nn.relu,name='dense_3'),
        tf.keras.layers.Dense(40,activation=tf.nn.relu,name='dense_4'),
        tf.keras.layers.Dense(20,activation=tf.nn.relu,name='dense_5'),
        #tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(5,activation=tf.nn.softmax,name='dense_final')
    ])
    sgd = tf.keras.optimizers.SGD(lr=lr,momentum=momentum,decay=decay)
    net.compile(optimizer=sgd,loss=tf.keras.losses.SparseCategoricalCrossentropy(),metrics=['accuracy'])
    net.summary()
    return net

def create_networks7(input_shape=(640,3),lr=0.01, momentum=0.8, decay=0.01):
    net = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape,name='input'),
        tf.keras.layers.Conv1D(filters=8,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d1_1'),
        tf.keras.layers.MaxPool1D(pool_size=2,strides=2,name='maxpool_1'),
        #320 * 8
        tf.keras.layers.Conv1D(filters=8,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d2_1'),
        tf.keras.layers.Conv1D(filters=16,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d2_2'),
        tf.keras.layers.MaxPool1D(pool_size=2,strides=2,name='maxpool_2'),
        #160 * 16
        tf.keras.layers.Conv1D(filters=16,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d3_1'),
        tf.keras.layers.Conv1D(filters=32,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d3_2'),
        tf.keras.layers.MaxPool1D(pool_size=2,strides=2,name='maxpool_3'),
        #80 * 32
        tf.keras.layers.Conv1D(filters=32,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d4_1'),
        tf.keras.layers.Conv1D(filters=64,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d4_2'),
        tf.keras.layers.MaxPool1D(pool_size=2,strides=2,name='maxpool_4'),
        #40 * 64
        tf.keras.layers.Conv1D(filters=64,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d5_1'),
        tf.keras.layers.Conv1D(filters=128,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d5_2'),
        tf.keras.layers.MaxPool1D(pool_size=2,strides=2,name='maxpool_5'),
        #20 * 64
        tf.keras.layers.Conv1D(filters=128,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d6_1'),
        tf.keras.layers.Conv1D(filters=256,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d6_2'),
        tf.keras.layers.MaxPool1D(pool_size=2,strides=2,name='maxpool_6'),
        #10 * 64
        tf.keras.layers.Conv1D(filters=256,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d7_1'),
        tf.keras.layers.Conv1D(filters=512,kernel_size=3,padding = 'same',activation=tf.nn.relu,name='conv1d7_2'),
        tf.keras.layers.GlobalAveragePooling1D(name='GlobalAveragePooling'),
        #10 * 64
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256,activation=tf.nn.relu,name='dense_1'),
        #tf.keras.layers.Dense(128,activation=tf.nn.relu,name='dense_2'),
        tf.keras.layers.Dense(64,activation=tf.nn.relu,name='dense_3'),
        #tf.keras.layers.Dense(32,activation=tf.nn.relu,name='dense_4'),
        #tf.keras.layers.Dense(20,activation=tf.nn.relu,name='dense_5'),
        #tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(5,activation=tf.nn.softmax,name='dense_final')
    ])
    sgd = tf.keras.optimizers.SGD(lr=lr,momentum=momentum,decay=decay)
    net.compile(optimizer=sgd,loss=tf.keras.losses.SparseCategoricalCrossentropy(),metrics=['accuracy'])
    net.summary()
    return net

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
    
    def feature_extractor(self, input_tensor):
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

        return x
    
    def train(self,x_train,y_train,validation_split=0.2,epochs=10000,batch_size=100,steps_per_epoch=1,validation_steps=1,snapshot = 1000,head_of_name = 'networks_liftnet_1111_',circle_num = 1):
        fit_steps = round(epochs/snapshot)
        steps = 0
        total_loss = []
        total_val_loss = []
        total_acc=[]
        total_val_acc = []
        for i in range(fit_steps):
            history = self.fit(x_train,y_train,validation_split=validation_split,epochs=snapshot,batch_size=batch_size,steps_per_epoch=steps_per_epoch,validation_steps=validation_steps)
            steps= steps+snapshot
            snapshot_save_model_name = head_of_name + str(circle_num) + '_data_the_' + str(steps)+ 'th_snapshot_with_' + str(steps_per_epoch) + '_steps_per_epoch.h5'
            snapshot_loss_name = head_of_name + str(circle_num) + '_data_the_' + str(steps)+ 'th_snapshot_with_' + str(steps_per_epoch) + '_steps_per_epoch_loss.jpg'
            snapshot_acc_name = head_of_name + str(circle_num) + '_data_the_' + str(steps)+ 'th_snapshot_with_' + str(steps_per_epoch) + '_steps_per_epoch_acc.jpg'
            self.save_weights(snapshot_save_model_name)
            print('save the '+ str(steps) + 'th snapshot model')
            loss = history.history['loss']
            val_loss = history.history['val_loss']
            acc = history.history['acc']
            val_acc = history.history['val_acc']

            total_loss.extend(loss)
            total_val_loss.extend(val_loss)
            total_acc.extend(acc)
            total_val_acc.extend(val_acc)

            plt.figure()
            newepochs = range(1,len(total_loss)+1)
            plt.plot(newepochs,total_loss,'b',label='train loss')
            plt.plot(newepochs,total_val_loss,'r',label='val_loss')
            plt.title('train and val loss')
            plt.xlabel('epochs')
            plt.ylabel('loss')
            plt.legend()
            plt.savefig(snapshot_loss_name)

            plt.figure()
            newepochs = range(1,len(total_acc)+1)
            plt.plot(newepochs,total_acc,'b',label='train acc')
            plt.plot(newepochs,total_val_acc,'r',label='val_acc')
            plt.title('train and val acc')
            plt.xlabel('epochs')
            plt.ylabel('acc')
            plt.legend()
            plt.savefig(snapshot_acc_name)

        return total_loss,total_val_loss,total_acc,total_val_acc

def create_LiftNet(class_num = 5, channel = 3, circle_num = 1,netname='LiftNet', input_shape=(640,3),lr=0.01, momentum=0.8, decay=0.01):
    model = LiftNet(class_num=class_num,input_shape=input_shape,channel=channel,netname=netname)
    sgd = tf.keras.optimizers.SGD(lr=lr,momentum=momentum,decay=decay)
    model.compile(optimizer=sgd,loss=tf.keras.losses.SparseCategoricalCrossentropy(),metrics=['accuracy'])
    model.build((None, circle_num*640, 3))
    model.summary()
    return model

class LiftNet2(tf.keras.Model):
    def __init__(self, class_num = 5, input_shape=(640,3), channel = 3, netname='LiftNet'):
        super(LiftNet2,self).__init__(name=netname)
        self.input_layer = tf.keras.layers.InputLayer(input_shape=input_shape,name = 'input')
        self.convinputs = tf.keras.layers.Conv1D(filters=channel,kernel_size=1,strides=1,padding='valid',name='convinputs')
        #split layer 320
        self.convsplit1_d = tf.keras.layers.Conv1D(filters=channel,kernel_size=2,strides=2,padding='valid',name='split1_d')
        self.convsplit1_s = tf.keras.layers.Conv1D(filters=channel,kernel_size=2,strides=2,padding='valid',name='split1_s')
        #predict layer
        self.convpredict1_1 = tf.keras.layers.Conv1D(filters=channel,kernel_size=3,strides=1,padding='same',name='predict1_1')
        self.convpredict1_2 = tf.keras.layers.Conv1D(filters=channel*5,kernel_size=1,strides=1,padding='valid',name='predict1_2',use_bias=True,kernel_regularizer=tf.keras.regularizers.l2(l=0.01),bias_initializer=tf.keras.initializers.RandomNormal(),activation=tf.nn.relu)
        #update layer
        self.convupdate1_1 = tf.keras.layers.Conv1D(filters=channel,kernel_size=3,strides=1,padding='same',name='update1_1')
        self.convupdate1_2 = tf.keras.layers.Conv1D(filters=channel*5,kernel_size=1,strides=1,padding='valid',name='update1_2',use_bias=True,kernel_regularizer=tf.keras.regularizers.l2(l=0.01),bias_initializer=tf.keras.initializers.RandomNormal(),activation=tf.nn.relu)

        #split layer 160
        self.convsplit2_d = tf.keras.layers.Conv1D(filters=channel*5,kernel_size=2,strides=2,padding='valid',name='split2_d')
        self.convsplit2_s = tf.keras.layers.Conv1D(filters=channel*5,kernel_size=2,strides=2,padding='valid',name='split2_s')
        #predict layer
        self.convpredict2_1 = tf.keras.layers.Conv1D(filters=channel*5,kernel_size=3,strides=1,padding='same',name='predict2_1')
        self.convpredict2_2 = tf.keras.layers.Conv1D(filters=channel*25,kernel_size=1,strides=1,padding='valid',name='predict2_2',use_bias=True,kernel_regularizer=tf.keras.regularizers.l2(l=0.01),bias_initializer=tf.keras.initializers.RandomNormal(),activation=tf.nn.relu)
        #update layer
        self.convupdate2_1 = tf.keras.layers.Conv1D(filters=channel*5,kernel_size=3,strides=1,padding='same',name='update2_1')
        self.convupdate2_2 = tf.keras.layers.Conv1D(filters=channel*25,kernel_size=1,strides=1,padding='valid',name='update2_2',use_bias=True,kernel_regularizer=tf.keras.regularizers.l2(l=0.01),bias_initializer=tf.keras.initializers.RandomNormal(),activation=tf.nn.relu)

        #split layer 80
        self.convsplit3_d = tf.keras.layers.Conv1D(filters=channel*25,kernel_size=2,strides=2,padding='valid',name='split3_d')
        self.convsplit3_s = tf.keras.layers.Conv1D(filters=channel*25,kernel_size=2,strides=2,padding='valid',name='split3_s')
        #predict layer
        self.convpredict3_1 = tf.keras.layers.Conv1D(filters=channel*25,kernel_size=3,strides=1,padding='same',name='predict3_1')
        self.convpredict3_2 = tf.keras.layers.Conv1D(filters=channel*125,kernel_size=1,strides=1,padding='valid',name='predict3_2',use_bias=True,kernel_regularizer=tf.keras.regularizers.l2(l=0.01),bias_initializer=tf.keras.initializers.RandomNormal(),activation=tf.nn.relu)
        #update layer
        self.convupdate3_1 = tf.keras.layers.Conv1D(filters=channel*25,kernel_size=3,strides=1,padding='same',name='update3_1')
        self.convupdate3_2 = tf.keras.layers.Conv1D(filters=channel*125,kernel_size=1,strides=1,padding='valid',name='update3_2',use_bias=True,kernel_regularizer=tf.keras.regularizers.l2(l=0.01),bias_initializer=tf.keras.initializers.RandomNormal(),activation=tf.nn.relu)
        # 40
        self.pooling = tf.keras.layers.GlobalMaxPool1D()
        self.flatten = tf.keras.layers.Flatten()
        self.FC = tf.keras.layers.Dense(class_num,kernel_regularizer=tf.keras.regularizers.l2(l=0.01),activation=tf.keras.activations.sigmoid,name='predict')
    
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
    
    def feature_extractor(self, input_tensor):
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

        return x
    
    def train(self,x_train,y_train,validation_split=0.2,epochs=10000,batch_size=100,steps_per_epoch=1,validation_steps=1,snapshot = 1000,head_of_name = 'networks_liftnet_1111_',circle_num = 1):
        fit_steps = round(epochs/snapshot)
        steps = 0
        total_loss = []
        total_val_loss = []
        total_acc=[]
        total_val_acc = []
        for i in range(fit_steps):
            history = self.fit(x_train,y_train,validation_split=validation_split,epochs=snapshot,batch_size=batch_size,steps_per_epoch=steps_per_epoch,validation_steps=validation_steps)
            steps= steps+snapshot
            snapshot_save_model_name = head_of_name + str(circle_num) + '_data_the_' + str(steps)+ 'th_snapshot_with_' + str(steps_per_epoch) + '_steps_per_epoch.h5'
            snapshot_loss_name = head_of_name + str(circle_num) + '_data_the_' + str(steps)+ 'th_snapshot_with_' + str(steps_per_epoch) + '_steps_per_epoch_loss.jpg'
            snapshot_acc_name = head_of_name + str(circle_num) + '_data_the_' + str(steps)+ 'th_snapshot_with_' + str(steps_per_epoch) + '_steps_per_epoch_acc.jpg'
            self.save_weights(snapshot_save_model_name)
            print('save the '+ str(steps) + 'th snapshot model')
            loss = history.history['loss']
            val_loss = history.history['val_loss']
            acc = history.history['acc']
            val_acc = history.history['val_acc']

            total_loss.extend(loss)
            total_val_loss.extend(val_loss)
            total_acc.extend(acc)
            total_val_acc.extend(val_acc)

            plt.figure()
            newepochs = range(1,len(total_loss)+1)
            plt.plot(newepochs,total_loss,'b',label='train loss')
            plt.plot(newepochs,total_val_loss,'r',label='val_loss')
            plt.title('train and val loss')
            plt.xlabel('epochs')
            plt.ylabel('loss')
            plt.legend()
            plt.savefig(snapshot_loss_name)

            plt.figure()
            newepochs = range(1,len(total_acc)+1)
            plt.plot(newepochs,total_acc,'b',label='train acc')
            plt.plot(newepochs,total_val_acc,'r',label='val_acc')
            plt.title('train and val acc')
            plt.xlabel('epochs')
            plt.ylabel('acc')
            plt.legend()
            plt.savefig(snapshot_acc_name)

        return total_loss,total_val_loss,total_acc,total_val_acc

def create_LiftNet2(class_num = 5, channel = 3, circle_num = 1,netname='LiftNet', input_shape=(640,3),lr=0.01, momentum=0.8, decay=0.01):
    model = LiftNet2(class_num=class_num,input_shape=input_shape,channel=channel,netname=netname)
    sgd = tf.keras.optimizers.SGD(lr=lr,momentum=momentum,decay=decay)
    model.compile(optimizer=sgd,loss=tf.keras.losses.SparseCategoricalCrossentropy(),metrics=['accuracy'])
    model.build((None, circle_num*640, 3))
    model.summary()
    return model

class Standard_LiftNet(tf.keras.Model):
    def __init__(self, class_num = 5, input_shape=(640,3), channel = 3, netname='LiftNet'):
        super(Standard_LiftNet,self).__init__(name=netname)
        self.input_layer = tf.keras.layers.InputLayer(input_shape=input_shape,name = 'input')
        self.convinputs = tf.keras.layers.Conv1D(filters=channel,kernel_size=1,strides=1,padding='valid',name='convinputs')
        #split layer 320
        #self.convsplit1_d = tf.keras.layers.Conv1D(filters=channel,kernel_size=2,strides=2,padding='valid',name='split1_d')
        #self.convsplit1_s = tf.keras.layers.Conv1D(filters=channel,kernel_size=2,strides=2,padding='valid',name='split1_s')
        #predict layer
        self.convpredict1_1 = tf.keras.layers.Conv1D(filters=channel,kernel_size=3,strides=1,padding='same',name='predict1_1')
        self.convpredict1_2 = tf.keras.layers.Conv1D(filters=channel*5,kernel_size=1,strides=1,padding='valid',name='predict1_2',use_bias=True,bias_initializer=tf.keras.initializers.RandomNormal(),activation=tf.nn.relu)
        #update layer
        self.convupdate1_1 = tf.keras.layers.Conv1D(filters=channel,kernel_size=3,strides=1,padding='same',name='update1_1')
        self.convupdate1_2 = tf.keras.layers.Conv1D(filters=channel*5,kernel_size=1,strides=1,padding='valid',name='update1_2',use_bias=True,bias_initializer=tf.keras.initializers.RandomNormal(),activation=tf.nn.relu)

        #split layer 160
        #self.convsplit2_d = tf.keras.layers.Conv1D(filters=channel*5,kernel_size=2,strides=2,padding='valid',name='split2_d')
        #self.convsplit2_s = tf.keras.layers.Conv1D(filters=channel*5,kernel_size=2,strides=2,padding='valid',name='split2_s')
        #predict layer
        self.convpredict2_1 = tf.keras.layers.Conv1D(filters=channel*5,kernel_size=3,strides=1,padding='same',name='predict2_1')
        self.convpredict2_2 = tf.keras.layers.Conv1D(filters=channel*25,kernel_size=1,strides=1,padding='valid',name='predict2_2',use_bias=True,bias_initializer=tf.keras.initializers.RandomNormal(),activation=tf.nn.relu)
        #update layer
        self.convupdate2_1 = tf.keras.layers.Conv1D(filters=channel*5,kernel_size=3,strides=1,padding='same',name='update2_1')
        self.convupdate2_2 = tf.keras.layers.Conv1D(filters=channel*25,kernel_size=1,strides=1,padding='valid',name='update2_2',use_bias=True,bias_initializer=tf.keras.initializers.RandomNormal(),activation=tf.nn.relu)

        #split layer 80
        #self.convsplit3_d = tf.keras.layers.Conv1D(filters=channel*25,kernel_size=2,strides=2,padding='valid',name='split3_d')
        #self.convsplit3_s = tf.keras.layers.Conv1D(filters=channel*25,kernel_size=2,strides=2,padding='valid',name='split3_s')
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
        length_x = x.shape[1]
        x_d = x[:,0:(length_x-1):2,:]
        x_s = x[:,1:(length_x):2,:]

        x_p = self.convpredict1_1(x_s)
        x_p = tf.math.subtract(x_d, x_p)
        x = self.convpredict1_2(x_p)

        x_u = self.convupdate1_1(x)
        x_u = tf.math.add(x_u, x_s)
        x = self.convupdate1_2(x_u)
        #2
        length_x = x.shape[1]
        x_d = x[:,0:(length_x-1):2,:]
        x_s = x[:,1:(length_x):2,:]

        x_p = self.convpredict2_1(x_s)
        x_p = tf.math.subtract(x_d, x_p)
        x = self.convpredict2_2(x_p)

        x_u = self.convupdate2_1(x)
        x_u = tf.math.add(x_u, x_s)
        x = self.convupdate2_2(x_u)
        #3
        length_x = x.shape[1]
        x_d = x[:,0:(length_x-1):2,:]
        x_s = x[:,1:(length_x):2,:]

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
    
    def feature_extractor(self, input_tensor):
        input_tensor = tf.cast(input_tensor,dtype=tf.float32)
        x = self.input_layer(input_tensor)
        x = self.convinputs(x)
        #1
        length_x = x.shape[1]
        x_d = x[:,0:(length_x-1):2,:]
        x_s = x[:,1:(length_x):2,:]

        x_p = self.convpredict1_1(x_s)
        x_p = tf.math.subtract(x_d, x_p)
        x = self.convpredict1_2(x_p)

        x_u = self.convupdate1_1(x)
        x_u = tf.math.add(x_u, x_s)
        x = self.convupdate1_2(x_u)
        #2
        length_x = x.shape[1]
        x_d = x[:,0:(length_x-1):2,:]
        x_s = x[:,1:(length_x):2,:]

        x_p = self.convpredict2_1(x_s)
        x_p = tf.math.subtract(x_d, x_p)
        x = self.convpredict2_2(x_p)

        x_u = self.convupdate2_1(x)
        x_u = tf.math.add(x_u, x_s)
        x = self.convupdate2_2(x_u)
        #3
        length_x = x.shape[1]
        x_d = x[:,0:(length_x-1):2,:]
        x_s = x[:,1:(length_x):2,:]

        x_p = self.convpredict3_1(x_s)
        x_p = tf.math.subtract(x_d, x_p)
        x = self.convpredict3_2(x_p)

        x_u = self.convupdate3_1(x)
        x_u = tf.math.add(x_u, x_s)
        x = self.convupdate3_2(x_u)
        #pooling
        x = self.pooling(x)
        x = self.flatten(x)

        return x
    
    def train(self,x_train,y_train,validation_split=0.2,epochs=10000,batch_size=100,steps_per_epoch=1,validation_steps=1,snapshot = 1000,head_of_name = 'networks_liftnet_1111_',circle_num = 1):
        fit_steps = round(epochs/snapshot)
        steps = 0
        total_loss = []
        total_val_loss = []
        total_acc=[]
        total_val_acc = []
        for i in range(fit_steps):
            history = self.fit(x_train,y_train,validation_split=validation_split,epochs=snapshot,batch_size=batch_size,steps_per_epoch=steps_per_epoch,validation_steps=validation_steps)
            steps= steps+snapshot
            snapshot_save_model_name = head_of_name + str(circle_num) + '_data_the_' + str(steps)+ 'th_snapshot_with_' + str(steps_per_epoch) + '_steps_per_epoch.h5'
            snapshot_loss_name = head_of_name + str(circle_num) + '_data_the_' + str(steps)+ 'th_snapshot_with_' + str(steps_per_epoch) + '_steps_per_epoch_loss.jpg'
            snapshot_acc_name = head_of_name + str(circle_num) + '_data_the_' + str(steps)+ 'th_snapshot_with_' + str(steps_per_epoch) + '_steps_per_epoch_acc.jpg'
            self.save_weights(snapshot_save_model_name)
            print('save the '+ str(steps) + 'th snapshot model')
            loss = history.history['loss']
            val_loss = history.history['val_loss']
            acc = history.history['acc']
            val_acc = history.history['val_acc']

            total_loss.extend(loss)
            total_val_loss.extend(val_loss)
            total_acc.extend(acc)
            total_val_acc.extend(val_acc)

            plt.figure()
            newepochs = range(1,len(total_loss)+1)
            plt.plot(newepochs,total_loss,'b',label='train loss')
            plt.plot(newepochs,total_val_loss,'r',label='val_loss')
            plt.title('train and val loss')
            plt.xlabel('epochs')
            plt.ylabel('loss')
            plt.legend()
            plt.savefig(snapshot_loss_name)

            plt.figure()
            newepochs = range(1,len(total_acc)+1)
            plt.plot(newepochs,total_acc,'b',label='train acc')
            plt.plot(newepochs,total_val_acc,'r',label='val_acc')
            plt.title('train and val acc')
            plt.xlabel('epochs')
            plt.ylabel('acc')
            plt.legend()
            plt.savefig(snapshot_acc_name)

        return total_loss,total_val_loss,total_acc,total_val_acc

def create_Standard_LiftNet(class_num = 5, channel = 3, circle_num = 1,netname='LiftNet', input_shape=(640,3),lr=0.01, momentum=0.8, decay=0.01):
    model = Standard_LiftNet(class_num=class_num,input_shape=input_shape,channel=channel,netname=netname)
    sgd = tf.keras.optimizers.SGD(lr=lr,momentum=momentum,decay=decay)
    model.compile(optimizer=sgd,loss=tf.keras.losses.SparseCategoricalCrossentropy(),metrics=['accuracy'])
    model.build((None, circle_num*640, 3))
    model.summary()
    return model

def create_Standard_LiftNet_CWRU(class_num = 6, channel = 2, cut_size=1024, netname='LiftNet', input_shape=(1024,2),lr=0.01, momentum=0.8, decay=0.01):
    model = Standard_LiftNet(class_num=class_num,input_shape=input_shape,channel=channel,netname=netname)
    sgd = tf.keras.optimizers.SGD(lr=lr,momentum=momentum,decay=decay)
    model.compile(optimizer=sgd,loss=tf.keras.losses.SparseCategoricalCrossentropy(),metrics=['accuracy'])
    model.build((None, cut_size, channel))
    model.summary()
    return model
"""
class liftnetlayer(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size = 3, name=None):
        super(liftnetlayer, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size

    def build()

"""

#---------------------------------------------------------------------------------#

def read_data(file_name):
    data = np.loadtxt(file_name,dtype=float,delimiter=',')
    return data

def read_and_savefig(loadpath,savepath,fig_size=(20,2)):
    label_name = ['/normal/', '/inner_ring/', '/outer_ring/', '/roller/', '/joint/']
    fig_num = 0
    for name in label_name:
        fnames = glob.glob(loadpath + name + '*.txt')
        for file_name in fnames:
            read_data = np.loadtxt(file_name,skiprows=16)
            f1 = plt.figure(num=fig_num,figsize=fig_size)
            plot(read_data[:,0],read_data[:,1])
            savefig_path = file_name.replace(loadpath,savepath)
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

def diliver_test_and_train_data(dataset, label, test_rate = 0.4):
    num = dataset.shape[0]
    permutation = np.random.permutation(num)
    dataset = dataset[permutation,:]
    label = label[permutation,:]
    print(dataset.shape)
    print(label.shape)
    print(label.T)
    test_num = round(test_rate * num)
    train_num = num - test_num
    train_index = random.sample(range(0,num-1),train_num)
    print('train_num:',train_num)
    data_index = np.zeros([num,1],dtype=np.int)
    data_index[train_index] = 1
    test_index = np.where(data_index==0)
    test_index=test_index[0]

    train_data = dataset[train_index,:,np.newaxis]
    test_data = dataset[test_index,:,np.newaxis]
    train_label = label[train_index]
    test_label = label[test_index]
    dataset = dataset[:, :, np.newaxis]

    return train_data, train_label, test_data, test_label, dataset, label

def diliver_test_and_train_newdata(dataset, label, test_rate = 0.4):
    num = dataset.shape[0]
    permutation = np.random.permutation(num)
    dataset = dataset[permutation,:]
    label = label[permutation,:]
    print(dataset.shape)
    print(label.shape)
    print(label.T)
    test_num = round(test_rate * num)
    train_num = num - test_num
    train_index = random.sample(range(0,num-1),train_num)
    print('train_num:',train_num)
    data_index = np.zeros([num,1],dtype=np.int)
    data_index[train_index] = 1
    test_index = np.where(data_index==0)
    test_index=test_index[0]

    train_data = dataset[train_index,:,:]
    test_data = dataset[test_index,:,:]
    train_label = label[train_index]
    test_label = label[test_index]

    return train_data, train_label, test_data, test_label, dataset, label


def count_tptnfpfn(y_true,y_predict,t=0):
    tp, tn, fp, fn = 0, 0, 0, 0
    for i in range(len(y_true)):
        if ((y_true[i,0] == t) & (y_predict[i,0] == t)):
            tp=tp+1
            continue
        if ((y_true[i,0] != t) & (y_predict[i,0] == t)):
            fp=fp+1
            continue
        if ((y_true[i,0] == t) & (y_predict[i,0] != t)):
            tn=tn+1
            continue
        if ((y_true[i,0] != t) & (y_predict[i,0] != t)):
            fn=fn+1
            continue
    return tp,tn,fp,fn

def count_tptnfpfn2(y_true,y_predict,t=0):
    tp, tn, fp, fn = 0, 0, 0, 0
    for i in range(len(y_true)):
        if ((y_true[i] == t) & (y_predict[i] == t)):
            tp=tp+1
            continue
        if ((y_true[i] != t) & (y_predict[i] == t)):
            fp=fp+1
            continue
        if ((y_true[i] == t) & (y_predict[i] != t)):
            tn=tn+1
            continue
        if ((y_true[i] != t) & (y_predict[i] != t)):
            fn=fn+1
            continue
    return tp,tn,fp,fn

def pre_to_index(pre_y):
    """
    max_p = np.max(pre_y,axis=1)
    n,_ = pre_y.shape
    index = []
    for i in range(n):
        i_max_index = np.where(pre_y[i,:]==max_p[i])[0]
        index.append(i_max_index)
    index = np.array(index)
    index = np.reshape(index,[-1,1])
    """
    index = np.argmax(pre_y,axis=1)
    index = np.reshape(index,[-1,1])
    return index

def train_and_plot(model,x_train,y_train,validation_split=0.2,epochs=1000,steps_per_epoch=1,validation_steps=1,save_model_name='trained_model.h5'):
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
    plt.savefig('loss.jpg')
    return model
    #plt.show()

def evaluate_model(pre_y, y_test, whether_save_result = 0, whether_use_CWRU_data_label= 0):
    pre_y = pre_to_index(pre_y)
    #y_test = y_test.astype(np.uint8)
    #pre_y = pre_y.astype(np.uint8)
    #print('truth:')
    #print(y_test.T)
    #print('predict:')
    #print(pre_y.T)
    result_list = []
    label_name = ['normal', 'inner_ring', 'outer_ring', 'ball', 'joint']
    if whether_use_CWRU_data_label==1:
        label_name = ['normal', 'ball', 'inner_ring', 'outer_ring_3', 'outer_ring_6', 'outer_ring_12']
    for i in range(len(label_name)):
        tp, tn, fp, fn = count_tptnfpfn2(y_test,pre_y,i)
        print(label_name[i] + ' result:')
        result_list.append(label_name[i] + ' result:\n')
        print('tp:',tp,' tn:',tn,' fp:',fp,' fn:',fn)
        result_list.append('tp:'+str(tp)+' tn:'+str(tn)+' fp:'+str(fp)+' fn:'+str(fn) + '\n')
        if (tp+tn)==0:
            rec = 0
            print('tp/(tp+tn): 0')
        else:
            rec = tp/(tp+tn)
            print('tp/(tp+tn): ', tp/(tp+tn))

        if (tp+fp)==0:
            acc = 0 
            print('tp/(tp+fp): 0')
        else:
            acc = tp/(tp+fp)
            print('tp/(tp+fp): ', tp/(tp+fp))
        result_list.append('tp/(tp+tn): '+str(rec)+'\n')
        result_list.append('tp/(tp+fp): '+str(acc)+'\n')
    if whether_save_result==1:
        return result_list
        
def evaluate_model2(pre_y, y_test):

    #y_test = y_test.astype(np.uint8)
    #pre_y = pre_y.astype(np.uint8)
    #print('truth:')
    #print(y_test.T)
    #print('predict:')
    #print(pre_y.T)

    for i in range(5):
        tp, tn, fp, fn = count_tptnfpfn2(y_test,pre_y,i)
        label_name = ['normal', 'inner_ring', 'outer_ring', 'roller', 'joint']
        print(label_name[i] + ' result:')
        print('tp:',tp,' tn:',tn,' fp:',fp,' fn:',fn)
        if (tp+tn)==0:
            print('tp/(tp+tn): 0')
        else:
            print('tp/(tp+tn): ', tp/(tp+tn))

        if (tp+fp)==0:
            print('tp/(tp+fp): 0')
        else:
            print('tp/(tp+fp): ', tp/(tp+fp))

def evaluate_model3(pre_y, y_test, whether_save_result = 0, whether_use_CWRU_data_label= 0):
    #pre_y = pre_to_index(pre_y)
    #y_test = y_test.astype(np.uint8)
    #pre_y = pre_y.astype(np.uint8)
    #print('truth:')
    #print(y_test.T)
    #print('predict:')
    #print(pre_y.T)
    result_list = []
    label_name = ['normal', 'inner_ring', 'outer_ring', 'ball', 'joint']
    if whether_use_CWRU_data_label==1:
        label_name = ['normal', 'ball', 'inner_ring', 'outer_ring_3', 'outer_ring_6', 'outer_ring_12']
    for i in range(len(label_name)):
        tp, tn, fp, fn = count_tptnfpfn2(y_test,pre_y,i)
        print(label_name[i] + ' result:')
        result_list.append(label_name[i] + ' result:\n')
        print('tp:',tp,' tn:',tn,' fp:',fp,' fn:',fn)
        result_list.append('tp:'+str(tp)+' tn:'+str(tn)+' fp:'+str(fp)+' fn:'+str(fn) + '\n')
        if (tp+tn)==0:
            rec = 0
            print('tp/(tp+tn): 0')
        else:
            rec = tp/(tp+tn)
            print('tp/(tp+tn): ', tp/(tp+tn))

        if (tp+fp)==0:
            acc = 0 
            print('tp/(tp+fp): 0')
        else:
            acc = tp/(tp+fp)
            print('tp/(tp+fp): ', tp/(tp+fp))
        result_list.append('tp/(tp+tn): '+str(rec)+'\n')
        result_list.append('tp/(tp+fp): '+str(acc)+'\n')
    if whether_save_result==1:
        return result_list


def plot_loss_map(loss,val_loss, loss_map_name='loss_map.png'):
    plt.figure()
    newepochs = range(1,len(loss)+1)
    plt.plot(newepochs,loss,'b',label='train loss')
    plt.plot(newepochs,val_loss,'r',label='val_loss')
    plt.title('train and val loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(loss_map_name)
    plt.cla()
    plt.close("all")

def plot_acc_map(acc,val_acc,acc_map_name='acc_map.png'):
    plt.figure()
    newepochs = range(1,len(acc)+1)
    plt.plot(newepochs,acc,'b',label='train acc')
    plt.plot(newepochs,val_acc,'r',label='val_acc')
    plt.title('train and val acc')
    plt.xlabel('epochs')
    plt.ylabel('acc')
    plt.legend()
    plt.savefig(acc_map_name)
    plt.cla()
    plt.close("all")