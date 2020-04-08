import tensorflow as tf
import numpy as np
import math
import glob
import matplotlib.pyplot as plt
import mylib as ml2
import random
from keras.utils import plot_model

test_rate = 0.2
epochs = 100
lr=0.01
momentum=0.8
decay=0.01
input_shape = 19
if input_shape == 11:
    add_orgin_data = False
else:
    add_orgin_data = True

batch = 100

validation_split=0.2
steps_per_epoch=2
validation_steps=1


model = ml2.create_networks3(input_shape=input_shape,lr=lr,momentum=momentum,decay=decay)
plot_model(model,to_file='model.png',show_shapes= True)