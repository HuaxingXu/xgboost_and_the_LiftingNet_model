import tensorflow as tf
import numpy as np
import math
import glob
import matplotlib

tensor = tf.ones([1000,1280,3])

tensor_shape = tensor.shape
print(tensor_shape[1])
index = range(0,tensor_shape[1],2)
print(len(index))
print(index)
#p = tensor[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19],:]
p = tensor[:,0:(tensor_shape[1]-1):2,:]
print(p.shape)


