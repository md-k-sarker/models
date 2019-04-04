import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
import json

print('tf.__version: ', tf.__version__)
print('keras.__version: ', keras.__version__)

dataset = tf.data.Dataset.from_tensors(([1.], [1.])).repeat(100).shuffle(buffer_size=10000).batch(10)

print("before ds_strategy declared \n\n\n")
# single node
ds_strategy=tf.distribute.MirroredStrategy()
# cross_device_ops=tf.distribute.HierarchicalCopyAllReduce()
# multinode
#ds_strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
print("ds_strategy declared \n\n\n")

with ds_strategy.scope():
    print(" inside ds_strategy \n\n\n")
    model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(1,))])
    model.compile(loss='mse', optimizer='sgd')
    print("model compiled \n\n\n")
    model.fit(dataset,epochs=20)
