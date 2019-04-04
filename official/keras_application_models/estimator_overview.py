import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
import json

print('tf.__version: ', tf.__version__)
print('keras.__version: ', keras.__version__)


#data = np.random.random((1000, 32))
#labels = np.random.random((1000, 10))

def input_fn():
  dataset = tf.data.Dataset.from_tensors(({"feats":[1.]}, [1.]))
  return dataset.repeat(1000).batch(10)

mirrored_strategy = tf.distribute.MirroredStrategy()
config = tf.estimator.RunConfig(
    train_distribute=mirrored_strategy, eval_distribute=mirrored_strategy)
regressor = tf.estimator.LinearRegressor(
    feature_columns=[tf.feature_column.numeric_column('feats')],
    optimizer='SGD',
    config=config)

# regressor.train(input_fn=input_fn, steps=100)
# regressor.evaluate(input_fn=input_fn, steps=10)
