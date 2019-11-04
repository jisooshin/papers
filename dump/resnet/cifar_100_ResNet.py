# coding: utf-8
import os
import sys
import math

import nni

import numpy as np
import tensorflow as tf
from tensorflow import keras

from datetime import datetime

base_path = os.environ['HOME'] + '/ResNet'
if os.path.isdir(base_path):
  pass
else:
  os.mkdir(base_path)

data = tf.keras.datasets.cifar100.load_data()

train = data[0]
test = data[1]

train_image, train_label = train[0].astype(np.float32), train[1]
test_image, test_label= test[0].astype(np.float32), test[1]

train_label = np.reshape(train_label, newshape=[-1])
test_label = np.reshape(test_label, newshape=[-1])

# Generate Dataset obj
dataset_obj = tf.data.Dataset.from_tensors(
    {'image': train_image, 'label': train_label})
dataset_obj = dataset_obj.shuffle(50000)
dataset_obj = dataset_obj.unbatch()

# split train-validation dataset
train_dataset = dataset_obj.take(40000)
val_dataset = dataset_obj.skip(40000).take(10000)

test_dataset = tf.data.Dataset.from_tensors(
  {'image': test_image, 'label': test_label})
test_dataset = test_dataset.shuffle(10000).unbatch()

def _preprocessing(dataset, train_mode):
  """
  While train steps, image will be padded random crop and filped(horizontaly)
  And entire steps, per-pixel mean subtracted will be required.
  Args:
    dataset: 'tf.data.Dataset'
    train_mode: 'bool'
  Returns:
    'tf.data.Dataset'
  """
  if train_mode:
    image = dataset['image']
    pad = tf.constant([[2, 2], [2, 2], [0, 0]])
    image = tf.pad(
      tensor=image, paddings=pad)
    image = tf.image.random_crop(
      value=image, size=[32, 32, 3])
    image = tf.image.random_flip_left_right(image=image)
  else:
    image = dataset['image']

  label = dataset['label']
  image = tf.math.subtract(
      x=image,
      y=tf.reshape(
          tf.math.reduce_mean(image, axis=2),
          shape=[32, 32, 1]))
  return (image, label)

train_dataset = train_dataset.map(
  lambda x: _preprocessing(x, train_mode=True))
val_dataset = val_dataset.map(
  lambda x: _preprocessing(x, train_mode=False))
test_dataset = test_dataset.map(
  lambda x: _preprocessing(x, train_mode=False))


train_dataset = train_dataset.repeat()
val_dataset = val_dataset.repeat()

# Experiment Parameter
WEIGHT_DECAY_COEFFICIENT = [1e-3, 1e-2, 1e-1, 1]
NUMBER_OF_LAYERS = 50

def residual_block(data, name, weight_decay):
  """
  "bottleneck" building block
  Args:
    data: 'tf.Tensor' generated passing through keras layers
    num_filters: 'int' # of feature map(activation map)
    name: 'str'
  Returns:
    'tf.Tensor' keras layers
  """
  with tf.name_scope(name) as scope:
    identity_data = data

    data = keras.layers.Conv2D(
        filters=64, kernel_size=[1, 1], strides=1, padding='same',
        kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(data)
    data = keras.layers.BatchNormalization()(data)
    data = keras.layers.ReLU()(data)

    data = keras.layers.Conv2D(
        filters=64, kernel_size=[3, 3], strides=1, padding='same',
        kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(data)
    data = keras.layers.BatchNormalization()(data)
    data = keras.layers.ReLU()(data)

    data = keras.layers.Conv2D(
        filters=256, kernel_size=[1, 1], strides=1, padding='same',
        kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(data)
    data = keras.layers.BatchNormalization()(data)

    data = keras.layers.Add()([data, identity_data])

    data = keras.layers.ReLU()(data)
    return data


wdc = 1e-2

# Ridge Regularization
inputs = keras.Input(shape=[32, 32, 3], name='input_image')

with tf.name_scope("First_block") as scope:
  x = keras.layers.Conv2D(
    filters=256, kernel_size=[2, 2], strides=2,
    kernel_regularizer=tf.keras.regularizers.l2(wdc))(inputs)
  x = keras.layers.BatchNormalization()(x)
  x = keras.layers.ReLU()(x)
  x = keras.layers.MaxPool2D()(x)


for layers in range(NUMBER_OF_LAYERS):
  x = residual_block(data=x, name="RB{}".format(layers), weight_decay=wdc)

with tf.name_scope("GlobalAveragePooling") as scope:
  x = keras.layers.GlobalAveragePooling2D()(x)
  outputs = keras.layers.Dense(100, activation='softmax',
      kernel_regularizer=tf.keras.regularizers.l2(wdc))(x)

model = keras.Model(
    inputs, outputs,
    name="{}layer_{}weight_decay_{}".format(
      NUMBER_OF_LAYERS, wdc,
      datetime.strftime(datetime.now(), "%Y%m%d-%H%M")))

# Callbacks
callbacks_list = [
  keras.callbacks.EarlyStopping(
    monitor='val_sparse_categorical_accuracy', min_delta=1e-2,
    patience=1000, verbose=1),
  keras.callbacks.ModelCheckpoint(
      filepath=base_path+'/resnet/ckpts/{}.h5'.format(model.name),
      verbos=1, save_best_only=True
  ),
  keras.callbacks.TensorBoard(
      log_dir=base_path+'/logs/{}/'.format(model.name),
      histogram_freq=10,
      update_freq='epoch')]

model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=[keras.metrics.SparseCategoricalAccuracy()])

model.fit(
  x=train_dataset.shuffle(1000).batch(100),
  epochs=10000,
  validation_data=val_dataset.shuffle(1000).batch(100),
  validation_freq=1,
  callbacks=callbacks_list,
  steps_per_epoch=100,
  validation_steps=100)


model.evaluate(
  x=test_dataset.batch(1000))
