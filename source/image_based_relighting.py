#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 11:39:54 2017

@author: alexco
"""

from __future__ import print_function

import os
import threading
os.environ["CUDA_VISIBLE_DEVICES"] = str(0)


import numpy as np

from tensorflow.python import keras
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.callbacks import ReduceLROnPlateau
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras import regularizers

import matplotlib.pyplot as plt
import image_data


# image utilities
def exposure_compensate(rgb_color, exposure, gamma):
    """ exposure multiply and gamma correction """
    rgb_color -= np.min(rgb_color)
    rgb_color /= np.max(rgb_color)
    rgb_color[rgb_color < 0.0] = 0.0
    m = np.power(2.0, exposure)
    g = 1.0 / gamma
    rgb = np.power(rgb_color * m, g)
    return rgb


def exposure_compensate4_22(rgb_color):
    """ docstring """
    return exposure_compensate(rgb_color, 4, 2.2)


def exposure_compensate5_22(rgb_color):
    """ docstring """
    return exposure_compensate(rgb_color, 5.0, 2.2)


# relative error use in Peiran Ren, Yue Dong, Stephen Lin, Xin Tong, and Baining Guo.
# 2015. Image based relighting using neural networks.
# ACM Trans. Graph. 34, 4, Article 111 (July 2015), 12 pages. DOI: https://doi.org/10.1145/2766899

# not quite WAPE (weighted absolute percentage error).

def relative_err_numpy(y_true, y_pred):
    """sum of absolute differences"""

    sum_sqr_diff = np.sum(np.square(y_true - y_pred))
    sum_sqr_val = np.sum(np.clip(np.square(y_true), K.epsilon(), None))
    return np.sqrt(sum_sqr_diff / sum_sqr_val)


class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            return self.it.next()

    def __next__(self):
        with self.lock:
            return self.it.__next__()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g


def create_per_pixel_model(input_size, output_size,
                           n_hidden=600, n_layers=4,
                           activation='relu',
                           dropout=0.0,
                           bn=False,
                           leaky=False):

    ar = regularizers.l2(1e-6)
    kr = None # regularizers.l2(1e-6)
    br = None # regularizers.l2(1e-6)

    def create_block(source):
        block = Dense(n_hidden, activation=activation,
                      activity_regularizer=ar,
                      kernel_regularizer=kr,
                      bias_regularizer=br)(source)
        if leaky:
            # set activation to None
            block = keras.layers.advanced_activations.PReLU()(block)
        if bn:
            block = BatchNormalization()(block)

        if dropout > 0.0:
            block = Dropout(dropout)(block)
        return block

    def model():
        inputs = Input(shape=(input_size,))

        b = Dense(n_hidden, activation=activation,
                  activity_regularizer=ar,
                  kernel_regularizer=kr,
                  bias_regularizer=br)(inputs)


        for i in range(n_layers):
            b = create_block(b)


        output = Dense(output_size, activation='relu')(b)
        model = Model(inputs=[inputs], outputs=[output])

        return model

    return model()


def update_history(history, update):
    for k in history.history.keys():
        if k in update.history:
            history.history[k].extend(update.history[k])
    return history


@threadsafe_generator
def data_generator(data, idx, batch_size, b_train=True):

    x_train = data.train.x_values
    y_train = data.train.y_values
    x_val = data.validation.x_values
    y_val = data.validation.y_values

    batch_size = np.min([batch_size, x_train.shape[0]])

    num_train_examples = x_train.shape[0]
    steps_per_epoch = int(num_train_examples / batch_size)

    if b_train:
        x_values = x_train
        y_truth = y_train
    else:
        x_values = x_val
        y_truth = y_val

    num_examples = x_values.shape[0]
    if num_examples < batch_size:
        batch_size = num_examples

    index_array = np.arange(num_examples)

    while 1:
        np.random.shuffle(index_array)
        start = 0
        for i in range(steps_per_epoch):
            if start + batch_size > num_examples:
                start = 0
                np.random.shuffle(index_array)
            end = start + batch_size
            idx = index_array[start:end]
            x = x_values[idx, :]
            y = y_truth[idx, :]

            start += batch_size
            yield (x, y)


def train_per_pixel_model(data, epochs=1000, batch_size=5000, batch_reductions=4, initial_lrate=0.001):

    num_train_examples = data.train.x_values.shape[0]
    num_validation_examples = data.validation.x_values.shape[0]

    batch_size = np.min([batch_size, num_train_examples])

    model = create_per_pixel_model(data.train.x_values.shape[1], data.train.y_values.shape[1])

    def relative_err(y_true, y_pred):
        sum_err_sqr = K.sum(K.square(y_true - y_pred))
        sum_val_sqr = K.clip(K.sum(K.square(y_true)), K.epsilon(), None)
        return K.sqrt(sum_err_sqr / sum_val_sqr)

    def luminance(rgb):
        return K.sqrt(0.299 * K.square(rgb[:,0]) + 0.587 * K.square(rgb[:,1]) + 0.114 * K.square(rgb[:,2]))

    def my_loss(y_true, y_pred):

        lum_true = luminance(y_true)
        lum_pred = luminance(y_pred)
        mse_rgb = K.mean(K.square(y_pred - y_true), axis=-1)
        mse_lum = K.mean(K.square(lum_pred - lum_true), axis=-1)
        return mse_rgb + mse_lum + (relative_err(y_true, y_pred) * 0.01)

    opt = keras.optimizers.Adam(lr=initial_lrate)
    model.compile(optimizer=opt,
                  loss=my_loss,
                  metrics=['mse', relative_err])
    model.summary()

    min_lr = 0.0000001
    callback = [ReduceLROnPlateau(monitor='loss',
                                  factor=0.5,
                                  patience=2,
                                  min_lr=min_lr,
                                  verbose=1)]

    initial_epoch = 0

    b = batch_size
    min_batch = batch_size - 1
    if batch_reductions > 0:
        batch_reductions = 2 ** (int(batch_reductions))
        min_batch = np.max([50, int(batch_size / batch_reductions)])

    loss = 1e10
    history = None
    while b > min_batch and loss > 1e-5:
        print("batch size: " + str(b))
        steps_per_epoch = int(num_train_examples / b)
        validation_steps = int(num_validation_examples / b) + 1

        generator = data_generator(data, None, b)
        validation = data_generator(data, None, b, False)

        h = model.fit_generator(generator, steps_per_epoch=steps_per_epoch,
                                callbacks=callback,
                                initial_epoch=initial_epoch,
                                epochs=epochs + initial_epoch,
                                validation_data=validation,
                                validation_steps=validation_steps,
                                workers=2)

        b = int(b / 2.0)
        initial_epoch += len(model.history.history['loss'])
        loss = model.history.history['val_loss'][-1]
        if history is None:
            history = h
        else:
            history = update_history(history, h)

        callback = [EarlyStopping(monitor='val_loss', patience=3, mode='auto', verbose=1),
                    ReduceLROnPlateau(monitor='loss', factor=0.5, patience=2, min_lr=min_lr, verbose=1)]

    return model, history






class IBR_Model(object):

    def __init__(self, data):
        self._data = data
        self._model = None
        self._history = None

    def train(self, fn=train_per_pixel_model, epochs=10, batch_size=100000):
        self._model, self._history = fn(self._data, epochs, batch_size)
        
    def reconstruct_image(self, dataset, idx, exposure=1, gamma=1, display=True):

        if self._model is None:
            return None
        
        if self._data is None:
            return None
        
        x_values, y_truth = dataset.get_data_slice(idx)
    

        y_pred = self._model.predict(x_values)
            
        if not display:
            return y_pred
    
    
        dataset.plot_prediction(y_pred, title="Prediction", exposure=exposure, gamma=gamma)
        dataset.plot_y_values(idx, exposure=exposure, gamma=gamma)
    
        mse = np.mean(np.square(y_pred - y_truth))
        maxe = np.max(np.abs(y_pred - y_truth))
        mape = np.mean(np.square(y_pred - y_truth) / (np.abs(y_truth) + 1e-6))
        mape_image = np.abs((y_truth - y_pred) / np.clip(np.abs(y_truth), K.epsilon(), None))
        mape_image = np.log(mape_image + 1.0)
        max_mape_pixel = np.max(mape_image)
        mape_image /= max_mape_pixel
    
        relative_err = relative_err_numpy(y_truth,y_pred)
    
        mse_image = np.square(y_truth - y_pred)
        mse_image = np.log(mse_image + 1.0)
        max_mse_pixel = np.max(mse_image)
        mse_image /= max_mse_pixel
    
        dataset.plot_prediction(mse_image, title="LOG(MSE)", exposure=0, gamma=1)
        dataset.plot_prediction(mape_image, title="LOG(MAPE)", exposure=0, gamma=1)
    
        difference = np.abs(y_pred - y_truth)
        maxval = np.max(np.max(difference)) + 1e-10
        difference /= maxval
    
        # convert to image space 0 - 1
        f = dataset.to_image(y_pred, exposure, gamma)
        y_truth = dataset.to_image(y_truth, exposure, gamma)
    
        difference = dataset.to_image(difference, exposure=0, gamma=1.0)
    
        fig, axarr = plt.subplots(1, 3)
        fig.set_size_inches(15, 7)
        axarr[0].imshow(f, interpolation='nearest')
        axarr[0].set_title('Predicted:\nMSE ' + str(mse) +
                           str("\nMAX abs(E) ") + str(maxe) +
                           str("\nMAPE ") + str(mape) +
                           str("\nRelative ") + str(relative_err))
    
        axarr[1].imshow(y_truth, interpolation='nearest')
        axarr[1].set_title('Original')
    
        axarr[2].imshow(difference, interpolation='nearest')
        axarr[2].set_title('Difference\n' + 'x' + str(1.0 / maxval))
    
        return y_pred

    def load_model(self, filename="keras_model"):

        json_model = None

        with open(filename + ".json", "r") as json_file:
            json_model = json_file.read()

        self._model = keras.models.model_from_json(json_model)
        self._model.load_weights(filename + ".hdf5")

    def save_model(self, filename="keras_model"):
        self._model.save(filename + ".hdf5")
        model_json = self._model.to_json()
        with open(filename + ".json", "wt") as json_file:
            json_file.write(model_json)


def run(data_dir=os.path.join("../data", "waldorf")):
    data = image_data.read_data(data_dir=data_dir,transform=exposure_compensate4_22, train=0.5, validation=0.1, test=0.4)
    IBR = IBR_Model(data)
    IBR.train(fn=train_per_pixel_model, epochs=5, batch_size=50000)
    return IBR


