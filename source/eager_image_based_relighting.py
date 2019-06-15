# Copyright 2019 Alex Colburn
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Implements basic file network training, metrics, and visualization
"""

from __future__ import absolute_import, division, print_function

import datetime
import os
import threading

os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

import tensorflow as tf
import tensorflow.contrib.eager as tfe
import tensorflow.keras.backend as K
from tensorflow.python.keras.utils.generic_utils import Progbar

#tf.enable_eager_execution()

# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
# sess = tf.Session(config=tf.ConfigProto(
#     allow_soft_placement=True, log_device_placement=True))

import numpy as np
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


def reconstruct_stats(model, ibr_dataset, idx, exposure=1, gamma=1):
    x_values, y_truth = ibr_dataset.get_data_slice(idx)
    x_values = x_values.astype(np.float32)
   
    y_pred = model(tf.convert_to_tensor(x_values), training=False).numpy()
    y_truth = y_truth.astype(np.float32)

    ibr_dataset.plot_prediction(y_pred, title="Prediction", exposure=exposure, gamma=gamma)
    ibr_dataset.plot_y_values(idx, exposure=exposure, gamma=gamma)

    mse = np.mean(np.square(y_pred - y_truth))
    maxe = np.max(np.abs(y_pred - y_truth))
    mape = np.mean(np.square(y_pred - y_truth) / (np.abs(y_truth) + 1e-6))
    mape_image = np.abs((y_truth - y_pred) / np.clip(np.abs(y_truth), K.epsilon(), None))
    mape_image = np.log(mape_image + 1.0)
    max_mape_pixel = np.max(mape_image)
    mape_image /= max_mape_pixel

    relative_err = relative_err_numpy(y_truth, y_pred)

    mse_image = np.square(y_truth - y_pred)
    mse_image = np.log(mse_image + 1.0)
    max_mse_pixel = np.max(mse_image)
    mse_image /= max_mse_pixel

    ibr_dataset.plot_prediction(mse_image, title="LOG(MSE)", exposure=0, gamma=1)
    ibr_dataset.plot_prediction(mape_image, title="LOG(MAPE)", exposure=0, gamma=1)

    difference = np.abs(y_pred - y_truth)
    maxval = np.max(np.max(difference)) + 1e-10
    difference /= maxval

    # convert to image space 0 - 1
    f = ibr_dataset.to_image(y_pred, exposure, gamma)
    y_truth = ibr_dataset.to_image(y_truth, exposure, gamma)

    difference = ibr_dataset.to_image(difference, exposure=0, gamma=1.0)

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


def reconstruct_conv_stats(model, ibr_dataset, idx, exposure=1, gamma=1):
    
    
    w, h = ibr_dataset.shape[:2]
    
    x_depth =  ibr_dataset.get_data_slice(0)[0].shape[-1]
    y_depth =  ibr_dataset.get_data_slice(0)[1].shape[-1]
    
    x_values, y_truth = ibr_dataset.get_data_slice(idx)
    x_values = x_values.reshape((1,w,h,x_depth)).transpose((0, 2, 1, 3)).astype(np.float32)
   
                
    y_pred = model(tf.convert_to_tensor(x_values), training=False).numpy()
    
    y_pred = y_pred.transpose((0, 2, 1, 3)).reshape(y_truth.shape)

    ibr_dataset.plot_prediction(y_pred, title="Prediction", exposure=exposure, gamma=gamma)
    ibr_dataset.plot_y_values(idx, exposure=exposure, gamma=gamma)

    mse = np.mean(np.square(y_pred - y_truth))
    maxe = np.max(np.abs(y_pred - y_truth))
    mape = np.mean(np.square(y_pred - y_truth) / (np.abs(y_truth) + 1e-6))
    mape_image = np.abs((y_truth - y_pred) / np.clip(np.abs(y_truth), K.epsilon(), None))
    mape_image = np.log(mape_image + 1.0)
    max_mape_pixel = np.max(mape_image)
    mape_image /= max_mape_pixel

    relative_err = relative_err_numpy(y_truth, y_pred)

    mse_image = np.square(y_truth - y_pred)
    mse_image = np.log(mse_image + 1.0)
    max_mse_pixel = np.max(mse_image)
    mse_image /= max_mse_pixel

    ibr_dataset.plot_prediction(mse_image, title="LOG(MSE)", exposure=0, gamma=1)
    ibr_dataset.plot_prediction(mape_image, title="LOG(MAPE)", exposure=0, gamma=1)

    difference = np.abs(y_pred - y_truth)
    maxval = np.max(np.max(difference)) + 1e-10
    difference /= maxval

    # convert to image space 0 - 1
    f = ibr_dataset.to_image(y_pred, exposure, gamma)
    y_truth = ibr_dataset.to_image(y_truth, exposure, gamma).astype(np.float32)

    difference = ibr_dataset.to_image(difference, exposure=0, gamma=1.0)

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



# data generators
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


@threadsafe_generator
def data_generator(data, batch_size, b_train=True):
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

    while 1:

        start = 0
        for i in range(steps_per_epoch):
            if start + batch_size > num_examples:
                start = 0

            end = start + batch_size
            x = x_values[start:end, :]
            y = y_truth[start:end, :]

            start += batch_size
            yield (x, y)



# the generator needs to convert to float32 - tensorflow
# float16 to Tensor are really slow at v1.13.1
def create_generator(data, batch_size, b_train=True):
    def create_generator():
        return data_generator(data, batch_size, b_train)

    return create_generator


def create_voxel_tfdataset(dataset, batch_size=10000, b_train=True):
    
    x_values = dataset.x_values
    y_values = dataset.y_values

    num_examples = x_values.shape[0]
    
    if num_examples < batch_size:
        batch_size = num_examples
    
    def f(i):

        def g(i):
            x = x_values[i,:].astype(np.float32)
            y = y_values[i,:].astype(np.float32)
            
            return x, y
    
        return tf.py_func(func=g, inp=[i], Tout=[tf.float32, tf.float32])
         
    tf_dataset = tf.data.Dataset.range(num_examples)
    tf_dataset = tf_dataset.shuffle(buffer_size=num_examples).batch(batch_size)
    tf_dataset = tf_dataset.prefetch(batch_size*10)
    
    tf_dataset = tf_dataset.map(map_func=f)
    return tf_dataset



def create_image_tfdataset(data, batch_size=1, b_train=True):
    
    dataset = data.train
    if not b_train:
        dataset = data.validation

    num_examples = dataset.num_source_images
    
    w, h = dataset.shape[:2]
    x_values, y_truth = dataset.get_data_slice(0)
    
    x_depth =  data.train.get_data_slice(0)[0].shape[-1]
    y_depth =  data.train.get_data_slice(0)[1].shape[-1]
    
    def f(i):

        def g(i):
            
            #print(len(i))
            x = np.zeros((batch_size, h, w, x_depth), dtype=np.float32)
            y = np.zeros((batch_size, h, w, y_depth), dtype=np.float32)
            
            #print(x.shape, y.shape)
            
            for b, idx in enumerate(i):
                #print(".")
                #print(b_train, idx)
                #print(".")
                x_values, y_values = dataset.get_data_slice(idx)
                x[b,:,:,:] = x_values.reshape((w,h,x_depth)).transpose((1,0,2))
                y[b,:,:,:] = y_values.reshape((w,h,y_depth)).transpose((1,0,2))
            
            return x, y
        
        return tf.py_func(func=g, inp=[i], Tout=[tf.float32, tf.float32])
         
    x = np.arange(num_examples)
    tf_dataset = tf.data.Dataset.from_tensor_slices(x)
    tf_dataset = tf_dataset.shuffle(buffer_size=num_examples*10).batch(batch_size)
    tf_dataset = tf_dataset.prefetch(10)
    
    tf_dataset = tf_dataset.map(map_func=f)
    return tf_dataset


# tensorflow code
def variable_summaries(var, name):
    with tf.name_scope(name):
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)


def cast32(inputs):
    return tf.cast(inputs, tf.float32)


def add(inputs):
    return tf.add(inputs,0.1)


def create_model():
    model = tf.keras.Sequential([
        # tf.keras.layers.Lambda(cast32),
        tf.keras.layers.Dense(600, activation='relu', name='Dense_1', kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform'),
        tf.keras.layers.Dense(600, activation='relu', name='Dense_2', kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform'),
        tf.keras.layers.Dense(600, activation='relu', name='Dense_3', kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform'),
        tf.keras.layers.Dense(600, activation='relu', name='Dense_4', kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform'),
        tf.keras.layers.Dense(600, activation='relu', name='Dense_5', kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform'),
        tf.keras.layers.Dense(3, name='RGB', kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform'),
    ])
    return model


def create_conv_model(filters=600, kernel_size=1, input_shape=(464, 696, 7)):
    
    
    ar = None # tf.keras.regularizers.l2(1e-6)
    kr = None # tf.keras.regularizers.l2(1e-6)
    br = None # tf.keras.regularizers.l2(1e-6)
    
    activation = tf.nn.leaky_relu
    dropout_rate = 0.2
    init = 'he_uniform'  #'glorot_uniform'
    #activation='relu'
    
    def conv_2d_layer_input(name):
        return tf.keras.layers.Conv2D(input_shape=input_shape,
                                      filters=filters, 
                                      kernel_size=kernel_size, 
                                      activation=activation, 
                                      name=name, 
                                      padding='same', 
                                      #activity_regularizer=ar,
                                      #kernel_regularizer=kr,
                                      #bias_regularizer=br,
                                      kernel_initializer=init, 
                                      bias_initializer=init)

    def conv_2d_layer(name, input_shape=None):
        return tf.keras.layers.Conv2D(filters=filters, 
                                      kernel_size=kernel_size, 
                                      activation=activation, 
                                      name=name, 
                                      padding='same', 
                                      #activity_regularizer=ar,
                                      #kernel_regularizer=kr,
                                      #bias_regularizer=br,
                                      kernel_initializer=init, 
                                      bias_initializer=init)
        
    def conv_2d_layer_rgb(name, filters=3, input_shape=None):
        return tf.keras.layers.Conv2D(filters=filters, 
                                      kernel_size=kernel_size, 
                                      activation=activation, 
                                      name=name, 
                                      padding='same', 
                                      #activity_regularizer=ar,
                                      #kernel_regularizer=kr,
                                      #bias_regularizer=br,
                                      kernel_initializer=init, 
                                      bias_initializer=init)
        
    model = tf.keras.Sequential([
        
        conv_2d_layer_input(name='Conv2D_1_'+str(kernel_size)),
        conv_2d_layer(name='Conv2D_2_'+str(kernel_size)),
        tf.keras.layers.Dropout(dropout_rate),
        conv_2d_layer(name='Conv2D_3_'+str(kernel_size)),
        tf.keras.layers.Dropout(dropout_rate),
        conv_2d_layer(name='Conv2D_4_'+str(kernel_size)),
        tf.keras.layers.Dropout(dropout_rate),
        conv_2d_layer(name='Conv2D_5_'+str(kernel_size)),
        tf.keras.layers.Dropout(dropout_rate),
        conv_2d_layer_rgb(name='RGB')
    ])
    return model


def save_model(model, filename):
    saver = tf.train.Saver()
    with tf.Session() as sess:
        save_path = saver.save(sess, filename+".ckpt")
        print("Model saved in path: %s" % save_path)   


def save_kmodel(model, filename="keras_model"):
        model.save(filename + ".hdf5")
        model_json = model.to_json()
        with open(filename + ".json", "wt") as json_file:
            json_file.write(model_json)

def load_kmodel(filename="keras_model"):

        json_model = None

        with open(filename + ".json", "r") as json_file:
            json_model = json_file.read()

        model = tf.keras.models.model_from_json(json_model)
        model.load_weights(filename + ".hdf5")
        return model

# loss functions in tf/keras    
def relative_err(y_true, y_pred):
    sum_err_sqr = K.sum(K.square(y_true - y_pred))
    sum_val_sqr = K.clip(K.sum(K.square(y_true)), K.epsilon(), None)
    return K.sqrt(sum_err_sqr / sum_val_sqr)


# not quite luminance
def luminance(rgb):
    return K.sqrt(0.299 * K.square(rgb[:, 0]) + 0.587 * K.square(rgb[:, 1]) + 0.114 * K.square(rgb[:, 2]))


def image_loss(y_true, y_pred):
    # y_true = tf.keras.layers.Lambda(cast32)(y_true)
    lum_true = luminance(y_true)
    lum_pred = luminance(y_pred)
    mse_rgb = tf.losses.mean_squared_error(y_true, y_pred)
    mse_lum = tf.losses.mean_squared_error(lum_true, lum_pred)
    return mse_rgb + mse_lum + (relative_err(y_true, y_pred) * 0.01)


# not quite luminance
def luminance_conv(rgb):
    return K.sqrt(0.299 * K.square(rgb[:,:, :, 0]) + 0.587 * K.square(rgb[:, :, :,  1]) + 0.114 * K.square(rgb[:, :, :, 2]))


def image_loss_conv(y_true, y_pred):
    # y_true = tf.keras.layers.Lambda(cast32)(y_true)
    lum_true = luminance_conv(y_true)
    lum_pred = luminance_conv(y_pred)
    mse_rgb = tf.losses.mean_squared_error(y_true, y_pred)
    mse_lum = tf.losses.mean_squared_error(lum_true, lum_pred)
    return mse_rgb + mse_lum + (relative_err(y_true, y_pred) * 0.01)


def run_batch(model, dataset, optimizer, steps_per_epoch, prefix="", training=True, image_loss=image_loss_conv):
    start = datetime.datetime.now()
    loss_history = []
    metric_history = []

    train_loss = tfe.metrics.Mean('train_loss')
    metric_loss = tfe.metrics.Mean('metric_loss')

    train_loss.init_variables()
    metric_loss.init_variables()

    pb = Progbar(steps_per_epoch)
    for (batch, (x, y_true)) in enumerate(dataset.take(steps_per_epoch)):

        if training:
            with tf.GradientTape() as tape:
                y_pred = model(x, training=True)
                loss_value = image_loss(y_true, y_pred)
            grads = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables),
                                      global_step=tf.train.get_or_create_global_step())
        else:
            y_pred = model(x, training=False)

        loss_value = image_loss(y_true, y_pred)
        train_loss(loss_value)

        metric_value = relative_err(y_true, y_pred)
        metric_loss(metric_value)

        loss_history.append(loss_value.numpy())
        metric_history.append(metric_value.numpy())

        pb.update(batch, [(prefix + 'loss', loss_value), (prefix + 'relative error', metric_value)])

    finish = datetime.datetime.now()
    duration = finish - start
    duration = (datetime.datetime(1, 1, 1) + duration).strftime("%H:%M:%S")
    return {'loss_history': loss_history,
            'metric_history': metric_history,
            'loss': train_loss.result().numpy(),
            'metric': metric_loss.result().numpy(),
            'duration': duration}



def train_model_eager(model, data, batch_size=50000, epochs=50):
    num_samples = data.train.x_values.shape[0]
    num_validation_samples = data.validation.x_values.shape[0]

    prefetch_batch_buffer = 5

    steps_per_epoch = int(num_samples / batch_size)
    tfds_train = tf.data.Dataset.from_generator(create_generator(data, 1000), (tf.float32, tf.float32))
    tfds_train = tfds_train.shuffle(steps_per_epoch).batch(int(50000 / 1000))
    tfds_train = tfds_train.prefetch(prefetch_batch_buffer)

    tfds_val = tf.data.Dataset.from_generator(create_generator(data, batch_size, b_train=False),
                                              (tf.float32, tf.float32))
    tfds_val = tfds_val.batch(1)
    tfds_val = tfds_val.prefetch(prefetch_batch_buffer)

    steps_per_validation_epoch = int(num_validation_samples / batch_size)

    epoch_history = []
    val_loss_history = []

    reduce_window = 3
    stopping_window = 5
    cool_down_steps = 3
    cool_down = 0

    optimizer = tf.train.AdamOptimizer()

    for epoch in range(epochs):

        print("Training epoch", epoch)
        train_history = run_batch(model, tfds_train, optimizer, steps_per_epoch, training=True)
        print("")

        print("Validation...")
        val_history = run_batch(model, tfds_val, optimizer, steps_per_validation_epoch, prefix="val_", training=False)
        print("")

        epoch_history.append({'train_history': train_history, 'val_history': val_history})

        ## print out summary 
        keys = ('loss', 'metric', 'duration')
        print("Summary: epoch", epoch)
        print("training summary:", {k: train_history[k] for k in keys})
        print("validation summary", {k: val_history[k] for k in keys})

        ## test for early stopping etc 
        val_loss = val_history['loss']

        if len(val_loss_history) > reduce_window + cool_down:
            count = np.sum(np.array(val_loss_history[-reduce_window:]) < val_loss)
            print("plateau count", count)
            if count == reduce_window:
                optimizer._lr *= 0.5
                print("Reducing learning rate to ", optimizer._lr)
                cool_down = cool_down_steps

        if len(val_loss_history) > stopping_window + cool_down:
            if np.sum(np.array(val_loss_history[-stopping_window:]) < val_loss) == stopping_window:
                print("Stopping Early")
                return epoch_history

        if cool_down > 0:
            cool_down = cool_down - 1

        val_loss_history.append(val_loss)

    return epoch_history




def train_conv_model_eager(model, data, batch_size=5, epochs=5000):

    num_samples = data.train.num_source_images
    num_validation_samples = data.validation.num_source_images

    steps_per_epoch = int(num_samples / batch_size)
    steps_per_validation_epoch = int(num_validation_samples / batch_size)
    
    tfds_train = create_image_tfdataset(data, batch_size, b_train = True)
    tfds_val = create_image_tfdataset(data, batch_size, b_train=False)

    

    epoch_history = []
    val_loss_history = []

    reduce_window = 5
    stopping_window = 10
    cool_down_steps = 3
    cool_down = 0
    min_lr = 1e-6

    optimizer = tf.train.AdamOptimizer()

    for epoch in range(epochs):

        print("Training epoch", epoch)
        train_history = run_batch(model, tfds_train, optimizer, steps_per_epoch, training=True)
        print("")

        print("Validation...")
        val_history = run_batch(model, tfds_val, optimizer, steps_per_validation_epoch, prefix="val_", training=False)
        print("")

        epoch_history.append({'train_history': train_history, 'val_history': val_history})

        ## print out summary 
        keys = ('loss', 'metric', 'duration')
        print("Summary: epoch", epoch)
        print("training summary:", {k: train_history[k] for k in keys})
        print("validation summary", {k: val_history[k] for k in keys})

        ## test for early stopping etc 
        val_loss = val_history['loss']

        if len(val_loss_history) > reduce_window + cool_down:
            # how many of the last losses are less than the current value
            count = np.sum(np.array(val_loss_history[-reduce_window:]) < val_loss)
            print("plateau count", count)
            
            # more all of the window?
            if count == reduce_window and min_lr >= optimizer._lr:
                optimizer._lr *= 0.5
                print("Reducing learning rate to ", optimizer._lr)
                cool_down = cool_down_steps
                # truncate history
                val_loss_history = val_loss_history[-reduce_window:]

        if len(val_loss_history) > stopping_window + cool_down:
            if np.sum(np.array(val_loss_history[-stopping_window:]) < val_loss) == stopping_window:
                print("Stopping Early")
                return epoch_history

        if cool_down > 0:
            cool_down = cool_down - 1

        val_loss_history.append(val_loss)

    return epoch_history


def train_model_tf(model, data, batch_size=50000, epochs=100):
    
    LOG_DIR = "logs" + datetime.datetime.now().strftime("%y_%h_%d_%H_%M_%S")
    MODEL_DIR = "model_" + datetime.datetime.now().strftime("%y_%h_%d_%H_%M")+".ckpt"
    

    num_samples = data.train.x_values.shape[0]
    num_validation_samples = data.validation.x_values.shape[0]

    prefetch_batch_buffer = 5

    steps_per_epoch = int(num_samples / batch_size)
    tfds_train = tf.data.Dataset.from_generator(create_generator(data, 1000, b_train=False),
                                                (tf.float32, tf.float32),
                                                (tf.TensorShape([1000, 7]), tf.TensorShape([1000, 3])))

    tfds_train = tfds_train.shuffle(steps_per_epoch).batch(int(50000 / 1000))
    tfds_train = tfds_train.prefetch(prefetch_batch_buffer)

    tfds_val = tf.data.Dataset.from_generator(create_generator(data, batch_size, b_train=False),
                                              (tf.float32, tf.float32),
                                              (tf.TensorShape([batch_size, 7]), tf.TensorShape([batch_size, 3])))

    tfds_val = tfds_val.batch(1)
    tfds_train = tfds_train.prefetch(prefetch_batch_buffer)
    tfds_val = tfds_val.prefetch(prefetch_batch_buffer)

    steps_per_validation_epoch = int(num_validation_samples / batch_size)

    def create_loss_and_metric(tfds, model):
        one_shot = tfds.make_one_shot_iterator()

        x_values, y_true = one_shot.get_next()
        
        
        
        y_pred = model(x_values)

        loss = image_loss(y_true, y_pred)
        metric = relative_err(y_true, y_pred)
        return loss, metric

    loss, metric = create_loss_and_metric(tfds_train, model)
    loss_val, metric_val = create_loss_and_metric(tfds_val, model)

    initial_lrate=0.001
    with tf.name_scope('Optimizer'):
        

        # Op to calculate every variable gradient
        optimizer = tf.train.AdamOptimizer(initial_lrate)
        grads_and_vars = optimizer.compute_gradients(loss)

        # gradient clipping doesn't seem necessary
        
        # Op to update all variables according to their gradient
        train_step = optimizer.apply_gradients(grads_and_vars=grads_and_vars)
        

    init = tf.global_variables_initializer()

    # Tensorboard metrics
    update_frequency = 100
    
    with tf.name_scope('train_and_val'):
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("relative_error", metric)

        tf.summary.scalar("val_loss", loss_val)
        tf.summary.scalar("val_relative_error", metric_val)

    # Create summaries to visualize weights
    for var in tf.trainable_variables():
        tf.summary.histogram(var.name, var)
    # Summarize all gradients
    
    with tf.name_scope('gradients'):
        for grad, var in grads_and_vars:
            tf.summary.histogram(var.name + '/gradient', grad)
            
            grad_norm = tf.sqrt(tf.reduce_mean(grad**2))
            tf.summary.scalar(var.name +'/grad_norm', grad_norm)
            
            grad_l1_norm = tf.norm(grad, ord=1)
            tf.summary.scalar(var.name +'/grad_l1_norm', grad_l1_norm)
            
        

    summary = tf.summary.merge_all()
    
    saver = tf.train.Saver()

    # finalize the graph to help reduce bugs, this means things need to be reset before re-calling
    # tf.reset_default_graph()

    #tf.get_default_graph().finalize()

    epoch_history = []
    epoch_history_val = []

    reduce_window = 3
    stopping_window = 5
    cool_down_steps = np.max([reduce_window, stopping_window])
    cool_down = 0





    with tf.Session() as sess:

        summary_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)

        step = 0

        sess.run(init)

        # Iterating through all the epochs 
        for epoch in range(epochs):

            loss_history = []
            metric_history = []

            loss_history_val = []
            metric_history_val = []

            start = datetime.datetime.now()
            pb = Progbar(steps_per_epoch)

            print("Training epoch", epoch, "of", epochs)

            for batch in range(steps_per_epoch):

                _, loss_t, metric_t, loss_v, metric_v = sess.run([train_step, loss, metric, loss_val, metric_val])

                pb.update(batch, [('loss', loss_t), ('relative error', metric_t), ('val_loss', loss_v),
                                  ('val_relative error', metric_v)])

                loss_history.append(loss_t)
                metric_history.append(metric_t)

                loss_history_val.append(loss_v)
                metric_history_val.append(metric_v)

                # write out summary
                if batch % update_frequency == 0:
                    # logs = sess.run(tb_callback.merged)
                    summary_str = sess.run(summary)
                    summary_writer.add_summary(summary_str, step)
                    summary_writer.flush()
                    step = step + 1

            print("")

            finish = datetime.datetime.now()
            duration = finish - start
            duration = (datetime.datetime(1, 1, 1) + duration).strftime("%H:%M:%S")
            train_history = {'loss_history': loss_history,
                             'metric_history': metric_history,
                             'loss': np.mean(loss_history),
                             'metric': np.mean(metric_history),
                             'duration': duration}

            val_history = {'loss_history': loss_history_val,
                           'metric_history': metric_history_val,
                           'loss': np.mean(loss_history_val),
                           'metric': np.mean(metric_history_val),
                           'duration': duration}

            epoch_history.append({'train_history': train_history, 'val_history': val_history})

            ## print out summary 
            keys = ('loss', 'metric', 'duration')
            print("Summary: epoch", epoch)
            print("training summary:", {k: train_history[k] for k in keys})
            print("validation summary", {k: val_history[k] for k in keys})

            ## test for early stopping etc 
            val_loss = val_history['loss']

            if len(epoch_history_val) > reduce_window + cool_down:
                count = np.sum(np.array(epoch_history_val[-reduce_window:]) < val_loss)
                print("plateau count", count)
                if count == reduce_window:
                    optimizer._lr *= 0.5
                    print("Reducing learning rate to ", optimizer._lr)
                    cool_down = cool_down_steps
                    continue

            if len(epoch_history_val) > stopping_window + cool_down:
                if np.sum(np.array(epoch_history_val[-stopping_window:]) < val_loss) == stopping_window:
                    print("Stopping Early")
                    break

            if cool_down > 0:
                cool_down = cool_down - 1

            epoch_history_val.append(val_loss)

        # save the current batch
        save_path = saver.save(sess, MODEL_DIR)
        print("Model saved in path: %s" % save_path)
        save_model(model, "./" + MODEL_DIR)

    return epoch_history


def train_conv_model_tf(model, data, batch_size=1, epochs=100):
    
    LOG_DIR = "logs" + datetime.datetime.now().strftime("%y_%h_%d_%H_%M_%S")
    MODEL_DIR = "model_" + datetime.datetime.now().strftime("%y_%h_%d_%H_%M")+".ckpt"
    
    num_samples = data.train.num_source_images
    num_validation_samples = data.validation.num_source_images
    steps_per_validation_epoch = int(num_validation_samples / batch_size)

    steps_per_epoch = int(num_samples / batch_size)
    
    tfds_train = create_image_tfdataset(data, batch_size, b_train = True).repeat(epochs*10)
    tfds_val = create_image_tfdataset(data, batch_size, b_train=False).repeat(epochs*1000 )

    def create_loss_and_metric(tfds, model):
        one_shot = tfds.make_one_shot_iterator()

        x_values, y_true = one_shot.get_next()
        
        x_shape = [-1 if x is None else x  for x in model.layers[0].input_shape]
        y_shape = [-1 if x is None else x  for x in model.layers[-1].output_shape]
        
        y_pred = model(tf.reshape(x_values, x_shape))

        loss = image_loss_conv(y_true, y_pred)
        metric = relative_err(y_true, y_pred)
        return loss, metric

    loss, metric = create_loss_and_metric(tfds_train, model)
    loss_val, metric_val = create_loss_and_metric(tfds_val, model)

    # initial_lrate=0.00001  
    with tf.name_scope('Optimizer'):
        # Gradient Descent
        # train_step = tf.train.AdamOptimizer().minimize(loss)

        # Op to calculate every variable gradient
        optimizer = tf.train.AdamOptimizer()
        
        # no clip
        # grads_and_vars = optimizer.compute_gradients(loss)
        # Op to update all variables according to their gradient
        # train_step = optimizer.apply_gradients(grads_and_vars=grads_and_vars)
        
        
        #clip
        gs, vs = zip(*optimizer.compute_gradients(loss))
        clipped, _ = tf.clip_by_global_norm(gs, 1.0)
        grads_and_vars = zip(clipped, vs)   
            
        
        train_step = optimizer.apply_gradients(grads_and_vars)

    init = tf.global_variables_initializer()

    # Tensorboard metrics
    update_frequency = int(steps_per_epoch/100)
    
    with tf.name_scope('train_and_val'):
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("relative_error", metric)

        tf.summary.scalar("val_loss", loss_val)
        tf.summary.scalar("val_relative_error", metric_val)

    # Create summaries to visualize weights
    for var in tf.trainable_variables():
        tf.summary.histogram(var.name, var)
    # Summarize all gradients
    
    with tf.name_scope('gradients'):
        for grad, var in grads_and_vars:
            tf.summary.histogram(var.name + '/gradient', grad)
            
            grad_l2_norm = tf.norm(grad, ord=2)
            tf.summary.scalar(var.name +'/grad_l2_norm', grad_l2_norm)
            
            grad_l1_norm = tf.norm(grad, ord=1)
            tf.summary.scalar(var.name +'/grad_l1_norm', grad_l1_norm)

    summary = tf.summary.merge_all()
    
    saver = tf.train.Saver()

    # finalize the graph to help reduce bugs, this means things need to be reset before re-calling
    # tf.reset_default_graph()

    #tf.get_default_graph().finalize()

    epoch_history = []
    epoch_history_val = []

    reduce_window = 3
    stopping_window = 5
    cool_down_steps = np.max([reduce_window, stopping_window])
    cool_down = 0





    with tf.Session() as sess:

        summary_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)

        step = 0

        sess.run(init)

        # Iterating through all the epochs 
        for epoch in range(epochs):

            loss_history = []
            metric_history = []

            loss_history_val = []
            metric_history_val = []

            start = datetime.datetime.now()
            pb = Progbar(steps_per_epoch)

            print("Training epoch", epoch, "of", epochs)

            
            for batch in range(steps_per_epoch):

                _, loss_t, metric_t, loss_v, metric_v = sess.run([train_step, loss, metric, loss_val, metric_val])

                pb.update(batch, [('loss', loss_t), ('relative error', metric_t), ('val_loss', loss_v),
                                  ('val_relative error', metric_v)])

                loss_history.append(loss_t)
                metric_history.append(metric_t)

                loss_history_val.append(loss_v)
                metric_history_val.append(metric_v)

                # write out summary
                if batch % update_frequency == 0:
                    # logs = sess.run(tb_callback.merged)
                    summary_str = sess.run(summary)
                    summary_writer.add_summary(summary_str, step)
                    summary_writer.flush()
                    step = step + 1

            print("")

               
                    
        

            print("")

            finish = datetime.datetime.now()
            duration = finish - start
            duration = (datetime.datetime(1, 1, 1) + duration).strftime("%H:%M:%S")
            train_history = {'loss_history': loss_history,
                             'metric_history': metric_history,
                             'loss': np.mean(loss_history),
                             'metric': np.mean(metric_history),
                             'duration': duration}

            val_history = {'loss_history': loss_history_val,
                           'metric_history': metric_history_val,
                           'loss': np.mean(loss_history_val),
                           'metric': np.mean(metric_history_val),
                           'duration': duration}

            epoch_history.append({'train_history': train_history, 'val_history': val_history})

            ## print out summary 
            keys = ('loss', 'metric', 'duration')
            print("Summary: epoch", epoch)
            print("training summary:", {k: train_history[k] for k in keys})
            print("validation summary", {k: val_history[k] for k in keys})

            ## test for early stopping etc 
            val_loss = val_history['loss']

            if len(epoch_history_val) > reduce_window + cool_down:
                count = np.sum(np.array(epoch_history_val[-reduce_window:]) < val_loss)
                print("plateau count", count)
                if count == reduce_window:
                    optimizer._lr *= 0.5
                    print("Reducing learning rate to ", optimizer._lr)
                    cool_down = cool_down_steps
                    continue

            if len(epoch_history_val) > stopping_window + cool_down:
                if np.sum(np.array(epoch_history_val[-stopping_window:]) < val_loss) == stopping_window:
                    print("Stopping Early")
                    break

            if cool_down > 0:
                cool_down = cool_down - 1

            epoch_history_val.append(val_loss)

        # save the current batch
        save_path = saver.save(sess, MODEL_DIR)
        print("Model saved in path: %s" % save_path)
        save_model(model, "./" + MODEL_DIR)

    return epoch_history


def run():
    data_dir = os.path.join("../data", "waldorf")
    data = image_data.read_data(data_dir=data_dir, transform=exposure_compensate4_22, train=0.5, validation=0.1,
                                test=0.4)
    model = create_model()

    if tf.executing_eagerly():
        h = train_model_eager(model, data)
    else:
        h = train_model_tf(model, data)
    return h

def run_conv():
    data_dir = os.path.join("../data", "waldorf")
    data = image_data.read_data(data_dir=data_dir, transform=exposure_compensate4_22, train=0.5, validation=0.1,
                                test=0.4)
    model = create_conv_model(filters=600, kernel_size=1)

    if tf.executing_eagerly():
        h = train_conv_model_eager(model, data, batch_size=1)
    else:
        h = train_conv_model_tf(model, data, batch_size=1)
    return h
