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
Implements basic file IO and a DataSet object
"""


from __future__ import print_function
import os
import glob
import gc
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import tables
import collections
import wget



Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])
data_floatx = np.float16


class DataSet(object):

    def __init__(self, x_values, y_values, shape, original_index):
        self._x_values = x_values
        self._y_values = y_values
        self._shape = shape
        self._original_index = original_index
        self._num_source_images = int(self._x_values.shape[0] / (self._shape[0] * self._shape[1]))
        self._num_examples = self._x_values.shape[0]
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def shape(self):
        return self._shape[0], self._shape[1], self._num_source_images

    @property
    def num_source_images(self):
        return self._num_source_images

    @property
    def x_values(self):
        return self._x_values

    @property
    def y_values(self):
        return self._y_values

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def x_value_size(self):
        return self._x_values[0, :].shape[0]

    @property
    def y_value_size(self):
        return self._y_values[0, :].shape[0]

    @property
    def window_size(self):
        return self._shape

    def to_image(self, y_value, exposure=2.0, gamma=2.2):
        # sub = sub.reshape( (self._shape[0],self._shape[1],3) )/2.0 + 0.5
        sub = y_value.reshape((self._shape[0], self._shape[1], 3))
        sub = np.clip(sub, 0, 1)
        sub = hdr_to_image(sub,exposure,gamma)
        return np.transpose(sub, (1, 0, 2))


    def plot_y_values(self, i, title=None, exposure=2.0, gamma=2.2):
        dim = self._shape[0] * self._shape[1]
        offset = i * dim
        sub = self.to_image(self._y_values[offset:(offset + dim), :], exposure, gamma)
        plt.figure(figsize=(10, 10))
        if title is None:
            title = "Image #" + str(self._original_index[i])

        plt.xlabel(title, fontsize=18)
        plt.imshow(sub.astype(np.float32))


    def plot_prediction(self, prediction, title=None, exposure=2.0, gamma=2.2):
        sub = self.to_image(prediction, exposure, gamma)
        plt.figure(figsize=(10, 10))
        if title is not None:
            plt.xlabel(title, fontsize=18)
        plt.imshow(sub.astype(np.float32))


    def get_data_slice(self, idx):
        # data can get re-arranged
        dim = self._shape[0] * self._shape[1]
        offset = idx * dim

        x_values = self._x_values[offset:(offset + dim), :]
        y_values = self._y_values[offset:(offset + dim), :]
        return x_values, y_values
    
    
    def gradient_image(self, image):
        # Get x-gradient in "sx"
        sx = ndimage.sobel(image.astype(np.float32).reshape((self._shape[0], self._shape[1], 3)),axis=0,mode='constant')
        # Get y-gradient in "sy"
        sy = ndimage.sobel(image.astype(np.float32).reshape((self._shape[0], self._shape[1], 3)),axis=1,mode='constant')
        # Get square root of sum of squares
        sobel=np.hypot(sx,sy)
        sobel /= np.max(sobel)
        
        return sobel.reshape(image.shape)


    def image_indices(self, indices):
        n_samples = self.shape[2]
        dim = self._shape[0] * self._shape[1]

        data = self._x_values.reshape((n_samples, dim, 7))
        labels = self._y_values.reshape((n_samples, dim, 3))

        sub_data = data[:, indices, :]
        sub_labels = labels[:, indices, :]

        s0 = sub_data.shape[0]*sub_data.shape[1]
        sub_data = sub_data.reshape((s0, 7))
        sub_labels = sub_labels.reshape((s0, 3))

        return sub_data, sub_labels


def xy(i, w):
    # a 32 x 32 light pattern
    y = float(np.floor(float(i) / float(w))) / float(w)
    x = float(np.mod(i, int(w))) / float(w)
    return x, y


def index_2_xy(i, w, h):
    y = float(np.floor(float(i) / float(w))) / float(h)
    x = float(np.mod(i, int(w))) / float(w)
    return x, y


def hdr_to_image(RGBColor, exposure, gamma, clip=True):
    m = np.power(2.0, exposure)
    g = 1.0 / gamma
    RGB = np.power(RGBColor * m, g)

    if clip:
        RGB[RGB < 0.0] = 0.0
        RGB[RGB > 1.0] = 1.0

    return RGB



def read_full_data(filename, max_value=None):
    h5file = tables.open_file(filename, mode="r")
    hdf5_data = h5file.root.LTM[:]
    h5file.close()

    # dim = 696*464

    if max_value is None:
        max_value = np.max(np.max(hdf5_data))

    images = ((np.array(hdf5_data) / max_value) - 0.5) * 2.0
    labels = np.zeros((hdf5_data.shape[0], 1)).astype(data_floatx)
    for i in range(hdf5_data.shape[0]):
        labels[i, :] = i

    return images, labels, max_value, (696, 464)


def read_data_rgb(base, filenames, transform=None):
    gc.collect()

    def get_channel(f):
        gc.collect()
        with tables.open_file(os.path.join(base, f), mode="r") as h5file:
            hdf5_data = h5file.root.LTM[:]
            h5file.close()
        return np.array(hdf5_data)

    rgb = np.stack([get_channel(f) for f in filenames], -1)
    gc.collect()

    if transform is not None:
        rgb = transform(rgb)

    maxval = np.max(rgb)
    rgb /= maxval

    print('rgb mean: ' + str(np.mean(rgb)))
    gc.collect()

    return rgb.astype(data_floatx), (696, 464)


def create_x_values(rgb, shape):
    # input Px,Py, Lx, Ly, mean
    data = np.zeros((rgb.shape[0], rgb.shape[1], 7)).astype(data_floatx)

    # fill in Light
    for i in range(rgb.shape[0]):
        Pos = xy(i, 32)
        data[i, :, 0] = Pos[0]
        data[i, :, 1] = Pos[1]

    # fill in Position, width & height are transposed
    for i in range(rgb.shape[1]):
        Pos = index_2_xy(i, shape[1], shape[0])
        data[:, i, 2] = Pos[0]
        data[:, i, 3] = Pos[1]

    # fill in mean RGB
    meanrgb = np.mean(rgb, 0)
    for i in range(rgb.shape[0]):
        data[i, :, 4:7] = meanrgb

    return data.reshape(rgb.shape[0] * rgb.shape[1], 7), \
           rgb.reshape(rgb.shape[0] * rgb.shape[1], rgb.shape[2]), \
           shape

# Try to download the data from the original source
# Matthew O'Toole and Kiriakos N. Kutulakos. 2010. Optical computing for fast light transport analysis.
# ACM Trans. Graph. 29, 6, Article 164 (December 2010), 12 pages. DOI: https://doi.org/10.1145/1882261.1866165

def download_data(data_dir=os.path.join("../data", "waldorf")):

    red = str("Red.mat")
    green = str("Green.mat")
    blue = str("Blue.mat")

    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
        print("creating: " + str(data_dir))

    for f in [red, green, blue]:
        if not os.path.exists(os.path.join(data_dir, f)):
            src = "http://www.cs.toronto.edu/~motoole/data/waldorf/" + str(f)
            dest = os.path.join(data_dir, f)
            print("Downloading data file:" + str(src) + " to: " + str(dest))
            wget.download(src, out=dest)

    return [red, green, blue]


def write_hdf_data(x_values, y_values, shape, name='waldorf.hdf5', data_dir=os.path.join("data", "waldorf")):
    filename = os.path.join(data_dir, name)
    hdf = tables.open_file(filename=filename, mode='w')
    hdf.create_array(hdf.root, "x_values", x_values)
    hdf.create_array(hdf.root, "y_values", y_values)
    hdf.create_array(hdf.root, "shape", shape)
    hdf.close()


def read_hdf_data(name='waldorf.hdf5', data_dir=os.path.join("data", "waldorf")):
    filename = os.path.join(data_dir, name)
    hdf = tables.open_file(filename=filename, mode='r')
    x_values = np.array(hdf.root.x_values[:]).astype(dtype=data_floatx)
    y_values = np.array(hdf.root.y_values[:]).astype(dtype=data_floatx)
    shape = tuple(hdf.root.shape[:])

    hdf.close()
    return x_values, y_values, shape




def merge_split_file(data_dir="../data/waldorf", color_channel='Red'):
    
    mat_file = os.path.join(data_dir, color_channel + '.mat')
    if os.path.exists(mat_file):
        return
    
    chunks = list(glob.iglob(os.path.join(data_dir, color_channel + '.mat.*')))
    if not chunks:
        return
    
    with open(mat_file, "wb") as outfile:
        for f in chunks:
            with open(f, 'rb') as file:
                outfile.write(file.read())


def read_data_and_convert(data_dir=os.path.join("../data", "waldorf"), transform=None):

    gc.collect()
    
    # if the files are checked in, but not merged do so
    [merge_split_file(data_dir=data_dir, color_channel=c) for c in ['Red', 'Green', 'Blue']]

    # download (if needed) and get the file names
    filenames = download_data(data_dir=data_dir)
    rgb, shape = read_data_rgb(data_dir, filenames, transform)
    return create_x_values(rgb, shape)



def read_data(data_dir=os.path.join("../data", "waldorf"),
              transform=None, train=0.8, validation=0.1, test=0.1):

    gc.collect()

    print(data_dir)

    filename = 'RGBDATA.hdf5'
    if transform is not None:
        filename = 'RGBDATA_' + str(transform.__name__) + '.hdf5'

    if not os.path.exists(os.path.join(data_dir, filename)):
        x_values, y_values, shape = read_data_and_convert(data_dir, transform)
        write_hdf_data(x_values, y_values, shape, filename, data_dir)
    else:
        print("Reading: " + filename)
        x_values, y_values, shape = read_hdf_data(filename, data_dir)


    print("read raw data")

    # divide into train, validate, & test
    n_samples = int(y_values.shape[0]/(shape[0]*shape[1]))

    # normalize
    s = np.sum(np.array([train,validation,test]))
    train,validation, test = [train, validation, test]/s

    n_train = int(float(n_samples) * train)
    n_validation = int(float(n_samples) * (train+validation))

    x_values = x_values.reshape((n_samples, shape[0]*shape[1], 7))
    y_values = y_values.reshape((n_samples, shape[0]*shape[1], 3))

    samples = np.array(range(n_samples))
    np.random.shuffle(samples)
    train_idx = samples[:n_train]
    validation_idx = samples[n_train:n_validation]
    test_idx = samples[n_validation:-1]

    train_x = x_values[train_idx, :, :]
    train_y = y_values[train_idx, :, :]

    train_x = train_x.reshape((n_train*shape[0]*shape[1], 7))
    train_y = train_y.reshape((n_train*shape[0]*shape[1], 3))

    test_x = x_values[test_idx, :, :]
    test_y = y_values[test_idx, :, :]

    test_x = test_x.reshape((test_idx.shape[0]*shape[0]*shape[1], 7))
    test_y = test_y.reshape((test_idx.shape[0]*shape[0]*shape[1], 3))

    validation_x = x_values[validation_idx, :, :]
    validation_y = y_values[validation_idx, :, :]

    validation_x = validation_x.reshape((validation_idx.shape[0]*shape[0]*shape[1], 7))
    validation_y = validation_y.reshape((validation_idx.shape[0]*shape[0]*shape[1], 3))

    train = DataSet(train_x, train_y, shape, train_idx)
    validation = DataSet(validation_x, validation_y, shape, validation_idx)
    test = DataSet(test_x, test_y, shape, test_idx)

    gc.collect()
    return Datasets(train=train, validation=validation, test=test)
