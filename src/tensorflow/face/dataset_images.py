#!/usr/bin/env python

from __future__ import print_function

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
import os.path


class DataSetImages(object):

    def __init__(self):
        self.size_min = 20
        self.file_pickle = 'data/images.pickle'
        self.file_img_list = 'data/images.txt'
        self.show_image = False
        self.test_percent = 0.10
        self.batch_num = 0
        self.samples_per_batch = 100
        self.test_size = 0
        self.train_size = 0

    def create_train_test_sets(self, one_hot=False):
        X = np.array([i for i in self.df.loc[:, 'img']])
        y = np.array(self.df.face.values, dtype=float).reshape(len(self.df), 1)
        files = np.array(self.df.file.values).reshape(len(self.df), 1)
        if one_hot:
            y = self.convert_one_hot(y)
        perm = np.random.permutation(len(X))
        X_shuf = X[perm]
        y_shuf = y[perm]
        files_shuf = files[perm]
        test = int(self.test_percent * len(X_shuf))
        self.X_test = X_shuf[0:test]
        self.y_test = y_shuf[0:test]
        self.files_test = files_shuf[0:test]
        self.X_train = X_shuf[test:]
        self.y_train = y_shuf[test:]
        self.files_train = files_shuf[test:]
        self.test_size = len(self.X_test)
        self.train_size = len(self.X_train)

    def convert_one_hot(self, y):
        """Convert class labels from scalars to one-hot vectors."""
        num_classes = len(np.unique(y))
        y_one_hot = np.zeros([len(y), num_classes])
        y_int = np.array(y, dtype=int)
        for i in range(len(y)):
            y_one_hot[i, y_int[i]] = 1
        return y_one_hot

    def load(self):
        '''
        Load images either from a list or from a pickle file
        '''
        if not os.path.exists(self.file_pickle):
            self.df = self.preprocess_images()
        else:
            print('Loading data from', self.file_pickle)
            self.df = pd.read_pickle(self.file_pickle)
        return self.df

    def next_batch(self):
        sample_size = len(self.X_train)
        start = (self.samples_per_batch * self.batch_num) % sample_size
        end = (self.samples_per_batch * (self.batch_num + 1)) % sample_size
        end = min(sample_size, end)
        self.batch_num = self.batch_num + 1
        return (self.X_train[start:end], self.y_train[start:end])

    def preprocess_images(self):
        '''
        Read a list of images as a data frame. Load each image, resize if
        necesary and add all images as last column in dataframe. Save dataframe
        as a pickle image
        '''
        print('Preprocessing data from ', self.file_img_list)
        dfimg = pd.read_table(self.file_img_list)
        i = 0
        img_list = []
        img_file_list = []
        face_list = []
        for index, row in dfimg.iterrows():
            img_file = row['file']
            img_face = row['face']
            im = ndimage.imread(img_file, mode='L')
            dim = im.shape

            # Skip non-square images
            if dim[0] != dim[1]:
                print('WARNING: Image is not square, skipping:',
                      ', size:', im.shape)
                continue

            size_im = dim[0]
            if size_im < self.size_min:
                print('WARNING: Image is too small, skipping:',
                      ', size:', im.shape)
                continue

            # Downsample to 20x20
            if size_im > self.size_min:
                ratio = (1.0 * self.size_min) / size_im
                im = zoom(im, ratio)

            # Reshape and scale
            im = im.reshape(im.size)
            im = (im - min(im)) / (1.0 * max(im) - min(im))
            if i % 1000 == 0:
                print('\t', i,
                      ', size:', self.size_min,
                      ', face:', img_face,
                      ', file:', img_file,
                      ', shape: ', im.shape
                      )

            if self.show_image:
                plt.imshow(im, cmap='gray')
                plt.show()

            img_list.append(im)
            img_file_list.append(img_file)
            face_list.append(img_face)
            i = i + 1

        # Add colum to dataframe and save using pickle format
        img_series = pd.Series(img_list)
        self.df = pd.DataFrame({'face': face_list,
                                'file': img_file_list,
                                'img': img_series.values
                                })
        self.df.to_pickle(self.file_pickle)
        return self.df

    def test_set(self):
        return (self.X_test, self.y_test)


# Main
if __name__ == '__main__':
    ds = DataSetImages()
    df = ds.load()
    print(df.head)
