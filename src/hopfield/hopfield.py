#!/usr/bin/env python

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os


def img_threshold(pixel, threshold=0.5):
    ''' Convert pixel into a B&W by applying a threshold '''
    if isinstance(pixel, np.ndarray):
        # IF it's RGB vector, convert to average
        pixel = (pixel[0] + pixel[1] + pixel[2]) / 3
    return 1.0 if pixel >= threshold else -1.0


def img2bw(file_name, show=False):
    ''' Load an image, convert it to black & white and return pixes '''
    img = mpimg.imread(file_name)
    imgbw = img.copy()
    size_x = imgbw.shape[0]
    size_y = imgbw.shape[1]
    pixels = np.zeros((size_x, size_y))
    for i in range(size_x):
        for j in range(size_y):
            # print(f"{i, j}\t{imgbw[i][j]}")
            pixels[i][j] = img_threshold(imgbw[i][j])

    imgbw = pixels
    imgbw.shape = pixels.shape
    print(f"DEBUG img2bw: File '{file_name}', image B&W shape {imgbw.shape}")

    if show:
        fig, (plt_ori, plt_bw) = plt.subplots(1, 2)
        plt_ori.imshow(img, cmap='gray')
        plt_ori.axis('off')
        plt_ori.set_title('Original')
        plt_bw.imshow(imgbw, cmap='gray')
        plt_bw.axis('off')
        plt_bw.set_title('B&W')
        plt.show()

    return pixels.flatten()


class Hopfield:

    def __init__(self):
        self.samples = list()
        self.len = -1
        self.weight_matrix_file = "hopfield_demo_w.npy"
        pass

    def add_image(self, file):
        ''' Add an image as input sample to be learned '''
        self.add_vector(img2bw(file))

    def add_vector(self, v):
        ''' Add a vector as input sample to be learned '''
        # Check that all vectors have the same length
        if self.len < 0:
            self.len = len(v)
        if self.len != len(v):
            print(f"ERROR: Length does not match, len={self.len}, input vector length {len(v)}")
        # Add vector to list
        self.samples.append(v)

    def calc(self, input):
        if len(input) != self.len:
            print(f"ERROR: Input length {len(input)} doesn't match network size {self.len}")
            exit(1)
        self.s = input
        self.s = np.matmul(self.w, self.s)
        self.s[self.s >= 0] = 1.0
        self.s[self.s < 0] = -1.0
        print(f"DEBUG Hopfield.calc: s = {self.s}")

    def learn(self):
        ''' Learn using Hopfield equations '''
        n = len(self.samples)
        self.w = np.zeros((self.len, self.len))
        for s in self.samples:
            print(f"DEBUG Hopfield.learn: W shape {np.shape(self.w)}, sample shape {np.shape(s)}")
            ws = np.outer(s, s)
            print(f"DEBUG Hopfield.learn: Ws shape {np.shape(ws)}")
            self.w = np.add(self.w, ws)
        self.w = self.w / n

    def train(self, images):
        if os.path.isfile(self.weight_matrix_file):
            print(f"DEBUG Hopfield.train: Loading weight matrix file '{self.weight_matrix_file}'")
            self.w = np.load(self.weight_matrix_file)
            self.len = np.shape(self.w)[0]
            return

        # Load all images and add them to training set
        for img in images:
            nn.add_image(img)

        # Learn weights and save them to a file
        self.learn()
        print(f"DEBUG Hopfield.train: Saving weight matrix to file '{self.weight_matrix_file}'")
        np.save(self.weight_matrix_file, self.w)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

# pixels = img2bw('data/images/lena.png', True)

img_path = 'data/images'
images = [f"{img_path}/baboon_small.png", f"{img_path}/lena_small.png"]

nn = Hopfield()
nn.train(images)

input = img2bw(images[0])
nn.calc(input)

# imgplot = plt.imshow(img)
