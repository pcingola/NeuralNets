#!/usr/bin/env python

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os


def img_noise(pixels, rand_th):
    ''' Randomly set pixes in an image '''
    r = np.random.rand(len(pixels))
    pixels[r < rand_th / 2] = -1
    pixels[(r >= rand_th / 2) & (r <= rand_th)] = +1
    return pixels


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
    # Convert to B&W using a simple threshold
    for i in range(size_x):
        for j in range(size_y):
            pixels[i][j] = img_threshold(imgbw[i][j])
    # Create new image
    imgbw = pixels
    imgbw.shape = pixels.shape
    print(f"DEBUG img2bw: File '{file_name}', image B&W shape {imgbw.shape}")
    # Show image?
    if show:
        show2(img, imgbw)
    # Return a flat array
    return pixels.flatten()


def show2(img1, img2):
    ''' Show two images '''
    fig, (plt1, plt2) = plt.subplots(1, 2)
    print(f"img1: {img1.shape}, img2: {img2.shape}")
    plt1.imshow(img1, cmap='gray')
    plt1.axis('off')
    plt2.imshow(img2, cmap='gray')
    plt2.axis('off')
    plt.show()


class Hopfield:
    ''' Simple neural network based on Hopfield model '''

    def __init__(self, size_x, size_y):
        self.size_x = size_x
        self.size_y = size_y
        self.samples = list()
        self.len = size_x * size_y
        self.weight_matrix_file = "hopfield_demo_w.npy"
        self.s = np.zeros(size_x * size_y)
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

    def calc(self):
        self.s_previous = np.copy(self.s)
        self.s = np.matmul(self.w, self.s)
        self.s[self.s >= 0] = 1.0
        self.s[self.s < 0] = -1.0
        print(f"DEBUG Hopfield.calc: s = {self.s}")

    def has_changed(self):
        ''' Has the network output changed? '''
        return np.any(self.s != self.s_previous)

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
        ''' Train neural network: Load all samples and calculate weigths '''
        if os.path.isfile(self.weight_matrix_file):
            print(f"DEBUG Hopfield.train: Loading weight matrix file '{self.weight_matrix_file}'")
            self.w = np.load(self.weight_matrix_file)
            if self.len != np.shape(self.w)[0]:
                raise Exception("Matrix dimensions doesn't match image sizes")
            return

        # Load all images and add them to training set
        for img in images:
            nn.add_image(img)

        # Learn weights and save them to a file
        self.learn()
        print(f"DEBUG Hopfield.train: Saving weight matrix to file '{self.weight_matrix_file}'")
        np.save(self.weight_matrix_file, self.w)

    def show(self, img):
        ''' Show current output '''
        img_s = np.copy(self.s)
        img_s.shape = (self.size_x, self.size_x)
        show2(img, img_s)

    def set(self, input):
        ''' Set network '''
        if len(input) != self.len:
            print(f"ERROR: Input length {len(input)} doesn't match network size {self.len}")
            exit(1)
        self.s = input


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

# pixels = img2bw('data/images/lena.png', True)

img_path = 'data/images'
images = [f"{img_path}/lena_small.png", f"{img_path}/baboon_small.png"]

# Create a NN and train it
size_x = 64
size_y = 64
nn = Hopfield(size_x, size_y)
nn.train(images)

# Test using an image
input_ori = img2bw(images[0])
input_ori.shape = (size_x, size_y)

input = img2bw(images[0])
input_noise = img_noise(input, 0.5)
nn.set(input_noise)

for i in range(100):
    print(f"Iteration {i}")
    nn.show(input_ori)
    nn.calc()
    if not nn.has_changed():
        print("Done!")
        nn.show(input_ori)
        exit(0)


# imgplot = plt.imshow(img)
