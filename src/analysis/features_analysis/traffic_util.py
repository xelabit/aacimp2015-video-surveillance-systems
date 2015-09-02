import pandas as pd
import numpy as np
from scipy.misc import imread, imsave, imresize
import matplotlib.pyplot as plt

def traffic_load(path_to_npy_files):
    """ Load serialized npy traffic data """
    Xtr = np.load(path_to_npy_files + '/X_train.npy')
    Xte = np.load(path_to_npy_files + '/X_test.npy')
    ytr = np.load(path_to_npy_files + '/Y_train.npy')
    yte = np.load(path_to_npy_files + '/Y_test.npy')


    return Xtr, ytr, Xte, yte

def traffic_info_load(path_to_npy_files):
    """ Load serialized npy traffic data """
    info_train = np.load(path_to_npy_files + '/X_train_videos.npy')
    info_test = np.load(path_to_npy_files + '/X_test_videos.npy')

    return info_train, info_test


def traffic_hist(y):
    """ Show how classes are distributed """
    classes = ['light', 'medium', 'heavy']
    freq = np.bincount(y)

    width = 1.0     # gives histogram aspect to the bar diagram
    pos = np.arange(len(classes))
    ax = plt.axes()
    ax.set_xticks(pos + (width / 2))
    ax.set_xticklabels(classes)

    plt.bar(pos, freq, width, color='r')
    plt.show()

def traffic_resize(X_train, X_test, t):
    """ Preprocessing: Resize all data """
    def img_resize(img, t):
        """ Rsizes given images """
        return imresize(img, (img.shape[0] / t, img.shape[1] / t))


    X_train = np.array([img_resize(img, t) for img in X_train])
    X_test = np.array([img_resize(img, t) for img in X_test])

    return X_train, X_test

def traffic_2grey(X_train, X_test):
    """ Preprocessing: Convert features to grescale """

    def rgb2gray(rgb):
        return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

    X_train = np.array([rgb2gray(img) for img in X_train])
    X_test = np.array([rgb2gray(img) for img in X_test])

    return X_train, X_test

def display_subsample(samples_per_class, X_train, y_train):
    """ Visualize 'somples_per_class' examples from the dataset """

    classes = ["Light", "Medium", "Heavy"]
    num_classes = len(classes)
    for y, cls in enumerate(classes):
        idxs = np.flatnonzero(y_train == y)
        idxs = np.random.choice(idxs, samples_per_class, replace=False)
        for i, idx in enumerate(idxs):
            plt_idx = i * num_classes + y + 1
            plt.subplot(samples_per_class, num_classes, plt_idx)
            plt.imshow(X_train[idx], cmap = 'gray')
            plt.axis('off')
            if i == 0:
                plt.title(cls)
    plt.show()

def traffic_separate_bg(X_train, X_test):
    """ TODO: you code here """

    return X_train, X_test
