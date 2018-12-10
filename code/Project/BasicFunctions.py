
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

all_names = ['city', 'factory', 'field', 'forest', 'forest_field']

def norm1(X, y):
    """ X_norm = (X - np.mean(X)) / np.std(X) for y - the same
    """
    X_norm = (X - np.mean(X)) / np.std(X)
    y_norm = (y - np.mean(y)) / np.std(y)
    return X_norm, y_norm

def norm2(X, y):
    """ X_norm = (X - np.mean(X, axis=0)) / np.std(X, axis=0) for y - the same
    """
    ym = np.mean(y, axis = 0)
    ys = np.std(y, axis = 0)
    Xm = np.mean(X, axis = 0)
    Xs = np.std(X, axis = 0)
    X_norm = (X - Xm) / Xs
    y_norm = (y - ym) / ys
    return X_norm, y_norm

def norm3(X, y):
    """ X_norm = (X - np.min(X)) / np.max(X) for y - the same
    """
    X_norm = (X - np.min(X)) / np.max(X)
    y_norm = (y - np.min(y)) / np.max(y)
    return X_norm, y_norm

def bigDataSet(dataset, names):
    """ Concatenate all dataset with names and give one big array with all pictures.
    """
    N = dataset[names[0]].shape[0]
    w, h = dataset[names[0]].shape[1], dataset[names[0]].shape[2]
    new_dataset = np.zeros((N * len(names), w, h))
    for i, name in enumerate(names):
        new_dataset[i * N: (i + 1) * N] = dataset[name] 
    return new_dataset
    
def whole_pic_line(X):
    """Make one picture from array X of pictures in lines 
    """
    L = int((X.shape[0] * X.shape[1]) ** 0.5)
    picture = np.zeros((L, L))
    d = int((X.shape[1]) ** 0.5)
    k = 0
    for i in range(0, L, d):
        for j in range(0, L, d):
            picture[i: i + d, j: j + d] = np.reshape(X[k], (d, d))
            k += 1
    return picture

def whole_pic(X, overlap=0):
    """ Make one picture from array X of pictures with overlap.
    Overlap is a number of overlapping pixels from one side
    """
    L = int(X.shape[0] ** 0.5 * (X.shape[1] - 2 * overlap))
    picture = np.zeros((L, L))
    d = int(X.shape[1] - 2 * overlap)
    k = 0
    for i in range(0, L, d):
        for j in range(0, L, d):
            picture[i: i + d, j: j + d] = X[k, overlap:d + overlap, overlap:d + overlap]
            k += 1
    return picture

def tt_plot(train_cuda_MSE, test_cuda_MSE):
    """ Plot train and test MSE.
    """
    plt.plot(train_cuda_MSE, label='train')
    plt.plot(test_cuda_MSE, label= 'test')
    plt.legend()
    plt.show()
    
def pic_90x90(X, overlap=0):
    """ Show the piece 90x90 of the left angle of picture X. X is numpy.array of overlap pictures.
    """
    plt.figure(figsize=(10,10))
    plt.imshow(whole_pic(X, overlap)[:90, :90], cmap='gray')
    plt.show()
