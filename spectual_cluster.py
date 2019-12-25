import numpy as np
import os
from PIL import Image
from scipy.spatial import distance
import matplotlib.pyplot as plt


def kernel(x, y, gamma_s, gamma_c):
    dist_color = (distance.cdist(x, y, metric='euclidean'))**2
    coord_list = list()
    for i in range(100):
        for j in range(100):
            coord_list.append((i, j))
    coord_x = np.array(coord_list)
    coord_y = np.copy(coord_x)
    dist_spatial = (distance.cdist(coord_x, coord_y, metric='euclidean'))**2

    return np.dot(np.exp(-gamma_s * dist_spatial), np.exp(-gamma_c * dist_color))


if __name__ == "__main__":
    im1 = np.array(Image.open('./image1.png')).reshape((10000, 3))
    # im2 = np.array(Image.open('./image2.png')).reshape((10000, 3))

    gamma_s, gamma_c = 1, 1
    kernel_im1 = kernel(im1, im1, gamma_s, gamma_c)
    for i in range(len(kernel_im1)):
        kernel_im1[i, i] = 0
    
