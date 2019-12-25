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


def spectual_clustering(img):
    kernel_img = kernel(im1, im1, gamma_s, gamma_c)
    for i in range(len(kernel_img)):
        kernel_img[i, i] = 0

    D = list()
    for i in range(len(kernel_img)):
        D.append(np.sum(kernel_img[i,:]))
    D = np.diag(np.array(D))
    
    # normal_cut
    D_inverse = np.linalg.inv(D)
    L = np.dot(D_inverse, np.dot(kernel_img, D_inverse))
    # ratio_cut
    L = kernel_img


    return


if __name__ == "__main__":
    im1 = np.array(Image.open('./image1.png')).reshape((10000, 3))
    # im2 = np.array(Image.open('./image2.png')).reshape((10000, 3))

    gamma_s, gamma_c = 1, 1
    spectual_clustering(im1)
