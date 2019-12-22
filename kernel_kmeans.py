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


def kmeans_plus(k, kernel):
    mu = np.zeros((k))
    # if method == 'kmeans++':
    #     pass
    # else:
    rand = np.random.uniform(0, 10000, k)
    rand = rand.astype(int)
    print (rand)
    for i in range(k):
        mu[i] = kernel[rand[i], rand[i]]

    return mu


def plot_res(prd_res, name, clusters):
    prd_res = prd_res.reshape((100, 100))
    plt.imshow(prd_res)
    plt.savefig(os.path.join(f'./{clusters}/{name}.png'))

    return


if __name__ == "__main__":
    im1 = np.array(Image.open('./image1.png')).reshape((10000, 3))
    # im2 = np.array(Image.open('./image2.png')).reshape((10000, 3))
    
    Gamma_s, Gamma_c = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    K = [2, 3, 4, 5]
    for gamma_c in Gamma_c:
        for gamma_s in Gamma_s:
            kernel_im1 = kernel(im1, im1, gamma_s, gamma_c)
            # kernel_im2 = kernel(im2, im2, gamma_s, gamma_c)
            for k in K:
                prd_res = np.random.randint(k, size=10000)
                prd_prev = np.zeros(10000)
                iteration = 0
                while not np.array_equal(prd_res, prd_prev) and iteration < 60:
                    dist = np.tile(np.diag(kernel_im1), k).reshape(-1, k)
                    prd_prev = np.copy(prd_res)
                    print ('iteration:', iteration)
                    num_points = np.array([len(prd_res[prd_res==c]) for c in range(k)])
                    print (num_points)
                    for c in range(k):
                        # print (kernel_im1[:, prd_res==c].shape)
                        dist[:, c] -= 2/num_points[c] * np.sum(kernel_im1[:, prd_res==c], axis=1)
                        dist[:, c] += (1/num_points[c])**2 * np.sum(kernel_im1[prd_res==c][:, prd_res==c])
                    prd_res = np.argmin(dist, axis=1)
                    iteration += 1

                plot_res(prd_res, f"{gamma_c}_{gamma_s}_{k}", k)
