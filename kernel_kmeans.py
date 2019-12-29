import numpy as np
import os
from PIL import Image
from scipy.spatial import distance
import matplotlib.pyplot as plt


def make_kernel(x, y, gamma_s, gamma_c):
    dist_color = (distance.cdist(x, y, metric='euclidean'))**2
    coord_list = list()
    for i in range(100):
        for j in range(100):
            coord_list.append((i, j))
    coord_x = np.array(coord_list)
    coord_y = np.copy(coord_x)
    dist_spatial = (distance.cdist(coord_x, coord_y, metric='euclidean'))**2

    return np.dot(np.exp(-gamma_s * dist_spatial), np.exp(-gamma_c * dist_color))


def plot_res(prd_res, name, clusters, img):
    prd_res = prd_res.reshape((100, 100))
    plt.imshow(prd_res)
    plt.savefig(os.path.join(f'./tmp/IMG{img}/{clusters}/{name}.png'))

    return


def kernel_kmeans(img, order):
    for gamma_c in Gamma_c:
        for gamma_s in Gamma_s:
            kernel = make_kernel(img, img, gamma_s, gamma_c)
            for k in K:
                # tmp = list()
                # for i in range(k-1):
                #     tmp += [i]*int(10000/k)
                # tmp += [k-1]*(10000-len(tmp))
                # prd_res = np.array(tmp)
                prd_res = np.random.randint(k, size=10000)
                prd_prev = np.zeros(10000)
                iteration = 0
                while not np.array_equal(prd_res, prd_prev) and iteration < 60:
                    dist = np.tile(np.diag(kernel), k).reshape(-1, k)
                    prd_prev = np.copy(prd_res)
                    print ('iteration:', iteration)
                    num_points = np.array([len(prd_res[prd_res==c]) for c in range(k)])
                    print (num_points)
                    for c in range(k):
                        # print (kernel_im1[:, prd_res==c].shape)
                        dist[:, c] -= 2/num_points[c] * np.sum(kernel[:, prd_res==c], axis=1)
                        dist[:, c] += (1/num_points[c])**2 * np.sum(kernel[prd_res==c][:, prd_res==c])
                    prd_res = np.argmin(dist, axis=1)
                    iteration += 1

                    plot_res(prd_res, f"m2_{gamma_c}_{gamma_s}_{k}_ite{iteration}", k, order)

    return


if __name__ == "__main__":
    im1 = np.array(Image.open('./image1.png')).reshape((10000, 3)) / 255
    im2 = np.array(Image.open('./image2.png')).reshape((10000, 3)) / 255
    
    Gamma_s, Gamma_c = [0.1], [0.9]
    K = [2, 3, 4, 5]

    kernel_kmeans(im1, 1)
    kernel_kmeans(im2, 2)
