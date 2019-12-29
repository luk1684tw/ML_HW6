import numpy as np
import os
from PIL import Image
from scipy.spatial import distance
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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


def kmeans(U, k, method, img):
    tmp = list()
    for i in range(k-1):
        tmp += [i]*int(10000/k)
    tmp += [k-1]*(10000-len(tmp))
    prd_result = np.array(tmp)
    # prd_result = np.random.randint(k, size=10000)
    center = np.zeros((k, k))
    prev_center = np.ones((k, k))
    iteration = 0
    plot_res(prd_result, f'{method}_M1_{k}_{iteration}', k, img)
    while not np.array_equal(center, prev_center) and iteration < 100:
        prev_center = np.copy(center)
        for i in range(k):
            center[i] = np.mean(U[prd_result == i], axis=0)
        print ([len(prd_result[prd_result == i]) for i in range(k)])
        prd_result = np.argmin(distance.cdist(U, center), axis=1)
        iteration += 1
        plot_res(prd_result, f'{method}_M1_{k}_{iteration}', k, img)
    return prd_result


def spectual_clustering(img, k, method, img_count):
    kernel_img = kernel(img, img, gamma_s, gamma_c)
    D = np.diag(np.sum(kernel_img, axis=1))

    if method == 'normal':
        D_prime = np.linalg.inv(np.sqrt(D))
        L = np.dot(D_prime, np.dot(D - kernel_img, D_prime))
    else:
        L = D - kernel_img

    print ('Start eigen decomposition')
    eig_val, eig_vec = np.linalg.eig(L)
    eigen_dict = dict(zip(eig_val, eig_vec.T))
    U = np.zeros((10000, k))
    for i, val in enumerate(np.sort(eig_val)[1:]):
        print (i, val)
        if i == k:
            break
        else:
            U[:, i] = eigen_dict[val]
    for i in range(len(U)):
        U[i] /= np.linalg.norm(U[i])

    prd_result = kmeans(U, k, method, img_count)
    # plot_data(U, f'{method}_{k}_{img_count}.png', prd_result, k)
    return


def plot_data(U, name, prd_result, k):
    fig = plt.figure()
    ax = Axes3D(fig)
    if k == 2:
        for i in range(k):
            cluster_data = U[prd_result == i]
            ax.scatter(cluster_data[:, 0], cluster_data[:, 1], c=colors[i])
    else:
        for i in range(k):
            cluster_data = U[prd_result == i]
            ax.scatter(cluster_data[:, 0], cluster_data[:, 1], cluster_data[:, 2], c=colors[i])
    
    plt.savefig(f'./{name}')
    return


def plot_res(prd_res, name, clusters, img):
    plt.close()
    prd_res = prd_res.reshape((100, 100))
    plt.imshow(prd_res)
    plt.savefig(os.path.join(f'./tmp/IMG{img}/sp/{clusters}/{name}.png'))

    return


if __name__ == "__main__":
    im1 = np.array(Image.open('./image1.png')).reshape((10000, 3)) / 255
    im2 = np.array(Image.open('./image2.png')).reshape((10000, 3)) / 255

    colors = ['b', 'g', 'r', 'c', 'm']
    gamma_s, gamma_c = 0.1, 0.9
    K = [2, 3, 4, 5]
    for k in K:
        spectual_clustering(im1, k, 'normal', 1)
        spectual_clustering(im1, k, 'ratio', 1)

        spectual_clustering(im2, k, 'normal', 2)
        spectual_clustering(im2, k, 'ratio', 2)
