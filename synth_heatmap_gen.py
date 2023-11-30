import argparse
import numpy as np
import pickle
import math
import matplotlib.pyplot as plt
import os
import shutil
from sklearn.metrics.pairwise import rbf_kernel
import csv

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Module for generating a synthetic elevation heatmap dataset',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--n', default=100, type=int, help='Size of the dataset')
    parser.add_argument('--d1', default=128, type=int, help='Width of the images')
    parser.add_argument('--d2', default=128, type=int, help='Height of the images')
    parser.add_argument('--L', default=100, type=int, help='Maximum number of random pins')
    parser.add_argument('--step', default=10, type=int, help='Distance between mesh grid pins')
    parser.add_argument('--seed', default=5, type=int, help='Seed for randomization')
    parser.add_argument('--pins_random', default=False, type=bool,
                        help='Pin selection mode. Options: random when True, mesh grid otherwise')
    parser.add_argument('--sigma', default=10, type=int, help='sigma value of the RBF kernel')
    parser.add_argument('--a', default=3, type=float, help='coefficient of the mean')
    parser.add_argument('--b', default=0.5, type=float, help='offset of the mean')

    args = parser.parse_args()
    n = args.n
    d1 = args.d1
    d2 = args.d2
    L = args.L
    step = args.step
    seed = args.seed
    pins_random = args.pins_random
    sigma = args.sigma
    np.random.seed(seed)
    # mean and covariance functions
    a = args.a
    b = args.b  # smaller b - small bias

    if pins_random:
        data_folder = f"data_{n}images_{d1}by{d2}pixels_upto{L}pins_{seed}seed/"
    else:
        data_folder = f"data_{n}images_{d1}by{d2}pixels_{step}_distanced_grid_pins_{seed}seed/"

    # delete the data folder if it already exists
    if os.path.exists(data_folder) and os.path.isdir(data_folder):
        shutil.rmtree(data_folder)

    # create a new data folder
    os.makedirs(f"{data_folder}images")

    # nonlinear mean (quadratic)
    def quad_mean(p_1, p_2, z):
        return a * z[(p_1, p_2)] ** 2 + b

    # linear mean
    def mean(p_1, p_2, z):
        return a * z[(p_1, p_2)] + b

    # RBF kernel with a small width
    gamma = (1 / (sigma ** 2))

    def cov(p_1, p_2, gamma):
        """
            K(p_1, p_2) = exp(-gamma ||p_1-p_2||^2)
        """
        return rbf_kernel(p_1, p_2, gamma=gamma)  # this should only depend on p1 and p2 - enforce continuity

    def show_pins(image, pins, outputs):
        """Show image with pins"""
        plt.imshow(image, extent=[0, d1, 0, d2])
        plt.scatter(pins[:, 0], pins[:, 1], s=10, marker='.', c='r')

    def reformat(arr):
        return [(i[0][0], i[1][0]) for i in arr]

    header = ['image', 'pins', 'outputs']
    with open(f"{data_folder}/pins.csv", 'w') as file:
        writer = csv.writer(file)
        writer.writerow(header)

    for i in range(n):
        # these will generate the images
        c = np.random.randint(0, 10, size=4)
        func_list = [np.sin, np.cos]
        funcs = np.random.choice(func_list, 3)
        p1 = np.linspace(0, 5, d1)
        p2 = np.linspace(0, 5, d2)
        P1, P2 = np.meshgrid(p1, p2, indexing='ij')

        def f(x, y):
            return c[0] * funcs[0](x) ** c[1] + c[2] * funcs[1](c[3] + y * x) * funcs[2](x)

        z = f(P1, P2)

        # indexing check
        # for i, vali in enumerate(p1):
        #     for j, valj in enumerate(p2):
        #         if z[i, j] != f(vali, valj):
        #             print("not equal!")

        fig = plt.figure(figsize=(7, 7))
        plt.imshow(z, origin='lower')  # http://brantr.github.io/blog/matplotlib-imshow-orientation/
        plt.axis('off')

        img_name = f"{i}.png"
        fig.savefig(f'{data_folder}/images/{img_name}', bbox_inches='tight', pad_inches=0)
        plt.close()

        if pins_random:
            # choose how many pins to sample
            L_i = np.random.randint(1, L)

            # coordinates of the random pins
            p_1i = np.random.randint(d1, size=(L_i, 1))
            p_2i = np.random.randint(d2, size=(L_i, 1))
            # p_i = np.concatenate((p_1i, p_2i), axis=1) # coordinates of the pins
            p_i = list(zip(p_1i.tolist(), p_2i.tolist()))  # coordinates of the pins
        else:
            # coordinates of the mesh pins
            p_1i = np.arange(0, d1, step).reshape(-1, 1)
            p_2i = np.arange(0, d2, step).reshape(-1, 1)
            p_i = [(x.tolist(), y.tolist()) for x in p_1i for y in p_2i]  # coordinates of the pins

        # print(p_i)
        # print(np.asarray(p_i)[:, 1])
        mean_vec = mean(np.asarray(p_i)[:, 1], np.asarray(p_i)[:, 0], z)
        # print(mean_vec)
        # print(np.shape(mean_vec.reshape(-1)))

        cov_mat = cov(np.asarray(p_i)[:, 1], np.asarray(p_i)[:, 0], gamma)

        # this should correct the potential negative eigenvalues due to floating errors
        min_eig = np.min(np.real(np.linalg.eigvals(cov_mat)))
        if min_eig < 0:
            cov_mat -= 10 * min_eig * np.eye(*cov_mat.shape)

        # https://numpy.org/doc/stable/reference/random/generated/numpy.random.multivariate_normal.html
        # .reshape(-1) converts a 2D matrix to a 1D vector (e.g. 100x1 --> 1x100)
        y_i = np.random.default_rng().multivariate_normal(mean_vec.reshape(-1), cov_mat)

        plt.scatter(np.asarray(p_i)[:, 0], np.asarray(p_i)[:, 1], s=10, marker='.', c='r')
        p_i = reformat(p_i)
        for j in range(len(p_i)):
            plt.annotate(f'{y_i[j]:.2f}', p_i[j])

        with open(f"{data_folder}/pins.csv", 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([img_name, p_i, y_i.tolist()])