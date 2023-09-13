import ctypes
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from hyperbolicTSNE.tsne_barnes_hut_hyperbolic import _QuadTree
from hyperbolicTSNE.tsne_barnes_hut_hyperbolic import distance_py
# from sklearn.neighbors._quad_tree import _QuadTree

MACHINE_EPSILON = np.finfo(np.double).eps
np.random.seed(42)


matplotlib.rcParams['figure.dpi'] = 300


def get_random_point(k):

    # if k > 50:
    #     length = np.sqrt(np.random.uniform(0.4, 0.5))
    #     angle = np.pi * np.random.uniform(1.98, 2)
    #
    #     return np.array([length, angle])

    length = np.sqrt(np.random.uniform(0, 0.6))
    angle = np.pi * np.random.uniform(0, 2)

    return np.array([length, angle])


def cart_to_polar(p):
    length = np.sqrt(p[0] ** 2 + p[1] ** 2)
    angle = np.arctan2(p[1], p[0])

    angle = angle if angle > 0 else angle + 2 * np.pi
    return np.array([length, angle])


def cart_to_polar_2(p):
    radius = np.sqrt(p[0] * p[0] + p[1] * p[1])
    # Calculating angle (theta) in radian
    theta = np.arctan(p[1] / p[0])
    # Converting theta from radian to degree
    theta = 180 * theta / np.pi

    return np.array([radius, theta])


def cart2pol(p):
    x, y = p
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)


def plot_tree(sc_data):
    rticks, thetagrids = [], []

    n = 100
    # points = np.array([get_random_point(i) for i in range(n)])
    points = np.array([cart_to_polar(p) for p in sc_data])
    # cart_points = np.array([[r * np.cos(th), r * np.sin(th)] for r, th in points])
    cart_points = sc_data
    reverted = np.array([cart_to_polar(p) for p in cart_points])

    # print(np.allclose(points, reverted))

    pqt = _QuadTree(cart_points.shape[1], verbose=0)
    pqt.build_tree(cart_points)

    theta = 0.5
    random_idx = np.random.randint(points.shape[0])
    # random_idx = 158
    # print(random_idx)
    # print(cart_points[random_idx])

    idx, summary = pqt._py_summarize(cart_points[random_idx], cart_points, angle=theta)
    colormap = np.zeros(points.shape[0])
    colormap[random_idx] = 1

    sizes = []
    for j in range(idx // 4):
        size = summary[j * 4 + 2 + 1]
        sizes.append(int(size))

    # plt.hist(sizes)
    # plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(projection='polar')
    # ax.set_facecolor('wheat')
    ax.scatter(points[:, 1], points[:, 0], linewidth=0.5, marker='.', c='lightgray', zorder=-10, s=2)
    ax.scatter(points[random_idx, 1], points[random_idx, 0], marker='x', c='#E31A1C', zorder=10)

    # cmap = matplotlib.cm.get_cmap('viridis')

    # c = ax.scatter(points[:i, 1], points[:i, 0])

    barycenters = []

    summarized = set()

    for c_id, cell in enumerate(pqt.__getstate__()['cells']):
        # if cell['is_leaf'] == 0:
        #     continue

        if cell['parent'] in summarized:
            summarized.add(c_id)
            continue

        range_min = cell['min_bounds'][0]
        range_max = cell['max_bounds'][0]
        angle_min = cell['min_bounds'][1]
        angle_max = cell['max_bounds'][1]
        depth = cell['depth']
        barycenter = cell['barycenter']
        max_width = cell['squared_max_width']
        polar_barycenter = cart_to_polar(barycenter)

        # if depth != 4:
        #     continue

        # TODO: cell size > 1
        h_dist = distance_py(np.array(cart_points[random_idx], dtype=ctypes.c_double),
                                     np.array(barycenter, dtype=ctypes.c_double)) ** 2

        if h_dist < MACHINE_EPSILON:
            continue
        ratio = (max_width / h_dist)
        is_summ = ratio < (theta ** 2)

        if is_summ:
            summarized.add(c_id)
        else:
            continue

        # c = 'r' if c_id in sizes else 'g'
        # c = 'red' if is_summ else 'lightseagreen'
        # c = cmap(ratio * (1 / theta ** 2))
        c = '#0173B2'
        s = '-'
        w = 0.5

        ax.scatter([polar_barycenter[1]], [polar_barycenter[0]], linewidth=0.5, marker='.', c="#253494", zorder=1, s=5)
        ax.plot(np.linspace(angle_min, angle_max, 100), np.ones(100) * range_min, color=c, linestyle=s, linewidth=w, antialiased=True, zorder=-1)
        ax.plot(np.linspace(angle_min, angle_max, 100), np.ones(100) * range_max, color=c, linestyle=s, linewidth=w, antialiased=True, zorder=-1)
        ax.plot(np.ones(100) * angle_min, np.linspace(range_min, range_max, 100), color=c, linestyle=s, linewidth=w, antialiased=True, zorder=-1)
        ax.plot(np.ones(100) * angle_max, np.linspace(range_min, range_max, 100), color=c, linestyle=s, linewidth=w, antialiased=True, zorder=-1)

    ax.set_rmax(1)
    ax.set_rticks(rticks)  # Less radial ticksz
    # ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
    ax.set_thetagrids(thetagrids)
    ax.grid(True)

    # ax.set_title("A line plot on a polar axis", va='bottom')
    # ax.set_title("Choose based on aspect ratio")

    # plt.gca().set_axis_off()
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
    #                     hspace=0, wspace=0)
    # plt.margins(0, 0)
    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
    # plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.tight_layout()
    # plt.gca().set_aspect('equal')

    # plt.show()
    plt.savefig("tree.pdf")
    print("done")


if __name__ == '__main__':
    # fast = False
    # plot_ee = False
    # scatter_data = []
    # for subdir, dirs, files in os.walk("/temp/poincare_bh"):
    #     for fi, file in enumerate(sorted(files, key=lambda x: int(x.split(", ")[0]))):
    #         root, ext = os.path.splitext(file)
    #         if ext == ".csv" and fi == 999:
    #             total_file = subdir.replace("\\", "/") + "/" + file
    #             d = np.genfromtxt(total_file, delimiter=',')
    #             scatter_data.append(d)
    #
    # plot_tree(scatter_data[0])

    plot_tree(np.load("final_embedding.npy"))
    # plot_tree(np.genfromtxt("c_bh.csv", delimiter=','))
