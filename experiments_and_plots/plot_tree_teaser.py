"""
This scripts takes an embedding of the C.Elegans data set and plots a polar quad tree on top of it.
"""
###########
# IMPORTS #
###########
import ctypes
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from hyperbolicTSNE.hyperbolic_barnes_hut.tsne import _QuadTree
from hyperbolicTSNE.hyperbolic_barnes_hut.tsne import distance_py

##############
# PLOT SETUP #
##############
MACHINE_EPSILON = np.finfo(np.double).eps
np.random.seed(594507)
matplotlib.rcParams['figure.dpi'] = 300
c = '#0173B2'  # Color for the tree
s = '-'  # style of the tree lines
w = 0.5  # width of the tree lines


##################
# Helper Methods #
##################
def get_random_point():
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
    return rho, phi


def plot_tree(sc_data):
    rticks, thetagrids = [], []

    # points = np.array([get_random_point(i) for i in range(n)])
    points = np.array([cart_to_polar(p) for p in sc_data])
    # cart_points = np.array([[r * np.cos(th), r * np.sin(th)] for r, th in points])
    cart_points = sc_data

    pqt = _QuadTree(cart_points.shape[1], verbose=0)
    pqt.build_tree(cart_points)

    theta = 0.5
    random_idx = np.random.randint(points.shape[0])

    idx, summary = pqt._py_summarize(cart_points[random_idx], cart_points, angle=theta)
    colormap = np.zeros(points.shape[0])
    colormap[random_idx] = 1

    sizes = []
    for j in range(idx // 4):
        size = summary[j * 4 + 2 + 1]
        sizes.append(int(size))

    fig = plt.figure()
    ax = fig.add_subplot(projection='polar')
    ax.scatter(points[:, 1], points[:, 0], linewidth=0.5, marker='.', c='lightgray', zorder=-10, s=2)
    ax.scatter(points[random_idx, 1], points[random_idx, 0], marker='x', c='#E31A1C', zorder=10)

    summarized = set()

    for c_id, cell in enumerate(pqt.__getstate__()['cells']):

        if cell['parent'] in summarized:
            summarized.add(c_id)
            continue

        range_min = cell['min_bounds'][0]
        range_max = cell['max_bounds'][0]
        angle_min = cell['min_bounds'][1]
        angle_max = cell['max_bounds'][1]
        barycenter = cell['barycenter']
        max_width = cell['squared_max_width']
        polar_barycenter = cart_to_polar(barycenter)

        h_dist = distance_py(
            np.array(cart_points[random_idx], dtype=ctypes.c_double), np.array(barycenter, dtype=ctypes.c_double)
        ) ** 2

        if h_dist < MACHINE_EPSILON:
            continue
        ratio = (max_width / h_dist)
        is_summ = ratio < (theta ** 2)

        if is_summ:
            summarized.add(c_id)
        else:
            continue

        ax.scatter([polar_barycenter[1]], [polar_barycenter[0]], linewidth=0.5, marker='.', c="#253494", zorder=1, s=5)
        ax.plot(
            np.linspace(angle_min, angle_max, 100),
            np.ones(100) * range_min,
            color=c,
            linestyle=s,
            linewidth=w,
            antialiased=True,
            zorder=-1
        )
        ax.plot(
            np.linspace(angle_min, angle_max, 100),
            np.ones(100) * range_max,
            color=c,
            linestyle=s,
            linewidth=w,
            antialiased=True,
            zorder=-1
        )
        ax.plot(
            np.ones(100) * angle_min,
            np.linspace(range_min, range_max, 100),
            color=c,
            linestyle=s,
            linewidth=w,
            antialiased=True,
            zorder=-1
        )
        ax.plot(
            np.ones(100) * angle_max,
            np.linspace(range_min, range_max, 100),
            color=c,
            linestyle=s,
            linewidth=w,
            antialiased=True,
            zorder=-1
        )

    ax.set_rmax(1)
    ax.set_rticks(rticks)  # Less radial ticksz
    ax.set_thetagrids(thetagrids)
    ax.grid(True)

    plt.tight_layout()
    plt.savefig("c_elegans.png", dpi=500)
    print("done")


if __name__ == '__main__':
    # TODO Add the path the embedding file
    plot_tree(np.load("c_elegans.npy"))
