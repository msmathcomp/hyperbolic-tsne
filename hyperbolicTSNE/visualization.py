import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm


def plot_poincare(points, labels=None):
    fig, ax = plt.subplots()
    ax.scatter(points[:, 0], points[:, 1],
               c=labels,
               marker=".")
    ax.add_patch(plt.Circle((0, 0), radius=1, edgecolor="b", facecolor="None"))
    ax.axis("square")
    return fig


def animate(log_dict, labels, file_name, fast=False, is_hyperbolic=True, plot_ee=False, first_frame=None):
    scatter_data = [] if first_frame is None else [(-1, first_frame)]
    for subdir, dirs, files in os.walk(log_dict["log_path"]):
        for fi, file in enumerate(sorted(files, key=lambda x: int(x.split(", ")[0]))):
            root, ext = os.path.splitext(file)
            if ext == ".csv":
                total_file = subdir.replace("\\", "/") + "/" + file
                if (not fast or fi % 10 == 0) and (plot_ee or subdir.split("/")[-1].endswith("1")):
                    data = np.genfromtxt(total_file, delimiter=',')
                    scatter_data.append((str(fi), data))

    fig, ax = plt.subplots()

    _, data = scatter_data[0 if is_hyperbolic else -1]

    scatter = ax.scatter(data[:, 0], data[:, 1], c=labels, marker=".", linewidth=0.5, s=20)

    if is_hyperbolic:
        uc = plt.Circle((0, 0), radius=1, edgecolor="b", facecolor="None")
        ax.add_patch(uc)

    ax.axis("square")

    print("Animation being saved to: " + file_name)

    pbar = tqdm(desc="Animating: ", total=len(scatter_data))

    def update(i):
        pbar.update()

        sd = scatter_data[i]

        scatter.set_offsets(sd[1])

        ax.set_title(f'Scatter (epoch {sd[0]})')
        return scatter,

    anim = FuncAnimation(fig, update, frames=len(scatter_data), interval=50, blit=True, save_count=50)
    anim.save(file_name)

    plt.clf()
    pbar.close()
