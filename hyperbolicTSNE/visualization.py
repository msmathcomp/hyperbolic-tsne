""" Visualization functions to generating embedding plots in the paper.
"""
import os
from collections import defaultdict
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

from hyperbolicTSNE import Datasets


def plot_poincare(points, labels=None):
    fig, ax = plt.subplots()
    ax.scatter(points[:, 0], points[:, 1],
               c=labels,
               marker=".",
               cmap="tab10")
    ax.add_patch(plt.Circle((0, 0), radius=1, edgecolor="b", facecolor="None"))
    ax.axis("square")
    return fig


color_dict = defaultdict(lambda: "tab10")

color_dict[Datasets.C_ELEGANS] = {
    'ABarpaaa_lineage': '#91003f',  # embryonic lineage
    'Germline': '#7f2704',
    # Somatic gonad precursor cell
    'Z1_Z4': '#800026',

    # Two embryonic hypodermal cells that may provide a scaffold for the early organization of ventral bodywall muscles
    'XXX': '#fb8072',

    'Ciliated_amphid_neuron': '#c51b8a', 'Ciliated_non_amphid_neuron': '#fa9fb5',

    # immune
    'Coelomocyte': '#ffff33', 'T': '#54278f',

    # Exceratory
    'Excretory_cell': '#004529',
    'Excretory_cell_parent': '#006837',
    'Excretory_duct_and_pore': '#238443',
    'Parent_of_exc_duct_pore_DB_1_3': '#41ab5d',
    'Excretory_gland': '#78c679',
    'Parent_of_exc_gland_AVK': '#addd8e',
    'Rectal_cell': '#d9f0a3',
    'Rectal_gland': '#f7fcb9',
    'Intestine': '#7fcdbb',

    # esophagus, crop, gizzard (usually) and intestine
    'Pharyngeal_gland': '#fed976',
    'Pharyngeal_intestinal_valve': '#feb24c',
    'Pharyngeal_marginal_cell': '#fd8d3c',
    'Pharyngeal_muscle': '#fc4e2a',
    'Pharyngeal_neuron': '#e31a1c',

    # hypodermis (epithelial)
    'Parent_of_hyp1V_and_ant_arc_V': '#a8ddb5',
    'hyp1V_and_ant_arc_V': '#ccebc5',
    'Hypodermis': '#253494',
    'Seam_cell': '#225ea8',
    'Arcade_cell': '#1d91c0',

    # set of six cells that form a thin cylindrical sheet between pharynx and ring neuropile
    'GLR': '#1f78b4',

    # Glia, also called glial cells or neuroglia, are non-neuronal cells in the central nervous system
    'Glia': '#377eb8',

    # head mesodermal cell: the middle layer of cells or tissues of an embryo
    'Body_wall_muscle': '#9e9ac8',
    'hmc': '#54278f',
    'hmc_and_homolog': '#02818a',
    'hmc_homolog': '#bcbddc',
    'Intestinal_and_rectal_muscle': '#41b6c4',
    # Postembryonic mesoblast: the mesoderm of an embryo in its earliest stages.
    'M_cell': '#3f007d',

    # pharyngeal gland cel
    'G2_and_W_blasts': '#abdda4',

    'unannotated': '#969696',
    'not provided': '#969696'
}

color_dict[Datasets.PLANARIA] = {'neoblast 1': '#CCCCCC',
                       'neoblast 2': '#7f7f7f',
                       'neoblast 3': '#E6E6E6',
                       'neoblast 4': '#D6D6D6',
                       'neoblast 5': '#C7C7C7',
                       'neoblast 6': '#B8B8B8',
                       'neoblast 7': '#A8A8A8',
                       'neoblast 8': '#999999',
                       'neoblast 9': '#8A8A8A',
                       'neoblast 10': '#7A7A7A',
                       'neoblast 11': '#6B6B6B',
                       'neoblast 12': '#5C5C5C',
                       'neoblast 13': '#4D4D4D',
                       'epidermis DVb neoblast': 'lightsteelblue',
                       'pharynx cell type progenitors': 'slategray',
                       'spp-11+ neurons': '#CC4C02',
                       'npp-18+ neurons': '#EC7014',
                       'otf+ cells 1': '#993404',
                       'ChAT neurons 1': '#FEC44F',
                       'neural progenitors': '#FFF7BC',
                       'otf+ cells 2': '#662506',
                       'cav-1+ neurons': '#eec900',
                       'GABA neurons': '#FEE391',
                       'ChAT neurons 2': '#FE9929',
                       'muscle body': 'firebrick',
                       'muscle pharynx': '#CD5C5C',
                       'muscle progenitors': '#FF6347',
                       'secretory 1': 'mediumpurple',
                       'secretory 3': 'purple',
                       'secretory 4': '#CBC9E2',
                       'secretory 2': '#551a8b',
                       'early epidermal progenitors': '#9ECAE1',
                       'epidermal neoblasts': '#C6DBEF',
                       'activated early epidermal progenitors': 'lightblue',
                       'late epidermal progenitors 2': '#4292C6',
                       'late epidermal progenitors 1': '#6BAED6',
                       'epidermis DVb': 'dodgerblue',
                       'epidermis': '#2171B5',
                       'pharynx cell type': 'royalblue',
                       'protonephridia': 'pink',
                       'ldlrr-1+ parenchymal cells': '#d02090',
                       'phagocytes': 'forestgreen',
                       'aqp+ parenchymal cells': '#cd96cd',
                       'pigment': '#cd6889',
                       'pgrn+ parenchymal cells': 'mediumorchid',
                       'psap+ parenchymal cells': 'deeppink',
                       'glia': '#cd69c9',
                       'goblet cells': 'yellow',
                       'parenchymal progenitors': 'hotpink',
                       'psd+ cells': 'darkolivegreen',
                       'gut progenitors': 'limegreen',
                       'branchNe': '#4292c6', 'neutrophil': '#08306b',
                       'branchMo': '#9e9ac8', 'monocyte': '#54278f',
                       'branchEr': '#fc9272', 'erythrocyt': '#cb181d',
                       'megakaryoc': '#006d2c', 'branchMe': '#74c476',
                       'proghead': '#525252',
                       'interpolation': '#525252',
                       'Eryth': '#1F77B4',
                       'Gran': '#FF7F0E',
                       'HSPC-1': '#2CA02C',
                       'HSPC-2': '#D62728',
                       'MDP': '#9467BD',
                       'Meg': '#8C564B',
                       'Mono': '#E377C2',
                       'Multi-Lin': '#BCBD22',
                       'Myelocyte': '#17BECF',
                       '12Baso': '#0570b0', '13Baso': '#034e7b',
                       '11DC': '#ffff33',
                       '18Eos': '#2CA02C',
                       '1Ery': '#fed976', '2Ery': '#feb24c', '3Ery': '#fd8d3c', '4Ery': '#fc4e2a', '5Ery': '#e31a1c',
                       '6Ery': '#b10026',
                       '9GMP': '#999999', '10GMP': '#4d4d4d',
                       '19Lymph': '#35978f',
                       '7MEP': '#E377C2',
                       '8Mk': '#BCBD22',
                       '14Mo': '#4eb3d3', '15Mo': '#7bccc4',
                       '16Neu': '#6a51a3', '17Neu': '#3f007d',
                       'root': '#000000'}

color_dict[Datasets.MYELOID] = {'branchNe': '#4292c6', 'neutrophil': '#08306b',
                    'branchMo': '#9e9ac8', 'monocyte': '#54278f',
                    'branchEr': '#fc9272', 'erythrocyt': '#cb181d',
                    'megakaryoc': '#006d2c', 'branchMe': '#74c476',
                    'proghead': '#525252', 'root': '#000000',
                    'interpolation': '#bdbdbd'}

legend_bbox_dict = defaultdict(lambda: (1., 1.))
legend_bbox_dict[Datasets.MNIST] = (1., 0.5)
legend_bbox_dict[Datasets.PLANARIA] = (1.5, 0.5)
legend_bbox_dict[Datasets.MYELOID] = (1.1, 0.5)
legend_bbox_dict[Datasets.C_ELEGANS] = (1.3, 0.5)


def myeloid_labels(data_home = Path("../datasets/")):
    full_path = Path.joinpath(data_home, "myeloid-progenitors")

    return np.loadtxt(str(Path.joinpath(full_path, "MyeloidProgenitors.csv")), delimiter=",", skiprows=1, usecols=11,
                      dtype=str)


def c_elegans_labels(data_home = Path("../datasets/")):

    full_path = Path.joinpath(data_home, "c_elegans")

    ad_obj = ad.read_h5ad(str(Path.joinpath(full_path, "packer2019.h5ad")))

    return np.array(ad_obj.obs.cell_type)


def planaria_labels(data_home = Path("../datasets/")):
    full_path = Path.joinpath(data_home, "planaria")

    return np.loadtxt(str(Path.joinpath(full_path, "R_annotation.txt")), delimiter=",", dtype=str)


def mnist_labels(data_home = Path("../datasets/")):
    full_path = Path.joinpath(data_home, 'mnist')

    labels_path_train = Path.joinpath(full_path, 'train-labels-idx1-ubyte.gz')
    labels_path_test = Path.joinpath(full_path, 't10k-labels-idx1-ubyte.gz')

    labels_arr = []

    import gzip
    with gzip.open(labels_path_train, 'rb') as lbpath:
        br = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)
        labels_arr.append(br)

    with gzip.open(labels_path_test, 'rb') as lbpath:
        br = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)
        labels_arr.append(br)

    labels = np.concatenate(labels_arr, axis=0)

    return list(map(str, labels))


# labels_dict = defaultdict(lambda: None)
# labels_dict[Datasets.MNIST] = mnist_labels()
# labels_dict[Datasets.PLANARIA] = planaria_labels()
# labels_dict[Datasets.C_ELEGANS] = c_elegans_labels()
# labels_dict[Datasets.MYELOID] = myeloid_labels()


def save_poincare_teaser(points, file_name, str_labels=None, dataset=None, save_fig_kwargs=dict()):
    labels_dict = dict()

    df = pd.DataFrame({"x": points[:, 0], "y": points[:, 1]})

    fig, ax = plt.subplots()

    point_size = 2
    font_size = 5
    alpha = 1.0

    if str_labels is None:
        str_labels = labels_dict[dataset]

    sns.scatterplot(data=df, x="x", y="y",
                    hue=str_labels, hue_order=np.unique(str_labels), palette=color_dict[dataset],
                    alpha=alpha, edgecolor="none", ax=ax, s=point_size)

    lgd = ax.legend(fontsize=font_size, loc='right', bbox_to_anchor=legend_bbox_dict[dataset], facecolor='white',
                    frameon=False,
                    ncol=2 if dataset is Datasets.PLANARIA else 1)

    circle = plt.Circle((0, 0), radius=1, fc='none', color='black')
    ax.add_patch(circle)
    ax.plot(0, 0, '.', c=(0, 0, 0), ms=4)

    fig.tight_layout()
    ax.axis('off')
    ax.axis('equal')

    ax.set_ylim([-1.01, 1.01])
    ax.set_xlim([-1.01, 1.01])

    plt.tight_layout()

    plt.savefig(file_name, bbox_extra_artists=(lgd,), bbox_inches='tight', **save_fig_kwargs)


def plot_poincare_zoomed(points, labels=None):
    fig, ax = plt.subplots()
    ax.scatter(points[:, 0], points[:, 1],
               c=labels,
               marker=".",
               cmap="tab10")
    ax.axis("square")
    ax.add_patch(plt.Circle((0, 0), radius=1, edgecolor="b", facecolor="None"))
    return fig


def animate(log_dict, labels, file_name, fast=False, is_hyperbolic=True, plot_ee=False, first_frame=None):
    scatter_data = [] if first_frame is None else [("-1", first_frame)]
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

    scatter = ax.scatter(data[:, 0], data[:, 1], c=labels, marker=".", linewidth=0.5, s=20, cmap="tab10")

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
    plt.close()
    pbar.close()


def plot_c_elegans_teaser():

    points = np.load("teaser_files/c_elegans_embedding.npy")
    labels = np.load("teaser_files/c_elegans_labels.npy", allow_pickle=True)

    df = pd.DataFrame({"x": points[:, 0], "y": points[:, 1]})

    fig, ax = plt.subplots()

    point_size = 2
    font_size = 5
    alpha = 1.0

    sns.scatterplot(
        data=df,
        x="x",
        y="y",
        hue=labels,
        hue_order=np.unique(labels),
        palette=color_dict[Datasets.C_ELEGANS],
        alpha=alpha,
        edgecolor="none",
        ax=ax,
        s=point_size
    )

    lgd = ax.legend(
        fontsize=font_size,
        loc='right',
        bbox_to_anchor=legend_bbox_dict[Datasets.C_ELEGANS],
        facecolor='white',
        frameon=False,
        ncol=1
    )

    circle = plt.Circle((0, 0), radius=1, fc='none', color='black')
    ax.add_patch(circle)
    ax.plot(0, 0, '.', c=(0, 0, 0), ms=4)

    fig.tight_layout()
    ax.axis('off')
    ax.axis('equal')

    ax.set_ylim([-1.01, 1.01])
    ax.set_xlim([-1.01, 1.01])

    plt.tight_layout()

    plt.savefig(
        "teaser_files/c_elegans_embedding.png", bbox_extra_artists=(lgd,), bbox_inches='tight'
    )
