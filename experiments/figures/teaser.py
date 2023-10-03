from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

from hyperbolicTSNE import Datasets
from hyperbolicTSNE.visualization import save_poincare_teaser, mnist_labels, planaria_labels, c_elegans_labels, myeloid_labels

base_dir = Path("/Users/chadepl/Downloads/exp_grid/")
dataset_name = ["MNIST", "PLANARIA", "C_ELEGANS", "MYELOID"][2]
version = ["exact", "accelerated"][1]
config = 38 if version == "exact" else 36
run = 0
file_appendix = f"configuration_{config}/run_{run}"

plot_details = dict(
    MNIST=dict(file_name=f"MNIST/size_70000/{file_appendix}", labels=mnist_labels(data_home=Path("../../datasets")), dataset=Datasets.MNIST),
    PLANARIA=dict(file_name=f"PLANARIA/size_21612/{file_appendix}", labels=planaria_labels(data_home=Path("../../datasets")), dataset=Datasets.PLANARIA),
    C_ELEGANS=dict(file_name=f"C_ELEGANS/size_89701/{file_appendix}", labels=c_elegans_labels(data_home=Path("../../datasets")), dataset=Datasets.C_ELEGANS),
    MYELOID=dict(file_name=f"MYELOID/size_70000/{file_appendix}", labels=myeloid_labels(data_home=Path("../../datasets")), dataset=Datasets.MYELOID)
)[dataset_name]


labels_dict = defaultdict(lambda: None)

file_name = base_dir.joinpath(plot_details["file_name"])
subset_idx = np.load(file_name.joinpath("subset_idx.npy"), allow_pickle=True)
points = np.load(file_name.joinpath("final_embedding.npy"), allow_pickle=True)
labels = plot_details["labels"]

save_poincare_teaser(points=points, file_name=f"{dataset_name}_{version}_run-{run}.png", str_labels=labels, dataset=plot_details["dataset"], save_fig_kwargs=dict(dpi=500))