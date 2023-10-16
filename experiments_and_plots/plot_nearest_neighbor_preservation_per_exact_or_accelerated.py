"""
This experiment runs the hyperbolic tsne code changing several parameters:
- dataset: [LUKK, MYELOID8000, PLANARIA, MNIST, C_ELEGANS, WORDNET]
- tsne_type: [accelerated, exact]
For each configuration combination, it saves the embedding coordinates, a plot of the embedding, and
timing data for the iterations.
If a run does not finish, the results are not saved.
The code only computes the runs that do not have a folder.
"""
###########
# IMPORTS #
###########

from pathlib import Path
import numpy as np
from scipy.sparse import load_npz
import matplotlib.pyplot as plt

from hyperbolicTSNE.quality_evaluation_ import hyperbolic_nearest_neighbor_preservation
from hyperbolicTSNE import Datasets, load_data

#################################
# GENERAL EXPERIMENT PARAMETERS #
#################################

BASE_DIR = Path("../results/full_size_one_run")
DATASETS_DIR = "../datasets"  # directory to read the data from

# Constants
NEIGHBORHOOD_SIZE = 100  # Neighborhood size to grab the k_max many neighbors from
K_START = 1  # Lowest point in the precision/recall curve
K_MAX = 30  # Highest point in the precision/recall curve
PERP = 30  # perplexity value to be used throughout the experiments
hd_params = {"perplexity": PERP}

# Iterate over all results,
for dataset_dir in BASE_DIR.glob("*"):
    if dataset_dir.is_dir():
        dataset_name = dataset_dir.stem
        dataX = load_data(Datasets[dataset_name], data_home=DATASETS_DIR, to_return="X", hd_params=hd_params)

        fig, ax = plt.subplots()
        ax.set_title(dataset_name)

        for size_dir in dataset_dir.glob("*"):
            if size_dir.is_dir():
                for config_dir in size_dir.glob("*"):
                    if config_dir.is_dir():
                        for run_dir in config_dir.glob("*"):
                            if run_dir.is_dir():
                                print(f"[NNP plot] Processing {run_dir}.")
                                subset_idx = np.load(run_dir.joinpath("subset_idx.npy"), allow_pickle=True)
                                dataX_sample = dataX[subset_idx]

                                D_path = list(run_dir.glob("D.np*"))[0]
                                sparse_D = D_path.suffix == ".npz"
                                if sparse_D:
                                    D_X = load_npz(D_path)
                                else:
                                    D_X = np.load(D_path, allow_pickle=True)[()]

                                dataY = np.load(run_dir.joinpath("final_embedding.npy"), allow_pickle=True)

                                thresholds, precisions, recalls, true_positives = \
                                    hyperbolic_nearest_neighbor_preservation(
                                        dataX_sample,
                                        dataY,
                                        K_START,
                                        K_MAX,
                                        D_X,
                                        False,
                                        False,
                                        False,
                                        "full",
                                        NEIGHBORHOOD_SIZE
                                    )

                                # save results inside folder
                                np.save(run_dir.joinpath("thresholds.npy"), thresholds)
                                np.save(run_dir.joinpath("precisions.npy"), precisions)
                                np.save(run_dir.joinpath("recalls.npy"), recalls)
                                np.save(run_dir.joinpath("true_positives.npy"), true_positives)

                                # Add points to plot
                                if config_dir.name == 'configuration_0':
                                    ax.plot(precisions, recalls, label="accelerated")
                                else:
                                    ax.plot(precisions, recalls, label="exact")

        ax.set_xlabel("Precision")
        ax.set_ylabel("Recall")
        ax.legend()
        fig.savefig(dataset_dir.joinpath(f"{dataset_name}_prec-vs-rec.png"))
        plt.close(fig)
