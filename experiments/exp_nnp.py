"""
This experiment evaluates the performance of hyperbolic tSNE by assessing how many nearest neighbors it preserved.
We want to produce similar results as: 
Which is based on the nearest neighbor preservation metric introduced at:
Inputs:
 - 
Outputs:
 - 
"""

# Sketch code:
# For each dataset (a directory in results folder)
# For each size (a directory within the dataset directory)
# For each configuration (a directory within the size directory)
# For each run 
# List all P matrices and final embeddings
# For each 
# Obtain: thresholds, precisions, recalls, num_true_positives

from pathlib import Path
import numpy as np
from scipy.sparse import load_npz
import matplotlib.pyplot as plt

from hyperbolicTSNE.quality_evaluation_ import hyperbolic_nearest_neighbor_preservation
from hyperbolicTSNE import Datasets, load_data
from hyperbolicTSNE.util import find_last_embedding

BASE_DIR = Path("../results/exp_grid")
DATASETS_DIR = "../datasets"

NEIGHBORHOOD_SIZE = 100
K_START = 1
K_MAX = 10
EXACT_NN = False
CONSIDER_ORDER = False
STRICT = False
TO_RETURN = "full"

PERP = 30

hd_params = {
    "perplexity": PERP
}

for dataset_dir in BASE_DIR.glob("*"):
    if dataset_dir.is_dir():
        dataset_name = dataset_dir.stem
        dataX = load_data(Datasets[dataset_name], data_home=DATASETS_DIR, to_return="X", hd_params=hd_params)
        for size_dir in dataset_dir.glob("*"):
            if size_dir.is_dir():
                for config_dir in size_dir.glob("*"):
                    if config_dir.is_dir():
                        for run_dir in config_dir.glob("*"):
                            if run_dir.is_dir():
                                print(run_dir)
                                subset_idx = np.load(run_dir.joinpath("subset_idx.npy"), allow_pickle=True)
                                dataX_sample = dataX[subset_idx]

                                D_path = list(run_dir.glob("D.np*"))[0]
                                sparse_D = D_path.suffix == ".npz"
                                if sparse_D:
                                    D_X = load_npz(D_path)
                                else:
                                    D_X = np.load(D_path, allow_pickle=True)[()]

                                # dataY = find_last_embedding(run_dir.joinpath("embeddings"))

                                dataY = np.load(run_dir.joinpath("final_embedding.npy"), allow_pickle=True)

                                thresholds, precisions, recalls, true_positives = \
                                    hyperbolic_nearest_neighbor_preservation(dataX_sample,
                                                                             dataY,
                                                                             K_START,
                                                                             K_MAX,
                                                                             D_X,
                                                                             EXACT_NN,
                                                                             CONSIDER_ORDER,
                                                                             STRICT,
                                                                             TO_RETURN,
                                                                             NEIGHBORHOOD_SIZE)

                                # save results inside folder
                                np.save(run_dir.joinpath("thresholds.npy"), thresholds)
                                np.save(run_dir.joinpath("precisions.npy"), precisions)
                                np.save(run_dir.joinpath("recalls.npy"), recalls)
                                np.save(run_dir.joinpath("true_positives.npy"), true_positives)

                                # Save final embedding
                                fig, ax = plt.subplots()
                                ax.scatter(precisions, recalls)
                                ax.set_xlabel("Precision")
                                ax.set_ylabel("Recall")
                                fig.savefig(run_dir.joinpath(f"prec-vs-rec.png"))
                                plt.close(fig)
