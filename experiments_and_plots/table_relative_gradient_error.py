"""
This script iterates over the results created by the `samples per data set` script. For each final exact embedding,
it extracts the exact cost function value, finds the corresponding accelerated embeddings, and extracts their respective
cost function values. From these cost function values, a relative error is computed and the statistics across this
relative errors are appended to the original data, which is then stored.
"""

###########
# IMPORTS #
###########

import ctypes
import math
from pathlib import Path
import pandas as pd
import numpy as np
import json
from tqdm import tqdm

from hyperbolicTSNE import Datasets, load_data
from hyperbolicTSNE.hd_mat_ import hd_matrix
from hyperbolicTSNE.hyperbolic_barnes_hut.tsne import distance_py, gradient


######################
# HELPER FUNCTION(S) #
######################

def squared_hyperbolic_distance(x, y):
    return math.pow(distance_py(x, y), 2)


#################################
# GENERAL EXPERIMENT PARAMETERS #
#################################
results_path = Path("../results/samples_per_data_set/")

# Load the overview of the experiments
df = pd.read_csv(results_path.joinpath("overview.csv"))

# Iterate over all snapshots and compare to the various approximated ones
averaged_results = df[["dataset"]].groupby(["dataset"]).first().reset_index()

# Constants
THETA = 0.5  # Theta value to be used in the approximate version of t-SNE
EMBEDDING_DIMENSIONS = 2  # Embed into two dimensions
VERBOSE = 0  # No additional debugging information when computing the gradient
NUM_THREADS = 4  # Number of threads to be used in gradient computation

###################
# EXPERIMENT LOOP #
###################

for i, dataset_record in enumerate(averaged_results.to_records()):
    dataset = dataset_record.dataset

    dataX = load_data(Datasets[dataset], to_return="X")

    # Iterate over all records for this dataset
    for record in df[(df.dataset == dataset)].to_records():

        gradient_errors_relative = []

        # Read the data from the specific experiment
        record_path = record.run_directory.replace(".", str(results_path))
        subset_idx = np.load(f"{record_path}/subset_idx.npy")

        with open(f"{record_path}/params.json", "r") as f:
            data = json.load(f)
            hd_params = {"perplexity": data["perplexity"]}
        f.close()

        D, V = hd_matrix(X=dataX[subset_idx], hd_params=hd_params)
        val_V = V.data
        neighbors = V.indices.astype(np.int64, copy=False)
        indptr = V.indptr.astype(np.int64, copy=False)

        # Iterate over all embeddings created for this specific data set
        for path in tqdm(list(Path(f"{record_path}/embeddings/").rglob("*.csv"))):
            if path.name.startswith("._"):
                continue

            Y = np.loadtxt(path, delimiter=",")
            Y = Y.astype(ctypes.c_double, copy=False)

            # Compute approximate grad
            grad_approx = np.zeros(Y.shape, dtype=ctypes.c_double)
            timings = np.zeros(4, dtype=ctypes.c_float)

            gradient(
                timings,
                val_V, Y, neighbors, indptr, grad_approx,
                theta=THETA,
                n_dimensions=EMBEDDING_DIMENSIONS,
                verbose=VERBOSE,
                compute_error=True,
                num_threads=NUM_THREADS,
                exact=False
            )

            # Compute exact grad
            grad_exact = np.zeros(Y.shape, dtype=ctypes.c_double)
            timings_exact = np.zeros(4, dtype=ctypes.c_float)

            gradient(
                timings_exact,
                val_V, Y, neighbors, indptr, grad_exact,
                theta=THETA,
                n_dimensions=EMBEDDING_DIMENSIONS,
                verbose=VERBOSE,
                compute_error=True,
                num_threads=NUM_THREADS,
                exact=True,
            )

            # Compute the error between the two gradient representations
            norm_difference = math.sqrt(sum(
                [squared_hyperbolic_distance(a, b) for a, b in zip(grad_approx, grad_exact)]
            ))
            norm_exact = math.sqrt(sum([squared_hyperbolic_distance(a, np.array([0., 0.])) for a in grad_exact]))
            gradient_errors_relative.append(norm_difference / norm_exact)

        # Compute descriptive statistics on the gradient errors
        avg_gradient_errors_relative = pd.DataFrame(gradient_errors_relative).describe().T.add_prefix("_gradient")
        for name, values in avg_gradient_errors_relative.items():
            averaged_results.at[i, name] = values.item()

        # Store results
        averaged_results.to_csv(results_path.joinpath(f"5-3_output_gradient-chkpt-{dataset}.csv"), index=False)

        print(f"Finished processing {dataset}")
