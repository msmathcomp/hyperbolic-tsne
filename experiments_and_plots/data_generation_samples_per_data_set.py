"""
This experiment runs the hyperbolic tsne code changing several parameters:
- dataset: [LUKK, MYELOID8000, PLANARIA, MNIST, C_ELEGANS, WORDNET] 
- tsne_type: [accelerated, exact]
- splitting strategy of the polar quad tree: [eqaul_area, equal_length]
- sample_size: [0.1, 0.2, ..., 1.0] times the full data set.
To ensure that the random sampling does not affect the results, each configuration is run several times.
For each data set and each configuration combination, it saves the embedding coordinates, a plot of the embedding, and
timing data for the iterations.
If a run does not finish, the results are not saved.
The code only computes the runs that do not have a folder.
"""

###########
# IMPORTS #
###########

import csv
import json
import traceback
from itertools import product
from pathlib import Path

import numpy as np
from scipy.sparse import issparse, save_npz
from matplotlib import pyplot as plt

from hyperbolicTSNE import Datasets, load_data, initialization, hd_matrix, SequentialOptimizer, HDEO
from hyperbolicTSNE.util import find_last_embedding
from hyperbolicTSNE.visualization import plot_poincare


#################################
# GENERAL EXPERIMENT PARAMETERS #
#################################

BASE_DIR = "../results/exp_grid"  # dir where results will be saved
DATASETS_DIR = "../datasets"  # directory to read the data from

# Constants
SEED = 42  # seed to initialize random processes
RUNS = 5  # number of repetitions to be run for each random sampling
SIZE_SAMPLES = 10  # how many sample sizes to consider for the data set
PERP = 30  # perplexity value to be used throughout the experiments
KNN_METHOD = "hnswlib"  # use hnswlib for determining nearest neighbors in high-dimensional space; note that this is
# an approximation, switch to "sklearn" for an exact method
VANILLA = False  # if vanilla is set to true, regular gradient descent without any modifications is performed; for
# vanilla set to false, the optimization makes use of momentum and gains
EXAG = 12  # the factor by which the attractive forces are amplified during early exaggeration
hd_params = {"perplexity": PERP}

# Variables
datasets = [
    Datasets.LUKK,
    Datasets.MYELOID8000,
    Datasets.PLANARIA,
    Datasets.MNIST,
    Datasets.C_ELEGANS,
    Datasets.WORDNET
]
tsne_types = ["accelerated", "exact"]  # the type "accelerated" uses the polar quad tree for acceleration, "exact"
# uses no acceleration and runs in quadratic time per iteration
splitting_strategies = ["equal_length", "equal_area"]  # the polar quad tree comes in two flavors: Splitting by equal
# area and by equal length in the embedding space. The "equal_length" splitting shows better performance in our
# experiments.

###################
# EXPERIMENT LOOP #
###################

overview_created = False
for dataset in datasets:  # Iterate over the data sets

    rng = np.random.default_rng(seed=SEED)  # random number generator
    
    dataX, dataLabels = load_data(  # Load the data
        dataset,
        data_home=DATASETS_DIR,
        to_return="X_labels",  # Return the high-dimensional data and its labels
        hd_params=hd_params,
        knn_method=KNN_METHOD
    )

    n_samples = dataX.shape[0]
    sample_sizes = np.linspace(0, n_samples, num=SIZE_SAMPLES + 1)[1:].astype(int)  # Equally distribute sample sizes
    # across the available number of data points

    X_embedded = initialization(  # create an initial embedding of the data into 2-dimensional space via PCA
        n_samples=n_samples,
        n_components=2,
        X=dataX,
        random_state=rng.integers(0, 1000000),
        method="pca"
    )

    for config_id, config in enumerate(product(sample_sizes, tsne_types, splitting_strategies)):

        sample_size, tsne_type, splitting_strategy = config

        # We only have to run one version of exact t-SNE, so we use the combination "exact" + "equal_length" (which
        # does not use the splitting property anyway since it is exact). Hence, we can skip the "equal_area" version
        # here as it does not provide more data.
        if tsne_type == "exact" and splitting_strategy == "equal_area":
            continue

        for run_n in range(RUNS):  # Run the same configuration multiple times to avaerage out random fluctuations
            # from the sampling
            
            print(f"[experiment_grid] Processing {dataset}, run_id {run_n}, config_id ({config_id}): {config}")

            # Generate random sample
            idx = rng.choice(np.arange(n_samples), sample_size, replace=False)
            idx = np.sort(idx)

            dataX_sample = dataX[idx]
            dataLabels_sample = dataLabels[idx]
            X_embedded_sample = X_embedded[idx]  # Create an initial embedding of the sampling by sampling the large
            # PCA embedding created for the entire data set

            D, V = hd_matrix(X=dataX_sample, hd_params=hd_params, knn_method=KNN_METHOD)  # Compute the NN matrix

            LR = (dataX_sample.shape[0] * 1) / (EXAG * 1000)  # Compute the learning rate

            opt_params = SequentialOptimizer.sequence_poincare(
                learning_rate_ex=LR,  # specify learning rate for the early exaggeration
                learning_rate_main=LR,  # specify learning rate for the non-exaggerated gradient descent iterations
                exaggeration=EXAG,
                vanilla=VANILLA,
                momentum_ex=0.5,  # momentum to be used during early exaggeration
                momentum=0.8,  # momentum to be used during non-exaggerated gradient descent
                exact=(tsne_type == "exact"),
                area_split=(splitting_strategy == "equal_area"),
                n_iter_check=10,  # Needed for early stopping criterion
                size_tol=0.999  # Size of the embedding to be used as early stopping criterion
            )

            run_dir = Path(f"{BASE_DIR}/{dataset.name}/size_{sample_size}/configuration_{config_id}/run_{run_n}/")

            if run_dir.exists():
                # Skip already computed embeddings
                print(f"[experiment_grid] - Exists so not computing it: {run_dir}")

            else:                
                run_dir.mkdir(parents=True, exist_ok=True)

                params = {
                    "lr": LR,
                    "perplexity": PERP,
                    "seed": SEED,
                    "sample_size": int(sample_size),
                    "tsne_type": tsne_type,
                    "splitting_strategy": splitting_strategy
                }

                print(f"Starting configuration {config_id} with dataset {dataset.name}: {params}")

                opt_params["logging_dict"] = {
                    "log_path": str(run_dir.joinpath("embeddings"))
                }

                # Save the high-dimensional neighborhood matrices for later use
                json.dump(params, open(run_dir.joinpath("params.json"), "w"))
                np.save(run_dir.joinpath("subset_idx.npy"), idx)
                if issparse(D):
                    save_npz(run_dir.joinpath("D.npz"), D)
                else:
                    np.save(run_dir.joinpath("D.npy"), D)
                if issparse(V):
                    save_npz(run_dir.joinpath("P.npz"), V)
                else:
                    np.save(run_dir.joinpath("P.npy"), V)

                hdeo_hyper = HDEO(  # Initialize an embedding object
                    init=X_embedded_sample,
                    n_components=X_embedded_sample.shape[1],
                    metric="precomputed",
                    verbose=2,
                    opt_method=SequentialOptimizer,
                    opt_params=opt_params
                )

                error_title = ""
                try:
                    res_hdeo_hyper = hdeo_hyper.fit_transform((D, V))  # Compute the hyperbolic embedding
                except ValueError: 

                    error_title = "_error"
                    res_hdeo_hyper = find_last_embedding(opt_params["logging_dict"]["log_path"])
                    traceback.print_exc(file=open(str(run_dir) + "traceback.txt", "w"))

                    print("[experiment_grid] - Run failed ...")

                else:  # we save the data if there were no errors

                    print("[experiment_grid] - Finished running, saving run data directory ...")

                    # Save the final embedding coordinates
                    np.save(run_dir.joinpath("final_embedding.npy"), res_hdeo_hyper)

                    # Save final embedding
                    fig = plot_poincare(res_hdeo_hyper, labels=dataLabels_sample)
                    fig.savefig(run_dir.joinpath(f"final_embedding{error_title}.png"))
                    plt.close(fig)

                    np.save(run_dir.joinpath("logging_dict.npy"), opt_params["logging_dict"])

                    # Write out timings csv
                    timings = np.array(hdeo_hyper.optimizer.cf.results)
                    with open(run_dir.joinpath("timings.csv"), "w", newline="") as timings_file:
                        timings_writer = csv.writer(timings_file)
                        timings_writer.writerow(["it_n", "time_type", "total_time"])

                        for n, row in enumerate(timings):
                            timings_writer.writerow([n, "tree_building", row[0]])
                            timings_writer.writerow([n, "tot_gradient", row[1]])
                            timings_writer.writerow([n, "neg_force", row[2]])
                            timings_writer.writerow([n, "pos_force", row[3]])

                    # Create or append to overview csv file after every run
                    with open(run_dir.joinpath(f"overview_part.csv"), "w", newline="") as overview_file:
                        overview_writer = csv.writer(overview_file)
                        overview_writer.writerow(["dataset", *params, "run", "run_directory", "error"])
                        overview_writer.writerow([
                            dataset.name,
                            *params.values(),
                            run_n,
                            str(run_dir).replace(str(BASE_DIR), "."),
                            error_title != ""
                        ])

                    print()
