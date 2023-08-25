"""
This experiments run the hyperbolic tsne code changing several parameters:
- dataset: [LUKK, MYELOID8000, PLANARIA, MNIST, C_ELEGANS, WORDNET] 
- tsne_type: [accelerated, exact]
- P2
- P3
For each run, it saves to disk several files. 
If a run does not finish, the results are not saved.
The code only computes the runs that do not have a folder.
"""

###########
# IMPORTS #
###########

import csv
import json
import os
import traceback
from itertools import product
from pathlib import Path
import sys

import numpy as np
from scipy.sparse import issparse, save_npz
from matplotlib import pyplot as plt

from hyperbolicTSNE import Datasets, load_data, initialization, hd_matrix, SequentialOptimizer, HDEO
from hyperbolicTSNE.util import find_last_embedding
from hyperbolicTSNE.visualization import plot_poincare, plot_poincare_zoomed


#################################
# GENERAL EXPERIMENT PARAMETERS #
#################################

BASE_DIR = "../results/exp_grid"  # dir where results will be saved
DATASETS_DIR = "../datasets"

# Constants

SEED = 42
RUNS = 5
SIZE_SAMPLES = 10  # How many sizes to consider in the interval

PERP = 30
# LR = 1  # Defined with heuristic (see below)
KNN_METHOD = ["sklearn", "hnswlib"][0]
VANILLA = False  # whether to use momentum or not
EXAG = 12

hd_params = {
    "perplexity": PERP
}

# Variables

datasets = [
    Datasets.LUKK,
    Datasets.MYELOID8000,
    Datasets.PLANARIA,
    Datasets.MNIST,
    Datasets.C_ELEGANS,
    Datasets.WORDNET
    ]

tsne_types = ["accelerated", "exact"]
splitting_strategies = ["equal_length", "equal_area"]


###################
# EXPERIMENT LOOP #
###################

overview_created = False
for dataset in datasets:    
    rng = np.random.default_rng(seed=SEED)  # random number generator
    
    dataX, dataY = load_data(dataset, data_home=DATASETS_DIR, to_return="X_labels", hd_params=hd_params, knn_method=KNN_METHOD)
    n_samples = dataX.shape[0]

    sample_sizes = np.linspace(0, n_samples, num=SIZE_SAMPLES + 1)[1:].astype(int)

    X_embedded = initialization(n_samples=n_samples,
                                n_components=2,
                                X=dataX,
                                random_state=rng.integers(0, 1000000),
                                method="pca")

    for config_id, config in enumerate(product(sample_sizes, tsne_types, splitting_strategies)):

        sample_size, tsne_type, splitting_strategy = config

        # We only have to run one version of exact t-SNE, so we use the combination "exact" + "equal_length" (which
        # does not use the splitting property anyway since it is exact). Hence, we can skip the "equal_area" version
        # here as it does not provide more data.
        if tsne_tpye == "exact" and splitting_strategy == "equal_area":
            continue

        for run_n in range(RUNS):
            
            print(f"[experiment_grid] Processing {dataset}, run_id {run_n}, config_id ({config_id}): {config}")

            # Generate random sample
            idx = rng.choice(np.arange(n_samples), sample_size, replace=False)  # TODO: keep track of random seed
            idx = np.sort(idx)

            dataX_sample = dataX[idx]
            dataY_sample = dataY[idx]
            X_embedded_sample = X_embedded[idx]

            D, V = hd_matrix(X=dataX_sample, hd_params=hd_params, knn_method=KNN_METHOD)

            LR = (dataX_sample.shape[0] * 1) / (EXAG * 1000)

            opt_params = SequentialOptimizer.sequence_poincare(learning_rate_ex=LR,  # TODO: change based on dataset size?
                                                               learning_rate_main=LR, # TODO: change based on dataset size?
                                                               exaggeration=EXAG,
                                                               vanilla=VANILLA,
                                                               momentum_ex=0.5,
                                                               momentum=0.8,
                                                               exact=(tsne_type == "exact"),
                                                               area_split=(splitting_strategy == "equal_area"),
                                                               grad_fix=False,
                                                               grad_scale_fix=False,
                                                               n_iter_check=10,  # Needed for size check
                                                               size_tol=0.999)  # Needed for size check
            

            run_dir = Path(f"{BASE_DIR}/{dataset.name}/size_{sample_size}/configuration_{config_id}/run_{run_n}/")

            if run_dir.exists():
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

                hdeo_hyper = HDEO(init=X_embedded_sample,
                                n_components=X_embedded_sample.shape[1],
                                metric="precomputed",
                                verbose=2,
                                opt_method=SequentialOptimizer,
                                opt_params=opt_params)

                error_title = ""
                try:
                    res_hdeo_hyper = hdeo_hyper.fit_transform((D, V))
                except ValueError: 

                    error_title = "_error"
                    res_hdeo_hyper = find_last_embedding(opt_params["logging_dict"]["log_path"])
                    traceback.print_exc(file=open(run_dir + "traceback.txt", "w"))

                    print("[experiment_grid] - Run failed ...")

                else:  # we save the data if there were no errors

                    print("[experiment_grid] - Finished running, saving run data directory ...")

                    np.save(run_dir.joinpath("final_embedding.npy"), res_hdeo_hyper)

                    # Save final embedding
                    fig = plot_poincare(res_hdeo_hyper, labels=dataY_sample)
                    fig.savefig(run_dir.joinpath(f"final_embedding{error_title}.png"))
                    plt.close(fig)

                    fig = plot_poincare_zoomed(res_hdeo_hyper, labels=dataY_sample)
                    fig.savefig(run_dir.joinpath(f"final_embedding_zoomed{error_title}.png"))
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
                        overview_writer.writerow([dataset.name, *params.values(), run_n, str(run_dir).replace(str(BASE_DIR), "."), error_title != ""])

                    print()
