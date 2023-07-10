import csv
import json
import os
import traceback
from itertools import product
from pathlib import Path
import sys

import numpy as np
from matplotlib import pyplot as plt

from hyperbolicTSNE import Datasets, load_data, initialization, hd_matrix, SequentialOptimizer, HDEO
from hyperbolicTSNE.util import find_last_embedding
from hyperbolicTSNE.visualization import plot_poincare, plot_poincare_zoomed

# TODO: use this?
# sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
# print(str(Path(__file__).resolve().parent.parent.parent))

# Experiment
# Run: one execution of HyperbolicTSNE with a parameter set.
# Each run has a dict of parameters associated to it.

# Fixed variables
# - Iterations: 1000 (bound to change based on dataset)
# - learning rate: (bound to change based on dataset)
# - Sampling protocol (to reduce datasets' N)

# Conditions (ways to slice our results table)
# - dataset (5 (for now): mnist, myeloid?, planaria, c_elegans, wordnet?)
# - dataset sample N'
# - TSNE type (3 options: Hyperbolic, Hyperbolic "BH", TSNE BH)
# - splitting strategy (2 options: equal area, equal length)

# - exag (3 options: combined, only exag, without exag)
# - einstein vs frechet (2 options: for now only einstein)
# - logging (2 options: with and without to avoid contaminating timings)

# Notes:
# - Run exact just for N < 20.000. Leave the dir blank if that is the case. TODO: correct?

# Measurements (values in the cells of the table)
# - per iteration:
# - per run:
# - iterations (embedding per iteration)
# - timings

# Folder structure of experiment
# /experiment_results
#   overview.csv TODO
#   /dataset_1
#      /dataset_size_1
#        /configuration_1
#           /run_1
#             embeddings/   # every 20 iterations
#             params.csv
#             timings.csv
#             choices.csv TODO
#             final_embedding.png?

# overview.csv
# dataset, ...parameters,
# dataset_1, seed, einstein, choice, run_1, run_dir
# dataset_1, seed, einstein, choice, run_2, run_dir
# dataset_1, seed, einstein, choice, run_3, run_dir

# timings.csv
# it_n, time_type, total_time
# 235, gradient_desc_step, 1.5435
# 235, negative_grad_comp, 1.456
# 235, positive_grad_comp, 1.456
# timings.groupby(gradient_desc_step).mean()
# timings.filterby(gradient_desc_step).groupby(it_n).mean()

# Questions
# - 

SEED = 42

n_of_runs = 5
perplexity = 30
lr = 1

hd_params = {
    "perplexity": perplexity
}

base_dir = "../results"

datasets = [Datasets.LUKK, Datasets.MYELOID8000, Datasets.PLANARIA,
            Datasets.MNIST, Datasets.C_ELEGANS, Datasets.WORDNET]
num_sample_sizes = 10  # TODO: how many intervals
tsne_types = ["accelerated", "exact"]
splitting_strategies = ["equal_length", "equal_area"]


overview_created = False
for dataset in datasets:    
    rng = np.random.default_rng(seed=SEED)
    dataX, dataY = load_data(dataset, to_return="X_labels", hd_params=hd_params)
    n_samples = dataX.shape[0]

    sample_sizes = np.linspace(0, n_samples, num=num_sample_sizes + 1)[1:].astype(int)

    X_embedded = initialization(n_samples=n_samples,
                                n_components=2,
                                X=dataX,
                                random_state=rng.integers(0, 1000000),
                                method="pca")

    for config_id, parameters in enumerate(product(sample_sizes, tsne_types, splitting_strategies)):        
        for run_n in range(n_of_runs):
            sample_size, tsne_type, splitting_strategy = parameters
            print(f"[experiment_grid] Processing {dataset}, config_id {config_id}, run_id {run_n}")

            # Generate random sample
            idx = rng.choice(np.arange(n_samples), sample_size, replace=False)  # TODO: keep track of random seed
            idx = np.sort(idx)

            dataX_sample = dataX[idx]
            dataY_sample = dataY[idx]
            X_embedded_sample = X_embedded[idx]

            D, V = hd_matrix(X=dataX_sample, hd_params=hd_params)

            opt_params = SequentialOptimizer.sequence_poincare(learning_rate=lr,  # TODO: change based on dataset size?
                                                               vanilla=True,
                                                               exact=tsne_type == "exact",
                                                               area_split=splitting_strategy == "equal_area")

            run_directory = f"{base_dir}/{dataset.name}/size_{sample_size}/configuration_{config_id}/run_{run_n}/"

            params = {
                "lr": lr,
                "perplexity": perplexity,
                "seed": SEED,
                "sample_size": int(sample_size),
                "tsne_type": tsne_type,
                "splitting_strategy": splitting_strategy
            }

            print(f"Starting configuration {config_id} with dataset {dataset.name}: {params}")

            # Create directories if non existent
            Path(run_directory).mkdir(parents=True, exist_ok=True)

            json.dump(params, open(run_directory + "params.json", "w"))
            np.save(run_directory + "subset_idx.npy", idx)
            np.save(run_directory + "D.npy", D)
            np.save(run_directory + "P.npy", V)

            opt_params["logging_dict"] = {
                "log_path": run_directory + "embeddings/"
            }

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
                traceback.print_exc(file=open(run_directory + "traceback.txt", "w"))

            # Save final embedding
            fig = plot_poincare(res_hdeo_hyper, labels=dataY_sample)
            fig.savefig(f"{run_directory}final_embedding{error_title}.png")
            plt.close(fig)

            fig = plot_poincare_zoomed(res_hdeo_hyper, labels=dataY_sample)
            fig.savefig(f"{run_directory}final_embedding_zoomed{error_title}.png")
            plt.close(fig)

            np.save(run_directory + "logging_dict.npy", opt_params["logging_dict"])

            # Write out timings csv
            timings = np.array(hdeo_hyper.optimizer.cf.results)
            with open(run_directory + "timings.csv", "w", newline="") as timings_file:
                timings_writer = csv.writer(timings_file)
                timings_writer.writerow(["it_n", "time_type", "total_time"])

                for n, row in enumerate(timings):
                    timings_writer.writerow([n, "tree_building", row[0]])
                    timings_writer.writerow([n, "tot_gradient", row[1]])
                    timings_writer.writerow([n, "neg_force", row[2]])
                    timings_writer.writerow([n, "pos_force", row[3]])

            # Create or append to overview csv file after every run
            with open(f"{base_dir}/overview.csv", "a" if overview_created else "w", newline="") as overview_file:
                overview_writer = csv.writer(overview_file)

                if not overview_created:
                    overview_writer.writerow(["dataset", *params, "run", "run_directory", "error"])
                    overview_created = True

                overview_writer.writerow([dataset.name, *params.values(),
                                          run_n, run_directory.replace(base_dir, "."), error_title != ""])

            print()
