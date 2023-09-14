from pathlib import Path
import sys

# sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
# print(str(Path(__file__).resolve().parent.parent.parent))

import csv
import json
import os
import traceback

import numpy as np
from matplotlib import pyplot as plt

from hyperbolicTSNE import Datasets, load_data, initialization, SequentialOptimizer, HDEO

SEED = 42

n_of_runs = 5
perplexity = 30
# lr = 1
EXAG = 12
VANILLA = False  # whether to use momentum or not
KNN_METHOD = ["sklearn", "hnswlib"][0]

hd_params = {
    "perplexity": perplexity
}

base_dir = "../results"


datasets = [
    Datasets.LUKK,
    Datasets.MYELOID8000,
    Datasets.PLANARIA,
    Datasets.MNIST,
    Datasets.C_ELEGANS,
    Datasets.WORDNET
]
thetas = [n / 20 for n in range(20, -1, -1)]


def find_last_embedding(log_path):
    for subdir, dirs, files in reversed(list(os.walk(log_path))):
        for fi, file in enumerate(reversed(sorted(files, key=lambda x: int(x.split(", ")[0])))):
            root, ext = os.path.splitext(file)
            if ext == ".csv":
                total_file = subdir.replace("\\", "/") + "/" + file

                return np.genfromtxt(total_file, delimiter=',')


overview_created = False
for dataset in datasets:
    rng = np.random.default_rng(seed=SEED)
    dataX, dataY, D, V = load_data(dataset, data_home="/home/martin/hyperbolic-tsne/datasets/", to_return="X_labels_D_V", hd_params=hd_params)
    n_samples = dataX.shape[0]

    X_embedded = initialization(n_samples=n_samples,
                                n_components=2,
                                X=dataX,
                                random_state=rng.integers(0, 1000000),
                                method="pca")

    for config_id, theta in enumerate(thetas):
        # for run_n in range(n_of_runs):
        #     sample_size, tsne_type, splitting_strategy = parameters
        #     print(f"[experiment_grid] Processing {dataset}, config_id {config_id}, run_id {run_n}")

        LR = (dataX.shape[0] * 1) / (EXAG * 1000)

        opt_params = SequentialOptimizer.sequence_poincare(learning_rate_ex=LR,
                                                           learning_rate_main=LR,
                                                           exaggeration=EXAG,
                                                           vanilla=VANILLA,
                                                           momentum_ex=0.5,
                                                           momentum=0.8,
                                                           exact=False,
                                                           area_split=False,
                                                           grad_fix=False,
                                                           grad_scale_fix=False,
                                                           n_iter_check=10,
                                                           size_tol=0.999,
                                                           angle=theta)

        run_directory = f"{base_dir}/exp_theta_times/{dataset.name}/theta_{theta}/"

        params = {
            "lr": LR,
            "perplexity": perplexity,
            "seed": SEED,
            "sample_size": n_samples,
            "tsne_type": "exact",
            "splitting_strategy": "equal_length",
            "theta": theta
        }

        print(f"Starting configuration {config_id} with dataset {dataset.name}: {params}")

        # Create directories if non existent
        Path(run_directory).mkdir(parents=True, exist_ok=True)

        json.dump(params, open(run_directory + "params.json", "w"))
        np.save(run_directory + "D.npy", D)
        np.save(run_directory + "P.npy", V)

        opt_params["logging_dict"] = {
            "log_path": run_directory + "embeddings/"
        }

        hdeo_hyper = HDEO(init=X_embedded,
                          n_components=X_embedded.shape[1],
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
        fig, ax = plt.subplots()
        ax.scatter(res_hdeo_hyper.reshape(-1, 2)[:, 0], res_hdeo_hyper.reshape(-1, 2)[:, 1],
                   c=dataY,
                   marker=".")
        ax.axis("square")
        ax.add_patch(plt.Circle((0, 0), radius=1, edgecolor="b", facecolor="None"))
        fig.savefig(f"{run_directory}final_embedding{error_title}.png")
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
        with open(f"{base_dir}/overview_theta.csv", "a" if overview_created else "w", newline="") as overview_file:
            overview_writer = csv.writer(overview_file)

            if not overview_created:
                overview_writer.writerow(["dataset", *params, "run", "run_directory", "error"])
                overview_created = True

            overview_writer.writerow([dataset.name, *params.values(),
                                      "theta", run_directory.replace(base_dir, "."), error_title != ""])

        print()
