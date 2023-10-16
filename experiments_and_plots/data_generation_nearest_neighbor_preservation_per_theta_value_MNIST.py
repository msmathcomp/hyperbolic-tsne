"""
This experiment runs the hyperbolic tsne code on the MINNST data set changing several parameters:
- theta: [0.0, 0.1, 0.2, ..., 1.0]
For each configuration combination, it saves the embedding coordinates, a plot of the embedding, and
timing data for the iterations.
If a run does not finish, the results are not saved.
The code only computes the runs that do not have a folder.
"""
###########
# IMPORTS #
###########

import json
import traceback
from pathlib import Path

import numpy as np
from scipy.sparse import issparse, save_npz

from hyperbolicTSNE.quality_evaluation_ import hyperbolic_nearest_neighbor_preservation
from matplotlib import pyplot as plt
from hyperbolicTSNE import load_data, Datasets, SequentialOptimizer, initialization, HyperbolicTSNE
from hyperbolicTSNE.util import find_last_embedding
from hyperbolicTSNE.visualization import plot_poincare

#################################
# GENERAL EXPERIMENT PARAMETERS #
#################################

BASE_DIR = "../results/nnp_per_theta_MNIST"
DATASETS_DIR = "../datasets"  # directory to read the data from

# Constants
SEED = 42  # seed to initialize random processes
PERP = 30  # perplexity value to be used throughout the experiments
VANILLA = False  # whether to use momentum or not
EXAG = 12  # the factor by which the attractive forces are amplified during early exaggeration
hd_params = {"perplexity": PERP}
dataset = Datasets.MNIST  # The dataset to run the experiment on

###################
# EXPERIMENT LOOP #
###################

dataX, dataLabels, D, V = load_data(  # Load the data
    dataset,
    data_home=DATASETS_DIR,
    random_state=SEED,
    to_return="X_labels_D_V",
    hd_params=hd_params
)

X_embedded = initialization(  # create an initial embedding of the data into 2-dimensional space via PCA
    n_samples=dataX.shape[0],
    n_components=2,
    X=dataX,
    random_state=SEED,
    method="pca"
)

# Save final embedding
_, ax = plt.subplots()
ax.set_title("Hyperbolic NNP")
ax.set_xlabel('Precision')
ax.set_ylabel('Recall')


LR = (dataX.shape[0] * 1) / (EXAG * 1000)  # Compute the learning rate

for theta in [x / 10 for x in range(0, 11, 1)]:  # Iterate over the different values for theta

    print(f"[nnp_per_theta] Processing {dataset}, Theta: {theta}")

    opt_params = SequentialOptimizer.sequence_poincare(
        learning_rate_ex=LR,  # specify learning rate for the early exaggeration
        learning_rate_main=LR,  # specify learning rate for the non-exaggerated gradient descent
        exaggeration=EXAG,
        vanilla=VANILLA,
        momentum_ex=0.5,  # momentum to be used during early exaggeration
        momentum=0.8,  # momentum to be used during non-exaggerated gradient descent
        exact=False,
        n_iter_check=10,  # Needed for early stopping criterion
        size_tol=0.999,  # Size of the embedding to be used as early stopping criterion
        angle=theta

    )

    run_dir = Path(f"{BASE_DIR}/theta_{theta}/")

    if run_dir.exists():
        # Skip already computed embeddings
        print(f"[nnp_per_theta] - Exists so not computing it: {run_dir}")
    else:
        run_dir.mkdir(parents=True, exist_ok=True)

        params = {
            "lr": LR,
            "perplexity": PERP,
            "seed": SEED,
            "theta": theta
        }

        print(f"[nnp_per_theta] - Starting with Theta: {theta}")

        opt_params["logging_dict"] = {
            "log_path": str(run_dir.joinpath("embeddings"))
        }

        # Save the high-dimensional neighborhood matrices for later use
        json.dump(params, open(run_dir.joinpath("params.json"), "w"))
        if issparse(D):
            save_npz(run_dir.joinpath("D.npz"), D)
        else:
            np.save(run_dir.joinpath("D.npy"), D)
        if issparse(V):
            save_npz(run_dir.joinpath("P.npz"), V)
        else:
            np.save(run_dir.joinpath("P.npy"), V)

        hdeo_hyper = HyperbolicTSNE(  # Initialize an embedding object
            init=X_embedded,
            n_components=X_embedded.shape[1],
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

            print("[nnp_per_theta] - Run failed ...")

        else:  # we save the data if there were no errors

            print("[nnp_per_theta] - Finished running, saving run data directory ...")

            # Save the final embedding coordinates
            np.save(run_dir.joinpath("final_embedding.npy"), res_hdeo_hyper)

            # Save a plot of the final embedding
            fig = plot_poincare(res_hdeo_hyper, labels=dataLabels)
            fig.savefig(run_dir.joinpath(f"final_embedding{error_title}.png"))
            plt.close(fig)

            np.save(run_dir.joinpath("logging_dict.npy"), opt_params["logging_dict"])

            # Compute Precision and Recall values for the embedding
            _, precisions, recalls, _ = hyperbolic_nearest_neighbor_preservation(
                dataX,
                res_hdeo_hyper,
                k_start=1,
                k_max=30,
                D_X=None,
                exact_nn=True,
                consider_order=False,
                strict=False,
                to_return="full"
            )
            np.save(run_dir.joinpath(f"precisions_theta_{theta}.npy"), precisions)
            np.save(run_dir.joinpath(f"recalls_theta_{theta}.npy"), recalls)

            print()
