import os

from hyperbolicTSNE.quality_evaluation_ import hyperbolic_nearest_neighbor_preservation, nearest_neighbor_preservation
from matplotlib import pyplot as plt
from hyperbolicTSNE import load_data, Datasets, SequentialOptimizer, initialization, HDEO
# from hyperbolicTSNE.visualization import plot_poincare

data_home = "datasets"
seed = 42
dataset = Datasets.MNIST
dataX, dataY, D, V = load_data(dataset, data_home=data_home, random_state=seed, to_return="X_labels_D_V",
                               hd_params={"perplexity": 30})

X_embedded = initialization(n_samples=dataX.shape[0],
                            n_components=2,
                            X=dataX,
                            random_state=seed,
                            method="pca")
# Save final embedding
_, ax = plt.subplots()
ax.set_title("Hyperbolic NNP")
ax.set_xlabel('Precision')
ax.set_ylabel('Recall')

PERP = 30
# LR = 1  # Defined with heuristic (see below)
VANILLA = False  # whether to use momentum or not
EXAG = 12
LR = (dataX.shape[0] * 1) / (EXAG * 1000)

for theta in [x / 10 for x in range(0, 11, 1)]:

    opt_params = SequentialOptimizer.sequence_poincare(learning_rate_ex=LR,  # TODO: change based on dataset size?
                                                       learning_rate_main=LR, # TODO: change based on dataset size?
                                                       exaggeration=EXAG,
                                                       vanilla=VANILLA,
                                                       momentum_ex=0.5,
                                                       momentum=0.8,
                                                       exact=False,
                                                       area_split=False,
                                                       grad_fix=False,
                                                       grad_scale_fix=False,
                                                       n_iter_check=10,  # Needed for size check
                                                       angle=theta,
                                                       size_tol=0.999)  # Needed for size check

    # Start: logging
    log_path = "../temp/poincare/"
    logging_dict = {
        "log_path": log_path
    }
    opt_params["logging_dict"] = logging_dict
    # Delete old log path
    if os.path.exists(log_path):
        import shutil
        shutil.rmtree(log_path)
    # End: logging

    hdeo_so = HDEO(init=X_embedded, n_components=2, metric="precomputed", verbose=True, opt_method=SequentialOptimizer,
                   opt_params=opt_params)
    res_hdeo_so = hdeo_so.fit_transform((D, V))
    # res_hdeo_so = find_ith_embedding(logging_dict["log_path"], 500)
    # plot_poincare(res_hdeo_so, dataY).show()

    _, precision, recall, _ = hyperbolic_nearest_neighbor_preservation(dataX, res_hdeo_so,
                                                                       k_start=1, k_max=10,
                                                                       D_X=None,
                                                                       exact_nn=True,
                                                                       consider_order=False,
                                                                       strict=False,
                                                                       to_return="full")



    line, = ax.plot(precision, recall)
    line.set_label(f"{theta}")

ax.legend()
plt.savefig(f"{dataset.name}_nnp_theta_quality_exp.pdf")
