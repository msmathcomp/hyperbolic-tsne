import os
import traceback

from hyperbolicTSNE.util import find_last_embedding
from hyperbolicTSNE.visualization import plot_poincare, animate, plot_poincare_zoomed
from hyperbolicTSNE import load_data, Datasets, SequentialOptimizer, initialization, HDEO

data_home = "../datasets"

# Based on nicolas_exp_log
# - do not forget to do changes to code when switching between v1 and v2

seed = 42
dataset = Datasets.MNIST
dataX, dataY, D, V, _ = load_data(dataset, data_home=data_home, random_state=seed, to_return="X_labels_D_V",
                                    hd_params={"perplexity": 30}, knn_method="hnswlib", sample=10000, verbose=True)

# dataset = Datasets.MYELOID
# dataX, dataY, D, V = load_data(dataset, data_home=data_home, random_state=seed, to_return="X_labels_D_V",
#                                hd_params={"perplexity": 30})

# originally, the early exaggeration had no momentum. How do they do this in the original tSNE?

learning_rate = (dataX.shape[0] * 1) / (12 * 50)
print(f"learning_rate: {learning_rate}")
configs = dict(
    v1=dict(learning_rate_ex=learning_rate, learning_rate_main=learning_rate, exaggeration=12, exaggeration_its=250, gradientDescent_its=750, vanilla=True, exact=False, grad_fix=False, grad_scale_fix=False),
    v2a=dict(learning_rate_ex=learning_rate, learning_rate_main=learning_rate, exaggeration=12, exaggeration_its=250, gradientDescent_its=750, vanilla=True, exact=False, grad_fix=True, grad_scale_fix=False),
    v2b=dict(learning_rate_ex=learning_rate*10, learning_rate_main=learning_rate*0.2, exaggeration=12, exaggeration_its=250, gradientDescent_its=750, vanilla=True, exact=False, grad_fix=True, grad_scale_fix=True),
    v3=dict(learning_rate_ex=1, learning_rate_main=1, exaggeration=12, exaggeration_its=250, gradientDescent_its=750, vanilla=False, exact=False, grad_fix=True, grad_scale_fix=True),
)
config = configs["v2b"]
print(f"config: {config}")


opt_params = SequentialOptimizer.sequence_poincare(**config)

X_embedded = initialization(n_samples=dataX.shape[0],
                            n_components=2,
                            X=dataX,
                            random_state=seed,
                            method="pca")

# Start: logging
logging_dict = {
    "log_path": "../temp/poincare/"
}
opt_params["logging_dict"] = logging_dict

log_path = opt_params["logging_dict"]["log_path"]
# Delete old log path
if os.path.exists(log_path):
    import shutil
    shutil.rmtree(log_path)
# End: logging

hdeo_hyper = HDEO(init=X_embedded, n_components=2, metric="precomputed", verbose=True, opt_method=SequentialOptimizer, opt_params=opt_params)

try:
    res_hdeo_hyper = hdeo_hyper.fit_transform((D, V))
except ValueError:
    res_hdeo_hyper = find_last_embedding(log_path)
    traceback.print_exc()

# res_hdeo_hyper = find_last_embedding(log_path)

fig = plot_poincare(res_hdeo_hyper, dataY)
fig.savefig(f"../results/{dataset.name}-final.png")
#fig.show()

# fig = plot_poincare_zoomed(res_hdeo_hyper, dataY)
# fig.show()

animate(logging_dict, dataY, f"../results/{dataset.name}_fast.mp4", fast=True, plot_ee=True)
animate(logging_dict, dataY, f"../results/{dataset.name}.mp4")
