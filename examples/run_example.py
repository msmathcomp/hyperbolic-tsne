import os
import traceback

from hyperbolicTSNE.util import find_last_embedding
from hyperbolicTSNE.visualization import plot_poincare, animate
from hyperbolicTSNE import load_data, Datasets, SequentialOptimizer, initialization, HDEO

data_home = "../datasets"

seed = 42
dataX, dataY, D, V = load_data(Datasets.MYELOID, data_home=data_home, random_state=seed, to_return="X_labels_D_V",
                               hd_params={"perplexity": 30})

opt_params = SequentialOptimizer.sequence_poincare(gradientDescent_its=140,
                                                   learning_rate=1,
                                                   vanilla=False,
                                                   exact=False)

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

# Delete old log path
log_path = opt_params["logging_dict"]["log_path"]
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

fig = plot_poincare(res_hdeo_hyper, dataY)
fig.show()

animate(logging_dict, dataY, "../results/poincare.mp4")
