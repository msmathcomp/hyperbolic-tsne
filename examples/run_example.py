import os
import traceback

from hyperbolicTSNE.util import find_last_embedding
from hyperbolicTSNE.visualization import plot_poincare, animate
from hyperbolicTSNE import load_data, Datasets, SequentialOptimizer, initialization, HDEO, PoincareDiskModel

data_home = "../datasets"

seed = 42
dataX, dataY, D, V = load_data(Datasets.MYELOID, data_home=data_home, random_state=seed, to_return="X_labels_D_V",
                               hd_params={"perplexity": 30})

opt_params = SequentialOptimizer.sequence_poincare(PoincareDiskModel,
                                                   gradientDescent_its=750,
                                                   learning_rate=1,
                                                   vanilla=True,
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
if os.path.exists(opt_params["logging_dict"]["log_path"]):
    import shutil
    shutil.rmtree(opt_params["logging_dict"]["log_path"])
# End: logging

hdeo_hyper = HDEO(init=X_embedded, n_components=2, metric="precomputed", verbose=True, opt_method=SequentialOptimizer, opt_params=opt_params)

try:
    res_hdeo_hyper = hdeo_hyper.fit_transform((D, V))
except ValueError:
    res_hdeo_hyper = find_last_embedding(opt_params)
    traceback.print_exc()

fig = plot_poincare(res_hdeo_hyper, dataY)
fig.show()

animate(logging_dict, dataY, "../results/poincare.mp4")
