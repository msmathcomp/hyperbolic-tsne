from hyperbolicTSNE.util import find_last_embedding
from hyperbolicTSNE.visualization import plot_poincare, animate, plot_poincare_zoomed
from hyperbolicTSNE import load_data, Datasets

data_home = "../datasets"

seed = 42
dataset = Datasets.MNIST
dataX, dataY, D, V, _ = load_data(dataset, data_home=data_home, random_state=seed, to_return="X_labels_D_V",
                                    hd_params={"perplexity": 30}, sample=10000)

logging_dict = {
    "log_path": "../temp/poincare/"
}

log_path = logging_dict["log_path"]

res_hdeo_hyper = find_last_embedding(log_path)

plot_poincare(res_hdeo_hyper, dataY).show()

plot_poincare_zoomed(res_hdeo_hyper, dataY).show()

animate(logging_dict, dataY, f"../results/{dataset.name}_only_anim.mp4", fast=True, plot_ee=True)
