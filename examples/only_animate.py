from hyperbolicTSNE.util import find_last_embedding, find_ith_embedding
from hyperbolicTSNE.visualization import plot_poincare, animate, plot_poincare_zoomed, save_poincare_teaser
from hyperbolicTSNE import load_data, Datasets

data_home = "../datasets"

seed = 42
# dataset = Datasets.C_ELEGANS
# dataY = load_data(dataset, data_home=data_home, random_state=42, to_return="labels", hd_params={"perplexity": 30})
# dataX, dataY, D, V = load_data(dataset, data_home=data_home, random_state=seed, to_return="labels",
#                                     hd_params={"perplexity": 30})

dataset = Datasets.MNIST
dataY = load_data(dataset, data_home=data_home, random_state=seed, to_return="labels",
                                    hd_params={"perplexity": 30}, knn_method="hnswlib", sample=70000, verbose=True)

logging_dict = {
    "log_path": "../temp/poincare/"
}

log_path = logging_dict["log_path"]

res_hdeo_hyper = find_last_embedding(log_path)

# plot_poincare(res_hdeo_hyper, dataY).show()
#
plot_poincare_zoomed(res_hdeo_hyper, dataY).show()

# i = 300
# save_poincare_teaser(find_ith_embedding(log_path, i), f"iteration_{i}_teaser.pdf", dataset=dataset, str_labels=dataY)

# save_poincare_teaser(res_hdeo_hyper, f"iteration_last_teaser.pdf", dataset=dataset)

# animate(logging_dict, dataY, f"../results/{dataset.name}_only_anim.mp4", fast=True, plot_ee=True)
