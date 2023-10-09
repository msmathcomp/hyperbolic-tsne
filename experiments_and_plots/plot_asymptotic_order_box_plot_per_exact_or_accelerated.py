"""
This script creates a plot to compare the estimated asymptotic order of convergence for the experiments run on each
data set. The boxes are grouped by the accelerated runs, i.e, thos that use the polar quad tree, and the exact, i.e.,
the non-accelerated ones.
"""

###########
# IMPORTS #
###########

from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

####################
# READING THE DATA #
####################
results_path = Path("../results/samples_per_data_set/")
df = pd.read_csv(results_path.joinpath("overview.csv"))
timings_dfs = []
for record in df.to_records():
    timing_df = pd.read_csv(record.run_directory.replace(".", str(results_path)) + "/timings.csv")
    timing_df = timing_df[(timing_df.time_type == "tot_gradient")]

    for cn in df.columns:
        timing_df[cn] = record[cn]
    timings_dfs.append(timing_df)

timings_df = pd.concat(timings_dfs, axis=0, ignore_index=True)
timings_df["early_exag"] = np.repeat(False, timings_df.shape[0])
timings_df.loc[timings_df.it_n <= 250, "early_exag"] = True
del timings_dfs

# Compute the asymptotic estimates
grouping_vars = ["dataset", "tsne_type"]
plot_asymptotic_df = timings_df.copy()
plot_asymptotic_df = plot_asymptotic_df[(plot_asymptotic_df.splitting_strategy == "equal_length")]
plot_asymptotic_df["log_size"] = np.log(plot_asymptotic_df.sample_size)
plot_asymptotic_df["log_total_time"] = np.log(plot_asymptotic_df.total_time)
plot_asymptotic_df = plot_asymptotic_df.groupby(
    grouping_vars + ["sample_size", "log_size", ]
)["log_total_time"].agg(mu_log_total_time=np.mean, std_log_total_time=np.std).reset_index()
diff_df = plot_asymptotic_df.groupby(grouping_vars).diff(periods=-1)
diff_df.columns = ["diff_" + cn for cn in diff_df.columns]
plot_asymptotic_df = pd.concat([plot_asymptotic_df, diff_df], axis=1)
plot_asymptotic_df["asymptotic_score"] = plot_asymptotic_df.diff_mu_log_total_time/plot_asymptotic_df.diff_log_size
plot_asymptotic_df = plot_asymptotic_df.dropna(axis=0)

##############
# PLOT SETUP #
##############
sns.set_palette("colorblind")
modes = timings_df.dataset.unique()
colors = sns.color_palette('colorblind', len(modes))
palette = {mode: color for mode, color in zip(modes, colors)}

#####################
# PLOTTING THE DATA #
#####################
_, axs = plt.subplots(figsize=(5, 5), ncols=1)
asym_boxplot = sns.boxplot(
    plot_asymptotic_df,
    x="tsne_type",
    y="asymptotic_score",
    hue="dataset",
    palette=palette
)
axs.set_title(f"Estimated Asymptotic Order")
axs.set_xlabel("t-SNE Type")
axs.set_ylabel("Asymptotic Order")
plt.savefig("est_asymp_order.png")
