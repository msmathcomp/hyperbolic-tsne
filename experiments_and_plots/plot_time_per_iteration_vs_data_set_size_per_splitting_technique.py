"""
This script creates a plot to show how the average time per iteration behaves with respect to the size of the data set
to be embedded. That is, for each data set, a line plot is drawn showing how the average iteration time behaves for its
samples. All data shown is from accelerated iterations, using two different splitting strategies: by "equal area" and by
"equal length". The lines are grouped according to the splitting strategies.
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

##############
# PLOT SETUP #
##############
sns.set_palette("colorblind")
modes = timings_df.dataset.unique()
colors = sns.color_palette('colorblind', len(modes))
palette = {mode: color for mode, color in zip(modes, colors)}
linewidth = 3.0

#####################
# PLOTTING THE DATA #
#####################

# Only work with the "accelerated" data, as we want to compare the different splitting strategies
plot_times_df = timings_df.copy()
plot_times_df = plot_times_df[(plot_times_df.tsne_type == "accelerated")]

_, axs = plt.subplots(figsize=(5, 5), ncols=1)
times_lineplot = sns.lineplot(
    plot_times_df,
    x="sample_size",
    y="total_time",
    hue="dataset",
    style="splitting_strategy",
    palette=palette,
    dashes=False,
    markers=True,
    linewidth=linewidth,
    ax=axs
)
times_lineplot.set(xscale='log', yscale='log')
axs.set_title(f"Average Total Time per Iteration vs Dataset Size")
axs.set_xlabel("Log(Sample Size)")
axs.set_ylabel("Time (Seconds)")
plt.tight_layout()
plt.savefig("time_per_it_vs_data_size_grouped_by_splitting_technique.png")
