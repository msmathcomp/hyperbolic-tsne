"""
This script creates a plot to show how the average time per iteration behaves with respect to the size of the data set
to be embedded. That is, for each data set, a line plot is drawn showing how the average iteration time behaves for its
samples. The lines are grouped according to the type of t-SNE: exact t-SNE that runs in quadratic time or accelerated
t-SNE that makes use of the polar quad tree data structure. To better compare this, trend lines are fitted to the line
plots.
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

# Work with the "equal length" data, as this splitting technique proved to be more efficient, filtering by
# "equal_length" contains both accelerated and exact data.
plot_times_df = timings_df.copy()
plot_times_df = plot_times_df[(plot_times_df.splitting_strategy == "equal_length")]

# Perform linear regression on the exact timinigs to a quadratic model
plot_times_df_exact = plot_times_df.copy()
plot_times_df_exact = plot_times_df_exact[(plot_times_df_exact.tsne_type == "exact")]
x = plot_times_df_exact["sample_size"].values
y = plot_times_df_exact["total_time"].values
model = np.poly1d(np.polyfit(x, y, 2))
x_predicted = np.logspace(2.65, 5.1, num=10)
# Only use the leading term, i.e., the x^2 term, to fit the line
y_predicted = [model.coef[0] * (a ** 2) for a in x_predicted]

_, axs = plt.subplots(figsize=(5, 5), ncols=1)
# Plot a trendline for the quadratic run times
axs.plot(
    x_predicted,
    y_predicted,
    color=(0, 0, 0, 1.0)
)

# Perform linear regression on the accelerated timings to an n*log(n) model
plot_times_df_accelerated = plot_times_df.copy()
plot_times_df_accelerated = plot_times_df_accelerated[(plot_times_df_accelerated.tsne_type == "accelerated")]
x = plot_times_df_accelerated["sample_size"].values
y = plot_times_df_accelerated["total_time"].values
model = np.poly1d(np.polyfit(np.log(x)*x, y, 1))
y_predicted = [model.coef[0] * a * np.log(a) for a in x_predicted]

# Plot a trendline for the lin-log run times
axs.plot(
    x_predicted,
    y_predicted,
    color=(0, 0, 0, 1.0),
    linestyle="dashed"
)

times_lineplot = sns.lineplot(
    data=timings_df,
    x="sample_size",
    y="total_time",
    hue="dataset",
    style="tsne_type",
    palette=palette,
    markers=False,
    linewidth=linewidth,
    ax=axs)
times_lineplot.set(xscale='log', yscale='log')

axs.set_title(f"Average Total Time per Iteration vs Dataset Size")
axs.set_xlabel("Log(Sample Size)")
axs.set_ylabel("Log(Time (Seconds))")
plt.tight_layout()
plt.savefig("time_per_it_vs_data_size_grouped_by_exact_or_accelerated.png")
