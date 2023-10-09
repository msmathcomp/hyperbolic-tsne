"""
This script creates a plot to show how the time spent on the computation of an embedding behaves when changing the
parameter Theta for the acceleration. One line is plotted per data set.
"""

###########
# IMPORTS #
###########

from pathlib import Path
import os
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

####################
# READING THE DATA #
####################

results_path = Path("../results/timings_per_theta/")
data = []
for subdir, dirs, files in os.walk(results_path):
    for file in files:
        if str(os.path.basename(os.path.join(subdir, file))) == "timings.csv":
            dataset = str(subdir).split('/')[-3]
            theta = float(str(subdir).split('/')[-1].split('_')[-1])
            timing_df = pd.read_csv(os.path.join(subdir, file))
            timing_df = timing_df[(timing_df.time_type == "tot_gradient")]
            average_time = float(timing_df[["total_time"]].mean())
            data.append({
                    "dataset": dataset,
                    "theta": theta,
                    "average_time": average_time
            })
average_times = pd.DataFrame(data)
average_times.loc[average_times.dataset == "LUKK", "order"] = 1
average_times.loc[average_times.dataset == "MYELOID8000", "order"] = 2
average_times.loc[average_times.dataset == "PLANARIA", "order"] = 3
average_times.loc[average_times.dataset == "MNIST", "order"] = 4
average_times.loc[average_times.dataset == "WORDNET", "order"] = 5
average_times.loc[average_times.dataset == "C_ELEGANS", "order"] = 6
average_times = average_times.sort_values(by="order", ascending=True)

##############
# PLOT SETUP #
##############
sns.set_palette("colorblind")
modes = average_times.dataset.unique()
colors = sns.color_palette('colorblind', len(modes))
palette = {mode: color for mode, color in zip(modes, colors)}
linewidth = 3.0

#####################
# PLOTTING THE DATA #
#####################

_, axs = plt.subplots(figsize=(5, 5), ncols=1, layout="tight")
times_lineplot = sns.lineplot(
    data=average_times,
    x="theta",
    y="average_time",
    hue="dataset",
    palette=palette,
    markers=False,
    linewidth=linewidth,
    ax=axs
)
times_lineplot.set(yscale='log')
axs.set_title(f"Average Total Time per Iteration vs Theta")
axs.set_xlabel("Theta")
axs.set_ylabel("log(Time (Seconds))")
plt.savefig("theta_timing_plot.png")
