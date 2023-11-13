"""
This script iterates over the results created by the `samples per data set` script. For each final exact embedding,
it extracts the exact cost function value, finds the corresponding accelerated embeddings, and extracts their respective
cost function values. From these cost function values, a relative error is computed and the statistics across this
relative errors are appended to the original data, which is then stored.
"""

###########
# IMPORTS #
###########

from pathlib import Path
import numpy as np
import pandas as pd


######################
# HELPER FUNCTION(S) #
######################

def max_iteration_cost_function(df_record):
    record_path = df_record.run_directory.replace(".", str(results_path))
    max_it = 0
    max_it_cf = 0
    for path in list(Path(f"{record_path}/embeddings/").rglob("*.csv")):
        if path.name.startswith("._"):
            continue
        (iteration_number, cost_function_value) = str(path.stem).split(',')
        if int(iteration_number) > max_it:
            max_it = int(iteration_number)
            max_it_cf = float(cost_function_value)
    return max_it, max_it_cf


####################
# READING THE DATA #
####################

# Use the data generated from the various runs on the data sets with several sample sizes
results_path = Path("../results/samples_per_data_set/")

# Load the overview of the experiments
df = pd.read_csv(results_path.joinpath("overview.csv"))

# For each dataset, find the maximum sample size, i.e., the size of the data set
maxes = df.groupby(["dataset"]).sample_size.transform(max)

# Filter the data frame to only include these
df = df[df.sample_size == maxes]

# Filter to only include one instance of the exact solutions (as they all have the same results)
exact_results = df[(df.tsne_type == "exact")].groupby(["dataset"]).first().reset_index()
accelerated_results = df[(df.tsne_type == "accelerated") & (df.splitting_strategy == "equal_length")]

# Iterate over the exact solutions and compare to the various approximated ones
for i, record in enumerate(exact_results.to_records()):
    # Get the maximum iteration number and corresponding cost function for this exact record
    max_iteration, max_iteration_cf = max_iteration_cost_function(record)
    print(f"{record.dataset} {record.tsne_type} {max_iteration} {max_iteration_cf}")

    # Array to store the relative approximated cost function errors
    cf_errors_relative = []

    # Iterate over all accelerated results for the current data set
    for accelerated_record in accelerated_results[accelerated_results.dataset == record.dataset].to_records():
        # Get the maximum iteration number and corresponding cost function for the accelerated record
        max_iteration_accelerated, max_iteration_cf_accelerated = max_iteration_cost_function(accelerated_record)
        # Ensure that this record is for the correct data set and that it ran for the same number of iterations
        if accelerated_record.dataset != record.dataset or max_iteration_accelerated != max_iteration:
            raise (
                f"When processing {record.dataset}, could not find an accelerated result with {max_iteration} "
                f"iterations, found one with {max_iteration_accelerated} iterations."
            )
        print(
            f"{accelerated_record.dataset} {accelerated_record.tsne_type} {max_iteration_accelerated} "
            f"{max_iteration_cf_accelerated}"
        )
        # Compute the relative error of the cost function
        cf_error_relative = np.abs(max_iteration_cf - max_iteration_cf_accelerated) / max_iteration_cf
        cf_errors_relative.append(cf_error_relative)

    # Compute descriptive statistics on the cost function error
    avg_cf_errors_relative = pd.DataFrame(cf_errors_relative).describe().T.add_prefix("cf_")
    for name, values in avg_cf_errors_relative.items():
        exact_results.at[i, name] = values.item()

    # Store results
    exact_results.to_csv(results_path.joinpath("relative_cost_function_errors.csv"), index=False)

    print(f"Finished processing {record.dataset}.")
