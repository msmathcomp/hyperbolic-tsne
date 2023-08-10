"""
Using the overview_part.csv files, creates a single overview 
"""

from pathlib import Path
import csv

BASE_DIR = "../results/exp_grid"

header_row = []
rows = []

for p in Path(BASE_DIR).rglob("overview_part.csv"):
    with open(p, "r") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i == 0:
                header_row = row
            else:
                rows.append(row)


# Create or append to overview csv file after every run
overview_path = Path(BASE_DIR).joinpath("overview.csv")
print(overview_path)
if overview_path.exists():
    print(f"Exists, did not write {overview_path}")
else:

    with open(overview_path, "w", newline="") as overview_file:
        overview_writer = csv.writer(overview_file)

        # Header
        overview_writer.writerow(header_row)

        # Rest of the rows
        for row in rows:
            overview_writer.writerow(row)

    print("Done")