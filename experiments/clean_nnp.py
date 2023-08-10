"""
Removes data from exp_nnp script
"""

from pathlib import Path

BASE_DIR = Path("../results/exp_grid")

for fn in ["thresholds.npy", "precisions.npy", "recalls.npy", "true_positives.npy", "prec-vs-rec.png"]:
    print(fn)
    for p in BASE_DIR.rglob(f"*{fn}"):
        print(f"Removing: {p}")
        p.unlink()
