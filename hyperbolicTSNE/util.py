import os

import numpy as np


def find_last_embedding(log_path):
    """ Give a path with logging results, finds the last embedding saved there.
    """
    for subdir, dirs, files in reversed(list(os.walk(log_path))):
        files = [f for f in files if not f[0] == '.']
        dirs[:] = [d for d in dirs if not d[0] == '.']

        for fi, file in enumerate(reversed(sorted(files, key=lambda x: int(x.split(", ")[0])))):
            root, ext = os.path.splitext(file)
            if ext == ".csv":
                total_file = subdir.replace("\\", "/") + "/" + file

                return np.genfromtxt(total_file, delimiter=',')


def find_ith_embedding(log_path, i):
    """ Give a path with logging results, finds the i-th embedding saved there.
    """
    j = 0
    for subdir, dirs, files in list(os.walk(log_path)):
        files = [f for f in files if not f[0] == '.']
        dirs[:] = [d for d in dirs if not d[0] == '.']

        for fi, file in enumerate(sorted(files, key=lambda x: int(x.split(", ")[0]))):
            root, ext = os.path.splitext(file)
            if ext == ".csv":
                j += 1

                if j >= i:
                    total_file = subdir.replace("\\", "/") + "/" + file

                    return np.genfromtxt(total_file, delimiter=',')
