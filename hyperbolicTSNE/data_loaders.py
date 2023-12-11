#############################################################
# This file has all the facilities for loading the datasets #
# All methods should return a numpy matrix X and a labels
# vector Y if  available
##############################################################

import os
import time
import gzip
from enum import Enum, auto

from sklearn.decomposition import PCA, TruncatedSVD
import numpy as np
import scipy.sparse

from .hd_mat_ import hd_matrix, check_knn_method, check_hd_method, get_n_neighbors


# Types of dataset interfaces in scikit learn:
# - Dataset loaders
# - Dataset fetchers
# - Dataset generators

# These methods return a dict with at least two elements:
# - data: array of shape n_samples * n_features
# - target: array of shape n_samples
# if return_X_y == True, methods return (X, y)


class Datasets(Enum):
    MNIST = auto()  # DONE
    MYELOID = auto()  # DONE
    PLANARIA = auto()  # DONE
    PAUL = auto()  # DONE
    C_ELEGANS = auto()  # DONE
    LUKK = auto()  # DONE
    MYELOID8000 = auto()  # DONE
    WORDNET = auto()  # DONE


def load_mnist(data_home=None, return_X_y=True, kind='all'):
    """
    Loads different versions of the MNIST dataset. The function was taken from
    https://github.com/zalandoresearch/fashion-mnist/blob/master/utils/mnist_reader.py

    Parameters
    __________
    data_home : str, optional
        Locations of the folder where the datasets are stored.
    return_X_y: bool, optional
        If True, method only returns tuple with the data and its labels.
    kind: str, optional
        Defines if the training set (60000 points) or the test set (10000)
        is loaded.
    """

    # Use default location
    if data_home is None:
        data_home = os.path.join(os.path.dirname(__file__), 'datasets')

    full_path = os.path.join(data_home, 'mnist')

    labels_path_train = os.path.join(full_path, 'train-labels-idx1-ubyte.gz')

    labels_path_test = os.path.join(full_path, 't10k-labels-idx1-ubyte.gz')

    images_path_train = os.path.join(full_path, 'train-images-idx3-ubyte.gz')

    images_path_test = os.path.join(full_path, 't10k-images-idx3-ubyte.gz')

    labels_arr = []
    images_arr = []

    if kind == 'all' or kind == 'train':
        with gzip.open(labels_path_train, 'rb') as lbpath:
            br = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)
            labels_arr.append(br)

    if kind == 'all' or kind == 'test':
        with gzip.open(labels_path_test, 'rb') as lbpath:
            br = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)
            labels_arr.append(br)

    if kind == 'all' or kind == 'train':
        with gzip.open(images_path_train, 'rb') as imgpath:
            br = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16)
            images = br.reshape(len(labels_arr[0]), 784)
            images_arr.append(images)

    if kind == 'all' or kind == 'test':
        with gzip.open(images_path_test, 'rb') as imgpath:
            br = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16)
            images = br.reshape(len(labels_arr[1]), 784)
            images_arr.append(images)

    labels = np.concatenate(labels_arr, axis=0)
    images = np.concatenate(images_arr, axis=0)

    if return_X_y:
        return images, labels
    else:
        return images


def load_c_elegans(data_home):
    """
    Dataset from: https://data.caltech.edu/records/1945
    Similar datasets at: https://github.com/Munfred/wormcells-data
    """
    # Starting to use path instead of os TODO: move these to global imports
    from pathlib import Path
    import anndata as ad

    # Use default location
    if data_home is None:
        data_home = Path.joinpath(Path(__file__).parent, "datasets")
    else:
        data_home = Path(str(data_home))  # quick fix to deal with incoming os.paths

    full_path = Path.joinpath(data_home, "c_elegans")

    ad_obj = ad.read_h5ad(str(Path.joinpath(full_path, "packer2019.h5ad")))
    X = ad_obj.X

    labels_str = np.array(ad_obj.obs.cell_type)

    _, labels = np.unique(labels_str, return_inverse=True)

    return X, labels


def load_myeloid(data_home):
    # Starting to use path instead of os TODO: move these to global imports
    from pathlib import Path

    # Use default location
    if data_home is None:
        data_home = Path.joinpath(Path(__file__).parent, "datasets")
    else:
        data_home = Path(str(data_home))  # quick fix to deal with incoming os.paths

    full_path = Path.joinpath(data_home, "myeloid-progenitors")

    X = np.loadtxt(str(Path.joinpath(full_path, "MyeloidProgenitors.csv")), delimiter=",", skiprows=1, usecols=np.arange(11))

    labels_str = np.loadtxt(str(Path.joinpath(full_path, "MyeloidProgenitors.csv")), delimiter=",", skiprows=1, usecols=11, dtype=str)

    _, labels = np.unique(labels_str, return_inverse=True)

    return X, labels


def load_myeloid8000(data_home):
    # Starting to use path instead of os TODO: move these to global imports
    from pathlib import Path

    # Use default location
    if data_home is None:
        data_home = Path.joinpath(Path(__file__).parent, "datasets")
    else:
        data_home = Path(str(data_home))  # quick fix to deal with incoming os.paths

    full_path = Path.joinpath(data_home, "myeloid8000")

    X = np.loadtxt(str(Path.joinpath(full_path, "myeloid_8000.csv")), delimiter=",")

    labels_str = np.loadtxt(str(Path.joinpath(full_path, "myeloid_8000_labels.csv")), delimiter=",", dtype=str)

    _, labels = np.unique(labels_str, return_inverse=True)

    return X, labels


def load_planaria(data_home):
    """
    Dataset from: https://shiny.mdc-berlin.de/psca/
    """
    # Starting to use path instead of os TODO: move these to global imports
    from pathlib import Path

    # Use default location
    if data_home is None:
        data_home = Path.joinpath(Path(__file__).parent, "datasets")
    else:
        data_home = Path(str(data_home))  # quick fix to deal with incoming os.paths

    full_path = Path.joinpath(data_home, "planaria")

    X = np.loadtxt(str(Path.joinpath(full_path, "R_pca_seurat.txt")), delimiter="\t")

    labels_str = np.loadtxt(str(Path.joinpath(full_path, "R_annotation.txt")), delimiter=",", dtype=str)
    _, labels = np.unique(labels_str, return_inverse=True)

    return X, labels


def load_wordnet(data_home):
    # Starting to use path instead of os TODO: move these to global imports
    from pathlib import Path
    import torch

    # Use default location
    if data_home is None:
        data_home = Path.joinpath(Path(__file__).parent, "datasets")
    else:
        data_home = Path(str(data_home))  # quick fix to deal with incoming os.paths

    full_path = Path.joinpath(data_home, "wordnet")

    model = torch.load(str(Path.joinpath(full_path, "nouns.bin.best")))

    X = np.array(model["embeddings"])

    labels_str = np.array(model["objects"])
    _, labels = np.unique(labels_str, return_inverse=True)

    return X, labels


def load_lukk(data_home):
    # Starting to use path instead of os TODO: move these to global imports
    from pathlib import Path

    # Use default location
    if data_home is None:
        data_home = Path.joinpath(Path(__file__).parent, "datasets")
    else:
        data_home = Path(str(data_home))  # quick fix to deal with incoming os.paths

    full_path = Path.joinpath(data_home, "lukk")

    if (x := Path.joinpath(full_path, "lukk_x.npy")).exists() and (y := Path.joinpath(full_path, "lukk_y.npy")).exists():
        return np.load(str(x)), np.load(str(y))

    import pandas as pd
    sample_data_rel = pd.read_csv(
        str(Path.joinpath(full_path, "E-MTAB-62.sdrf.txt")),
        sep='\t',
        index_col='Source Name'
    )

    affymetrix = (
        pd.read_csv(
            str(Path.joinpath(full_path, "E-MTAB-62_processed_2.csv")),
            sep='\t',
            index_col='Hybridization REF',
            dtype='object',
            engine='python'
        )
        .drop('CompositeElement REF')
        .astype('float32')
        .T
        .loc[sample_data_rel.index]
    )

    X = affymetrix.values

    labels_str = sample_data_rel['Factor Value[4 groups from blood to incompletely diff]'].values
    # labels_str = sample_data_rel['Characteristics[4 meta-groups]'].values

    _, labels = np.unique(labels_str, return_inverse=True)

    return X, labels.astype(int)


def _load_dataset(dataset, data_home=None, verbose=False, **kwargs):
    X = None
    labels = None
    if verbose:
        print("[Data Loader] Preparing to load the dataset")
    start = time.time()
    if dataset == Datasets.MNIST:
        X, labels = load_mnist(data_home, **kwargs)
    if dataset == Datasets.MYELOID:
        X, labels = load_myeloid(data_home)
    if dataset == Datasets.MYELOID8000:
        X, labels = load_myeloid8000(data_home)
    if dataset == Datasets.PLANARIA:
        X, labels = load_planaria(data_home)
    if dataset == Datasets.C_ELEGANS:
        X, labels = load_c_elegans(data_home)
    if dataset == Datasets.LUKK:
        X, labels = load_lukk(data_home)
    if dataset == Datasets.WORDNET:
        X, labels = load_wordnet(data_home)
    end = time.time()
    if verbose:
        print("[Data Loader] Data has been loaded and it took {}".format(end - start))
    return X, labels


def load_data(dataset, data_home=None, to_return='all', pca_components=100,
              knn_method="sklearn", metric="euclidean", n_neighbors=None, knn_params=None,
              hd_method='vdm2008', hd_params=None,
              sample=-1, random_state=42, verbose=False, **kwargs):
    """
    Loads the selected dataset.
    Parameters
    __________
    dataset : Datasets
        The selected dataset out of the available ones.
    to_return: bool, optional
        Separate what you want to obtain with underscores, possible options are:
        `X`, `labels`, `D`, `V`. You can use `all` to obtain all the quantities
        E.g. X_labels
    pca_components : int, optional
        Number of components to take out of the PCA representation of X to build P.
        If 0, PCA is not applied and the full versions are returned.
        If >0, a reduced dataset X and its corresponding matrix (only_data=False) are returned.
        The number of dimensions of this dataset is min(X.shape[1], pca_components).
    method : str, optional
        Method to use when computing V.
        If 'exact', a point is compared against all the others.
        If 'sparse', only its NN are used and a sparse matrix (csr) is returned).
    sample: int, float optional
        Size of the sample to produce.
        If 0 < sample < 1 then denotes the fraction of the data to return.
        If sample < 1 then denotes the number of entries of X to return.
    random_state: int, optional
        Sets the random state to generate consistent samples.
        If less than 0 then random seed is not set.
    kwargs : dict
        Args to be used in specific loading methods.

    Returns
    _______
    X : ndarray
        Matrix of the data of the selected dataset.
    labels : ndarray, optional
        Array with the labels of each observation.
    D : ndarray, optional
        Matrix of distances in high-dimensional space.
    V : ndarray, optional
        Matrix of similarities.
        Can be either a ndarray in squareform (if dense) or a sparse csr matrix.
    sample_idx : ndarray, optional
        List of sampling indices.
    """
    # TODO: how to deal with parameter complexity?
    if random_state > 0:
        np.random.seed(random_state)
    D_filepath = None
    V_filepath = None

    # Load selected dataset
    if data_home is None:
        data_home = os.path.join(os.path.dirname(__file__), "datasets")
    raw_X, raw_labels = _load_dataset(dataset, data_home, verbose=verbose, **kwargs)

    # Setup sample
    sample_idx = None
    if sample <= 0:
        sample_data = False
        X, labels = raw_X, raw_labels
    else:
        sample_data = True
        if 0 < sample < 1:  # sample a fraction of the data
            sample = round(sample * raw_X.shape[0])
        # Generate sample
        raw_idx = np.arange(raw_X.shape[0])
        sample_idx = np.sort(np.random.choice(raw_idx, size=sample, replace=False))  # uniformly sampled
        X = raw_X[sample_idx].copy()
        labels = raw_labels[sample_idx].copy()

    # Preprocess data by reducing its dimensionality
    if "X" in to_return or "D" in to_return or "V" in to_return:
        if pca_components > 0:
            pca_components = np.min([pca_components, X.shape[0], X.shape[1]])
            if scipy.sparse.isspmatrix(X):
                if verbose:
                    print("[Data Loader] Input matrix X is sparse ... using sparse PCA")
                pca = TruncatedSVD(n_components=pca_components, random_state=random_state)
                X = pca.fit_transform(X)
            else:
                if verbose:
                    print("[Data Loader] Input matrix X is dense ... using dense PCA")
                pca = PCA(n_components=pca_components, svd_solver="randomized", random_state=random_state)  # remember random state
                X = pca.fit_transform(X)

    X = X.astype(np.float32, copy=False)

    to_return = "labels_X_D_V" if to_return == "all" else to_return
    to_return = to_return.split("_")
    out = []

    if verbose:
        print("[Data Loader] The following elements will be returned: {}".format(", ".join(to_return)))

    if "X" in to_return:
        out.append(X)
    if "labels" in to_return:
        out.append(labels)
    if "D" in to_return or "V" in to_return:
        D = None
        V = None
        # Here, the V matrix is the high-dimensional matrix using by each method
        # For example, in t-SNE, this matrix is the "P" matrix.
        # A V matrix file has the following convention: Vmat-dataset-method-matrix_type-pca_components-other_params.npz

        # Other parameters check
        if verbose:
            print("[Data Loader] Fetching and updating parameters of selected `knn_method`")
        knn_method, knn_params = check_knn_method(knn_method, knn_params)

        if verbose:
            print("[Data Loader] Fetching and updating parameters of selected `method`")
        hd_method, hd_params = check_hd_method(hd_method, hd_params)

        if verbose > 0:
            print("[Data Loader] Params to use for the hd_method: {}".format(hd_params))
        n_neighbors = get_n_neighbors(X.shape[0], n_neighbors, hd_method, hd_params)

        # TODO: hardcoded, streamline
        if knn_method == "hnswlib" and knn_params["search_ef"] == -1:
            if verbose > 0:
                print("[Data Loader] Using default value for `search_ef` in hnswlib: n_neighbors + 1 = {}".format(n_neighbors+1))
            knn_params["search_ef"] = n_neighbors + 1

        if not sample_data:  # If its not a sample then the matrix is cached

            def load_mat_from_cache(folder, load_data_home, load_filename):
                home_path = os.path.join(load_data_home, folder)
                if not os.path.exists(home_path):
                    os.mkdir(home_path)

                filepath = os.path.join(home_path, load_filename)
                if os.path.exists(filepath):
                    return scipy.sparse.load_npz(filepath), filepath
                else:
                    return None, filepath

            # D matrices caching
            # - Make sure a place where the V matrices are stored is available

            def D_filename(D_dataset, D_pca_components, D_knn_method, D_n_neighbors, D_metric, D_knn_params, other_args):
                fn_str = "Dmat"
                fn_str += "-dataset$%s" % str(D_dataset).split(".")[1]
                fn_str += "-%s" % ('pca$%i' % D_pca_components if (D_pca_components > 0) else 'nopca')
                fn_str += "-knn_method$%s" % str(D_knn_method)
                fn_str += "-n_neighbors$%s" % str(D_n_neighbors)
                fn_str += "-metric$%s" % str(D_metric)
                if D_knn_params is not None and type(D_knn_params) is dict:
                    for k, v in D_knn_params.items():
                        fn_str += "-%s$%s" % (str(k), str(v))
                if other_args is not None and len(other_args) > 0 and type(other_args) is dict:
                    for k, v in other_args.items(): # TODO: assumes that kwargs are valid, but this might not be the case
                        fn_str += "-%s$%s" % (str(k), str(v))
                fn_str += ".npz"
                return fn_str

            filename = D_filename(dataset, pca_components, knn_method, n_neighbors, metric, knn_params, kwargs)
            D, D_filepath = load_mat_from_cache('D_matrices', data_home, filename)

            # V matrices caching
            # - Make sure a place where the V matrices are stored is available

            def V_filename(
                    V_dataset, V_pca_components, V_knn_method, V_n_neighbors, V_metric, V_knn_params, V_hd_method,
                    V_hd_params, other_args
            ):
                fn_str = "Vmat"
                fn_str += "-dataset$%s" % str(V_dataset).split(".")[1]
                fn_str += "-%s" % ('pca$%i' % V_pca_components if (V_pca_components > 0) else 'nopca')
                fn_str += "-knn_method$%s" % str(V_knn_method)
                fn_str += "-n_neighbors$%s" % str(V_n_neighbors)
                fn_str += "-metric$%s" % str(V_metric)
                if V_knn_params is not None and type(V_knn_params) is dict:
                    for k, v in V_knn_params.items():
                        fn_str += "-%s$%s" % (str(k), str(v))
                fn_str += "-hd_method$%s" % str(V_hd_method)
                if V_hd_params is not None and type(V_hd_params) is dict:
                    for k, v in V_hd_params.items():
                        fn_str += "-%s$%s" % (str(k), str(v))
                if other_args is not None and len(other_args) > 0 and type(other_args) is dict:
                    for k, v in other_args.items(): # TODO: assumes that kwargs are valid, but this might not be the case
                        fn_str += "-%s$%s" % (str(k), str(v))
                fn_str += ".npz"
                return fn_str

            filename = V_filename(
                dataset, pca_components, knn_method, n_neighbors, metric, knn_params, hd_method, hd_params, kwargs
            )
            V, V_filepath = load_mat_from_cache('V_matrices', data_home, filename)

        if D is None or V is None:
            if verbose:
                print("[Data Loader] Either D or V was not cached, computing them now ...")
            D, V = hd_matrix(X=X, D=D, V=V,
                             knn_method=knn_method, metric=metric, n_neighbors=n_neighbors, knn_params=knn_params,
                             hd_method=hd_method, hd_params=hd_params, verbose=verbose)
            if not sample_data:
                if verbose:
                    print("[Data Loader] Caching computed matrices ...")
                scipy.sparse.save_npz(D_filepath, D)
                scipy.sparse.save_npz(V_filepath, V)

        if "D" in to_return:
            out.append(D)
        if "V" in to_return:
            out.append(V)
        if sample_idx is not None:
            out.append(sample_idx)

    if len(out) == 1:
        return out[0]
    else:
        return out
