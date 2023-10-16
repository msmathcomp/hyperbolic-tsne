""" Methods for processing distance and affinity matrices.
"""
from time import time
import numpy as np

from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

import hnswlib

from .hyperbolic_barnes_hut import tsne_utils


MACHINE_EPSILON = np.finfo(np.double).eps

available_knn_methods = {
    "sklearn": {},
    "hnswlib": {"M": 48, "construction_ef": 200, "search_ef": -1}
}

available_knn_methods_names = list(available_knn_methods.keys())

available_hd_methods = {
    "vdm2008": {"perplexity": 50},
}

available_hd_methods_names = list(available_hd_methods.keys())


def check_knn_method(method, params):
    """Checks that provided method and params dict are valid.

    Parameters
    ----------
    method : str
        kNN method name.
    params : dict
        kNN method params in key-value format.

    Returns
    -------
    (string, dict)
        Method name and method params in key-value format.
    """    
    if method not in available_knn_methods:
        raise ValueError(f"Chosen knn method is not a valid one, valid methods: {available_knn_methods_names}")
    ps = available_knn_methods[method]
    if params is not None:
        if isinstance(params, dict):
            for k, v in params.items():
                if k not in ps:
                    raise ValueError(f"A key in the hd_params dict is not a valid one. Valid keys for the selelected method: {list(available_knn_methods[method].keys())}")               
                ps[k] = v
        else:
            raise TypeError("knn_params should be a dict")
    return method, ps


def check_hd_method(method, params):
    """Checks that provided method and params dict are valid.

    Parameters
    ----------
    method : str
        HD method name.
    params : dict
        HD method params in key-value format.

    Returns
    -------
    (string, dict)
        Method name and method params in key-value format.
    """
    if method not in available_hd_methods:
        raise ValueError(f"Chosen hd method is not a valid one, valid methods: {available_hd_methods_names}")
    ps = available_hd_methods[method]
    if params is not None:
        if isinstance(params, dict):
            for k, v in params.items():
                if k not in ps:
                    raise ValueError(f"A key in the hd_params dict is not a valid one. Valid keys for the selelected method: {list(available_hd_methods[method].keys())}")
                ps[k] = v
        else:
            raise ValueError("hd_params should be a dict")
    return method, ps


def get_n_neighbors(n_samples, n_neighbors, hd_method, other_params):
    """Returns the number of neighbors for a specific method.

    Parameters
    ----------
    n_samples : int
        Number of samples.
    n_neighbors : int
        Default number of neighbors.
    hd_method : str
        HD method name.
    other_params : dict
        Parameters of HD method in key-value format.

    Returns
    -------
    int
        Effective number of neighbors to use.
    """
    if hd_method == "vdm2008":
        perplexity = other_params.get("perplexity", None)
        if perplexity is None:
            raise ValueError("For method vdm2008 parameter perplexity is needed")
        return min(n_samples - 1, int(3. * perplexity + 1))
    else:
        return n_neighbors


def hd_matrix(*, X=None, D=None, V=None,
              knn_method="sklearn", metric="euclidean", n_neighbors=100, knn_params=None,  # params for D matrix
              hd_method="vdm2008", hd_params=None,  # params for V matrix
              verbose=0, n_jobs=1):
    """Computes the affinity matrix of a high-dimensional dataset X. 
    If precomputed distances D are provided, then these are not computed.
    If affinities V are provided, then no computations are performed.

    Parameters
    ----------
    X : ndarray, optional
        High-dimensional data matrix, by default None.
    D : ndarray, optional
        Distance matrix, by default None.
    V : ndarray, optional
        Affinity matrix, by default None.
    knn_method : str, optional
        Name of kNN method, by default "sklearn".
    metric : str, optional
        Metric for distance computation, by default "euclidean".
    n_neighbors : int, optional
        Number of neighbors to consider, by default 100.
    knn_params : dict, optional
        kNN method params in key-value format, by default None.
    hd_params : dict, optional
        HD method params in key-value format, by default None.
    n_jobs : int, optional
        Number of jobs used to run the process, by default 1.

    Returns
    -------
    (ndarray, ndarray)
        Tuple with distance and affinity matrices, which 
        are numpy ndarrays of shape num_samples x num_samples.
    """
    # Check general params
    if X is not None:
        n_samples = X.shape[0]
    elif D is not None:
        n_samples = D.shape[0]
    elif V is not None:
        n_samples = V.shape[0]
    else:
        raise ValueError("X, D and V are None, nothing do do ...")

    D_precomputed = D is not None
    V_precomputed = V is not None
    compute_D = False
    compute_V = False

    if V_precomputed or \
        (D_precomputed and V_precomputed) or \
        (X is None and not D_precomputed):
        print("[hd_mat] Warning: There is nothing to do with given parameters. Returning given D and V")
        return D, V

    if X is not None:
        if not D_precomputed:
            compute_D = True
            if knn_method not in ["sklearn", "hnswlib"]:
                raise ValueError("Unsupported kNN method")

    if hd_method is not None:
        hd_method, hd_params = check_hd_method(hd_method, hd_params)
        compute_V = True

    # Compute D
    n_neighbors = get_n_neighbors(n_samples, n_neighbors, hd_method, hd_params)
    if compute_D:
        D = _distance_matrix(X, method=knn_method, metric=metric, n_neighbors=n_neighbors, other_params=knn_params,
                             verbose=verbose, n_jobs=n_jobs)
    # Compute V
    if compute_V:
        if hd_method == "vdm2008":
            if verbose>0:
                print(f"`hd_method` set to `vdm2008`, running with perplexity {hd_params['perplexity']}. Returns (D, V)")
            V = _vdm2008(D, verbose=verbose, n_jobs=n_jobs, **hd_params)

    return D, V


############################################################
# D matrix computation which is the base of all DR methods #
############################################################

def _distance_matrix(X, method="sklearn", n_neighbors=None, metric="euclidean", other_params=None, verbose=0, n_jobs=1):
    """Computes the distance matrix of a high-dimensional dataset X. 

    Parameters
    ----------
    X : ndarray
        High-dimensional data matrix, by default None.
    method : str, optional
        Name of kNN method, by default "sklearn".
    n_neighbors : int, optional
        Number of neighbors to consider, by default 100.
    metric : str, optional
        Metric for distance computation, by default "euclidean".
    other_params : dict, optional
        kNN method params in key-value format, by default None.
    verbose : int, optional
        Verbosity level, by default 0.
    n_jobs : int, optional
        Number of jobs used to run the process, by default 1.

    Returns
    -------
    ndarray
        Computed distance matrix.
    """

    if n_neighbors is None:
        raise ValueError("For computing the D kNN matrix, `n_neighbors` should be a number larger than 0")

    if other_params is None:
        other_params = {}

    if verbose > 0:
        print(f"Computing the kNN D matrix with k={n_neighbors} nearest neighbors...")

    if method == "sklearn":
        if verbose > 0:
            print("Using sklearn NearestNeighbor, an exact method, for the knn computation")

        n_samples = X.shape[0]

        # Find the nearest neighbors for every point
        knn = NearestNeighbors(algorithm='auto', n_jobs=n_jobs, n_neighbors=n_neighbors, metric=metric)

        t0 = time()
        knn.fit(X)
        duration = time() - t0
        if verbose:
            print(f"Indexed {n_samples} samples in {duration:.3f}s...")

        t0 = time()
        distances_nn = knn.kneighbors_graph(mode='distance')
        duration = time() - t0
        if verbose:
            print(f"Computed neighbors for {n_samples} samples ""in {duration:.3f}s...")

        # Free the memory used by the ball_tree
        del knn

        if metric == "euclidean":
            # knn return the euclidean distance but we need it squared
            # to be consistent with the 'exact' method. Note that the
            # the method was derived using the euclidean method as in the
            # input space. Not sure of the implication of using a different
            # metric.
            distances_nn.data **= 2

        D = distances_nn

    elif method == "hnswlib":
        if verbose > 0:
            print("Using hnswlib, an approximate method, for the knn computation")

        if metric not in ["euclidean","l2","cosine"]:
            raise ValueError("Approximate kNN calculation based on hnswlib supports euclidean (l2) and cosine distances")
        if metric == "euclidean":
            metric = "l2"

        n_samples, n_dim = X.shape

        # parse other params that this method uses
        # https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md
        M = other_params.get("M", 48) # increase if intrinsic n_dim is large / want more recall -> more memory consumption
        construction_ef = other_params.get("construction_ef",200)
        search_ef = other_params.get("search_ef",n_neighbors+1) # n_samples-1 for max precision
        if len(other_params) == 0 and verbose > 0:
            print(f"[hd_mat] Using default parameters for the knn computation. M: {M}, construction_ef: {construction_ef}, search_ef: {search_ef}")

        # construction
        index = hnswlib.Index(space = metric, dim=n_dim)
        index.init_index(max_elements=n_samples, ef_construction=construction_ef, M=M)
        index.add_items(X)

        # search
        index.set_ef(search_ef)
        labels, distances = index.knn_query(X, k=n_neighbors+1)
        labels = labels[:, 1:]
        distances = distances[:, 1:]

        D = csr_matrix((distances.ravel(), labels.ravel(), np.arange(0,distances.size+1,n_neighbors)),
                       shape=(n_samples, n_samples))

    return D


##############
# V matrices #
##############

def _vdm2008(distances, perplexity=50, verbose=0, n_jobs=1):
    """Computation of affinity matrix as described in TSNE paper. 
    Compute joint probabilities v_ij from distances using just nearest
    neighbors. This method takes advantage of the sparsity structure of 
    the distance matrix yielding a complexity of O(uN) (vs O(N^2) with 
    the dense matrix).        

    Parameters
    ----------
    distances : csr sparse matrix
        Distance matrix.
    perplexity : float, optional
        Perplexity as defined in TSNE paper, by default 50. It determines the
        number of neighbors.
    verbose : int, optional
        Verbosity level, by default 0.
    n_jobs : int, optional
        Number of jobs used to run the process, by default 1.

    Returns
    -------
    csr sparse matrix
        Computed affinity matrix.
    """

    t0 = time()
    # Compute conditional probabilities such that they approximately match
    # the desired perplexity
    distances.sort_indices()
    n_samples = distances.shape[0]
    distances_data = distances.data.reshape(n_samples, -1)
    distances_data = distances_data.astype(np.float32, copy=False)
    conditional_V = tsne_utils._binary_search_perplexity(distances_data, perplexity, verbose)
    assert np.all(np.isfinite(conditional_V)), \
        "All probabilities should be finite"

    # Symmetrize the joint probability distribution using sparse operations
    V = csr_matrix((conditional_V.ravel(), distances.indices,
                    distances.indptr),
                   shape=(n_samples, n_samples))
    V = V + V.T

    # Normalize the joint probability distribution
    sum_V = np.maximum(V.sum(), MACHINE_EPSILON)
    V /= sum_V

    assert np.all(np.abs(V.data) <= 1.0)
    if verbose >= 2:
        duration = time() - t0
        print(f"Computed conditional probabilities in {duration:.3f}s")

    return V
