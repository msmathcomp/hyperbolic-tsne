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
    "umap": {}
}

available_hd_methods_names = list(available_hd_methods.keys())


def check_knn_method(method, params):
    if method not in available_knn_methods:
        raise ValueError("Chosen knn method is not a valid one, valid methods: {}".format(available_knn_methods_names))
    ps = available_knn_methods[method]
    if params is not None:
        if type(params) is dict:
            for k, v in params.items():
                if k not in ps:
                    raise ValueError("A key in the hd_params dict is not a valid one. "
                                     "Valid keys for the selelected method: {}".format(list(available_knn_methods[method].keys())))
                else:
                    ps[k] = v
        else:
            raise ValueError("knn_params should be a dict")
    return method, ps


def check_hd_method(method, params):
    if method not in available_hd_methods:
        raise ValueError("Chosen hd method is not a valid one, valid methods: {}".format(available_hd_methods_names))
    ps = available_hd_methods[method]
    if params is not None:
        if type(params) is dict:
            for k, v in params.items():
                if k not in ps:
                    raise ValueError("A key in the hd_params dict is not a valid one. "
                                     "Valid keys for the selelected method: {}".format(list(available_hd_methods[method].keys())))
                else:
                    ps[k] = v
        else:
            raise ValueError("hd_params should be a dict")
    return method, ps


def get_n_neighbors(n_samples, n_neighbors, hd_method, other_params):
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
                print("[hd_mat] `hd_method` set to `vdm2008`, running with perplexity {}. Returns (D, V)".format(hd_params["perplexity"]))
            V = _vdm2008(D, verbose=verbose, n_jobs=n_jobs, **hd_params)
        elif hd_method == "umap":
            if verbose>0:
                print("[hd_mat `hd_method` set to `umap`") # TODO: show default params
            V = _umap(D, n_neighbors=n_neighbors, verbose=verbose, n_jobs=n_jobs)

    return D, V


############################################################
# D matrix computation which is the base of all DR methods #
############################################################

def _distance_matrix(X, method="sklearn", n_neighbors=None, metric="euclidean", other_params=None, verbose=0, n_jobs=1):

    if n_neighbors is None:
        raise Exception("For computing the D kNN matrix, `n_neighbors` should be a number larger than 0")

    if other_params is None:
        other_params = {}

    if verbose > 0:
        print("[hd_mat] Computing the kNN D matrix with k={} nearest neighbors...".format(n_neighbors))

    if method == "sklearn":
        if verbose > 0:
            print("[hd_mat] Using sklearn NearestNeighbor, an exact method, for the knn computation")

        n_samples = X.shape[0]

        # Find the nearest neighbors for every point
        knn = NearestNeighbors(algorithm='auto', n_jobs=n_jobs, n_neighbors=n_neighbors, metric=metric)

        t0 = time()
        knn.fit(X)
        duration = time() - t0
        if verbose:
            print("[hd_mat] Indexed {} samples in {:.3f}s...".format(n_samples, duration))

        t0 = time()
        distances_nn = knn.kneighbors_graph(mode='distance')
        duration = time() - t0
        if verbose:
            print("[hd_mat] Computed neighbors for {} samples ""in {:.3f}s...".format(n_samples, duration))

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
            print("[hd_mat] Using hnswlib, an approximate method, for the knn computation")

        if metric not in ["euclidean","l2","cosine"]:
            raise Exception("Approximate kNN calculation based on hnswlib supports euclidean (l2) and cosine distances")
        if metric == "euclidean":
            metric = "l2"

        n_samples, n_dim = X.shape

        # parse other params that this method uses
        # https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md
        M = other_params.get("M", 48) # increase if intrinsic n_dim is large / want more recall -> more memory consumption
        construction_ef = other_params.get("construction_ef",200)
        search_ef = other_params.get("search_ef",n_neighbors+1) # n_samples-1 for max precision
        if len(other_params) == 0 and verbose > 0:
            print("[hd_mat] Using default parameters for the knn computation. M: {}, construction_ef: {}, search_ef: {}"
                  .format(M, construction_ef, search_ef))

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

    D = np.sqrt(D) # TODO: for debugging purposes, remove
    return D


##############
# V matrices #
##############

def _vdm2008(distances, perplexity=50, verbose=0, n_jobs=1):
    """Compute joint probabilities v_ij from distances using just nearest
        neighbors. The output of this method is the high-dimensional matrix V
        that tSNE uses.
        This method takes advantage of the sparsity structure of the distance
        matrix yielding a complexity of O(uN) (vs O(N^2) with the dense matrix).
        Parameters
        ----------
        distances : CSR sparse matrix, shape (n_samples, n_samples)
            Distances of samples to its n_neighbors nearest neighbors. All other
            distances are left to zero (and are not materialized in memory).
        perplexity : float
            Desired perplexity of the joint probability distributions.
        verbose : int
            Verbosity level.
        Returns
        -------
        V : csr sparse matrix, shape (n_samples, n_samples)
        Condensed joint probability matrix with only nearest neighbors.
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
        print("[t-SNE] Computed conditional probabilities in {:.3f}s"
              .format(duration))

    return V


NPY_INFINITY = np.inf
UMAP_SMOOTH_K_TOLERANCE = 1e-5
UMAP_MIN_K_DIST_SCALE = 1e-3


def _umap(distances, n_neighbors, n_iter=100, bandwidth=1.0, verbose=0, n_jobs=1):
    """
    TODO: reformat
    distances = sparse nn_distance matrix with distances.shape[1] neighbors
    n_neighbors = neighbors to use for the calibration
    n_iter = number of iterations for the line search
    """

    distances.sort_indices()
    n_samples, n_knn = distances.shape

    dists_data = distances.data.astype(np.float32, copy=False)
    dists_indices = distances.indices
    dists_indptr = distances.indptr

    # 1. Find sigma_i and rho_i per sample (Algorithm 3 in paper)
    # - In UMAP repo this part corresponds to the function "smooth_knn_dist"
    print("Computing rho_i and sigma_i for UMAP's V matrix")

    rhos = np.zeros((n_samples, 1))
    sigmas = np.zeros((n_samples, 1))
    mean_distances = np.mean(dists_data)

    # Iterate over every row of the dist matrix (ith point)
    # TODO: parallelize with numba
    for i in range(n_samples):
        start_indptr = dists_indptr[i]
        end_indptr = dists_indptr[i+1]
        ith_dists = dists_data[start_indptr:end_indptr]

        # rho_i (other fancy things done in the paper)
        non_zero_dists = ith_dists[ith_dists > 0]
        rhos[i] = np.min(non_zero_dists)

        # sigma_i
        # - find sigma_i such that sum (-(v_i-rho_i)/sigma_i) = log_2(n_neighbors)
        # - uses line search for this
        target = np.log2(n_neighbors) * bandwidth
        lo = 0.0
        mid = 1.0
        hi = NPY_INFINITY

        for it in range(n_iter):
            psum = 0.0

            for j in range(ith_dists.size): # iterate over neighbors of point i
                d = ith_dists[j] - rhos[i]
                if d > 0:
                    psum += np.exp(-(d/mid))
                else:
                    psum += 1.0

            # sigma is close enough to tolerance, break iterations
            if np.fabs(psum-target) < UMAP_SMOOTH_K_TOLERANCE:
                break

            # binary search
            if psum > target:
                hi = mid
                mid = (lo + hi) / 2.0
            else:
                lo = mid
                if hi == NPY_INFINITY:
                    mid *= 2
                else:
                    mid = (lo + hi) / 2.0

        sigmas[i] = mid

        if rhos[i] > 0.0:
            mean_ith_distances = np.mean(ith_dists)
            if sigmas[i] < UMAP_MIN_K_DIST_SCALE * mean_ith_distances:
                sigmas[i] = UMAP_MIN_K_DIST_SCALE * mean_ith_distances
        else:
            if sigmas[i] < UMAP_MIN_K_DIST_SCALE * mean_distances:
                sigmas[i] = UMAP_MIN_K_DIST_SCALE * mean_distances

        print(" - point {} | its = {} | rho = {} | sigma = {}".format(i, it, rhos[i], sigmas[i]))

    # 2. Compute conditional matrix V_j|i (Algorithm 2 in paper)
    # - v_j|i = exp[(-d(x_i,x_j)-rho_i)/sigma_i]
    # - In UMAP repo this part corresponds to the function "compute_memberships_strengths"

    V_data = distances.data.copy()
    V_indices = distances.indices.copy()
    V_indptr = distances.indptr.copy()

    for i in range(n_samples):
        start_indptr = dists_indptr[i]
        end_indptr = dists_indptr[i+1]
        rho = rhos[i]
        sigma = sigmas[i]
        vals = np.zeros_like(V_data[start_indptr: end_indptr])

        for j, (j_index, v) in enumerate(zip(V_indices[start_indptr: end_indptr], V_data[start_indptr: end_indptr])):
            if i == j_index:
                v = 0
            elif v - rho <= 0.0 or sigma == 0.0:
                v = 1.0 # cap the v so it cannot go above 1, which happens if v - rho <= 0
            else:
                v = np.exp(-((v-rho)/sigma)) # Used as on GitHub (different from paper)
            vals[j] = v

        V_data[start_indptr: end_indptr] = vals

    V = csr_matrix((V_data, V_indices, V_indptr), shape=distances.shape)
    V.eliminate_zeros()

    # 3. Symmetrize matrix V_ij = (V_j|i + V_j|i^T) - V_j|i o V_j|i^T (Algorithm 1 in paper)
    # - Here I only consider the "union" operator but in the code "intersection" is also possible

    V = (V + V.transpose()) - V.multiply(V.transpose())

    return V