""" Quality metrics for the embeddings.
"""

import numpy as np

from .hd_mat_ import _distance_matrix
from hyperbolicTSNE.hyperbolic_barnes_hut.tsne import distance_py


def hyperbolic_nearest_neighbor_preservation(X, Y, k_start=1, k_max=15, D_X=None, exact_nn=False,
                                             consider_order=False, strict=False, to_return="aggregated",
                                             k_hyper_approx=100):
    """
    Approximates the nearest neighbor preservation of the hyperbolic embedding `Y` in the poincare disk
    of the high-dimensional data `X`. See Section 6 of
    Pezzotti, Nicola, et al. "Hierarchical stochastic neighbor embedding."
    Computer Graphics Forum. Vol. 35. No. 3. 2016. for a detailed description of the approach.

    Parameters
    ----------
    X: ndarray
        Coordinates in original -high dimensional- space.
    Y: ndarray
        Coordinates in embedded -lower dimensional- hyperbolic space.
    k_start: int
        Neighborhood size from which the TPs will be computed. Should hold k_start < k_max
    k_max: int
        Size of neighborhoods to consider.
    D_X: ndarray
        Precomputed distance matrix for the high-dimensional points.
        If not None then this matrix will be used instead of computing it.
    exact_nn: ndarray
        Whether to use exact or approximate nearest neighbors for building the
        distance matrices. Approximate methods might introduce noise in the
        precision/recall curves.
    consider_order: bool
        if True, the ordered neighborhoods will be compared for obtaining the TPs.
        If False, the neighborhoods will be compared using set intersection (order
        agnostic).
    strict: bool
        If True, TPs will be computed based on k neighborhood in the high dimensional
        space. If False, the k_max neighborhood will be used.
    to_return: str
        Which values should be returned. Either `aggregate` or `full`.

    Returns
    -------
    thresholds: list
        Thresholds used for computing precisions and recalls.
    precisions: list
        Precisions for every threshold averaged over all the points.
        For a single point, precision = TP/threshold
    recalls: list
        Recalls for every threshold averaged over all the points.
        For a single point, recall = TP/k_max
    nums_true_positives: list of lists
        Number of true positives for every threshold value and every point.
    """

    # Compute kNN neighborhoods
    if exact_nn:
        nn_method = "sklearn"
    else:
        nn_method = "hnswlib"

    if D_X is None:
        D_X = _distance_matrix(X, method=nn_method, n_neighbors=k_max)
    D_Y = _distance_matrix(Y, method=nn_method, n_neighbors=k_hyper_approx)

    num_points = D_X.shape[0]
    num_available_nn_hd = D_X[0, :].nnz
    num_available_nn_ld = D_Y[0, :].nnz

    # Check for potential problems
    size_smaller_neighborhood = min(num_available_nn_ld, num_available_nn_hd)
    if size_smaller_neighborhood < k_max:
        print("[nearest_neighbor_preservation] Warning: k_max is {} but the size of the available neighborhoods is {}."
              "Adjusting k_max to the available neighbors.".format(k_max, size_smaller_neighborhood))
        k_max = size_smaller_neighborhood
    if k_start > k_max:
        raise Exception("[nearest_neighbor_preservation] Error: k_start is larger than k_max. Please adjust"
                        " this value to satisfy k_start <= k_max.")
    if k_start <= 0:
        print("[nearest_neighbor_preservation] Warning: k_start must be a value above 0. Setting it to 1.")
        k_start = 1

    # Compute precision recall curves for every value of k_emb
    precisions = []
    recalls = []

    # Computation of ordered neighbourhoods
    nz_D_X = D_X.nonzero()
    nz_rows_D_X = nz_D_X[0].reshape(-1, num_available_nn_hd)  # row coordinate of nz elements from D_X
    nz_cols_D_X = nz_D_X[1].reshape(-1, num_available_nn_hd)  # col coordinate of nz elements from D_X
    nz_dists_D_X = np.asarray(D_X[nz_rows_D_X, nz_cols_D_X].todense())
    sorting_ids_nz_dists_D_X = np.argsort(nz_dists_D_X, axis=1)
    sorted_nz_cols_D_X = nz_cols_D_X[nz_rows_D_X, sorting_ids_nz_dists_D_X]  # sorted cols of nz_D_X
    sorted_nz_cols_D_X = sorted_nz_cols_D_X[:, 0:k_max]  # only get NNs that will be used

    nz_D_Y = D_Y.nonzero()
    nz_rows_D_Y = nz_D_Y[0].reshape(-1, num_available_nn_ld)  # row coordinate of nz elements from D_Y
    nz_cols_D_Y = nz_D_Y[1].reshape(-1, num_available_nn_ld)  # col coordinate of nz elements from D_Y
    nz_dists_D_Y = np.asarray(D_Y[nz_rows_D_Y, nz_cols_D_Y].todense())
    sorting_ids_nz_dists_D_Y = np.argsort(nz_dists_D_Y, axis=1)
    sorted_nz_cols_D_Y = nz_cols_D_Y[nz_rows_D_Y, sorting_ids_nz_dists_D_Y]  # sorted cols of nz_D_Y

    # Replace with hyperbolic distances for the closest k_hyper_approx
    sorted_nz_cols_D_Y = sorted_nz_cols_D_Y[:, 0:k_hyper_approx]

    arr = np.zeros(sorted_nz_cols_D_Y.shape)
    for (i, j), v in np.ndenumerate(sorted_nz_cols_D_Y):
        arr[i, j] = distance_py(Y[i], Y[v])

    sorting_ids_arr = np.argsort(arr, axis=1)
    sorted_nz_cols_D_Y = sorted_nz_cols_D_Y[nz_rows_D_Y, sorting_ids_arr]

    sorted_nz_cols_D_Y = sorted_nz_cols_D_Y[:, 0:k_max]  # only get NNs that will be used

    # Compute metrics
    thresholds = np.arange(k_start, k_max+1)
    nums_true_positives = []
    index_hd = np.arange(0, k_max)

    for k_emb in thresholds:
        tps = []
        for point_id in range(num_points):  # point_id between 0 and N-1
            if strict:  # considers same neighborhood size for high and low dimension
                index_hd = np.arange(0, k_emb)
            high_dim_arr = sorted_nz_cols_D_X[point_id, index_hd]
            low_dim_arr = sorted_nz_cols_D_Y[point_id, 0:k_emb]
            if consider_order:
                low_dim_arr = np.pad(low_dim_arr, pad_width=(0, high_dim_arr.size-low_dim_arr.size), constant_values=-1)
                neighbourhood_intersection = (high_dim_arr == low_dim_arr).sum()
                tps.append(neighbourhood_intersection)
            else:
                neighbourhood_intersection = np.array(np.intersect1d(high_dim_arr, low_dim_arr))
                tps.append(neighbourhood_intersection.size)

        tps = np.array(tps)  # tps for all points for a given k_emb

        precision = (tps/k_emb).mean()  # precision = TP / k_emb
        recall = (tps/k_max).mean()  # recall = TP / h_high

        if to_return == "full":
            nums_true_positives.append(tps)
        precisions.append(precision)
        recalls.append(recall)

    precisions = precisions
    recalls = recalls
    nums_true_positives = nums_true_positives

    if to_return == "aggregated":
        return thresholds, precisions, recalls
    elif to_return == "full":
        return thresholds, precisions, recalls, nums_true_positives
    else:
        raise ("Unknown value for parameter `to_return`: " + str(to_return))
