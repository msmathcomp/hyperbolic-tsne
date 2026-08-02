""" Collection of Solvers for high dimensional data embedding optimization

This file collects several methods, each representing a solver to be used in the context of high dimensional
data embedding optimization.

Currently available solvers include:
 - Gradient Descent
"""
import ctypes
from time import time
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import linalg
from tqdm import tqdm

from .cost_functions_ import HyperbolicKL
from .hyperbolic_barnes_hut import tsne

MACHINE_EPSILON = np.finfo(np.double).eps


def log_iteration(logging_dict, logging_key, it, y, n_samples, n_components,
                  cf=None, cf_params=None, cf_val=None, grad=None, grad_norm=None, log_arrays=False,
                  log_arrays_ids=None):
    """
    Log information about an optimization iteration.

    Parameters
    ----------
    logging_dict : dict 
        A dictionary containing logging information for different keys.
    logging_key : str 
        The key to identify the specific logging information within the logging_dict.
    it: int 
        The iteration number.
    y : numpy.ndarray 
        The embedding or solution obtained at the current iteration.
    n_samples : int 
        The number of samples or data points.
    n_components : int 
        The number of components or features in the embedding.
    cf : object 
        Cost function class with obj_grad method to compute the cost function value and gradient.
    cf_params : dict 
        Parameters to be passed to the obj_grad method of the cost function class.
    cf_val : float 
        The precomputed cost function value. If None, it will be computed using cf.
    grad : numpy.ndarray 
        The precomputed gradient. If None, it will be computed using cf.
    grad_norm : float 
        The norm of the gradient.
    log_arrays : bool 
        Whether to log embedding and gradient arrays.
    log_arrays_id : list 
        Iteration numbers to log arrays if log_arrays is True.

    Returns
    -------
    None
    """
    if cf_val is None and grad is None:
        if cf is not None and cf_params is not None:
            cf_val, grad = cf.obj_grad(y, **cf_params)
        else:
            print("No cost function class provided, cf and grad not computed")

    y = y.copy().reshape(n_samples, n_components)
    y_bbox = list(np.concatenate((y.min(axis=0), y.max(axis=0))))
    y_bbox = (
        y_bbox[0], y_bbox[2], y_bbox[1], y_bbox[3], np.sqrt(
            (y_bbox[0] - y_bbox[2]) ** 2 + (y_bbox[1] - y_bbox[3]) ** 2)
    )

    logging_dict[logging_key]["its"].append(it)
    logging_dict[logging_key]["cfs"].append(cf_val)
    logging_dict[logging_key]["grads_norms"].append(grad_norm)
    logging_dict[logging_key]["y_bbox"].append(y_bbox)
    if log_arrays and log_arrays_ids is not None and it in log_arrays_ids:
        # Store embedding and gradient to be returned in memory
        logging_dict[logging_key]["embeddings"].append(y)
        logging_dict[logging_key]["gradients"].append(
            grad.copy().reshape(n_samples, n_components))

    # Store the embedding as CSV file at the given location
    log_path = logging_dict.get("log_path", None)
    if log_path is not None:
        Path(log_path + logging_key).mkdir(parents=True, exist_ok=True)
        pd.DataFrame(y).to_csv(log_path + logging_key + "/" + str(it) + ", " + str(cf_val) + ".csv", header=False,
                               index=False)


##########################################
# Solver: gradient descent with momentum #
##########################################

def gradient_descent(
        y0, cf, cf_params, *, start_it=0, n_iter=100, n_iter_check=np.inf, n_iter_without_progress=300,
        threshold_cf=0., threshold_its=-1, threshold_check_size=-1.,
        momentum=0.8, learning_rate=200.0, min_gain=0.01, vanilla=False, min_grad_norm=1e-7, error_tol=1e-9, size_tol=None,
        verbose=0, rescale=None, n_iter_rescale=np.inf, gradient_mask=np.ones, grad_scale_fix=True,
        logging_dict=None, logging_key=None,
):
    """Batch gradient descent with momentum and individual gains.
    Parameters
    ----------
    y0 : array-like, shape (n_params, n_components)
        Initial parameter vector.
    cf : function or callable
        Class that implements the method's obj, grad, hess. These methods
        should return a float, vector and matrix, respectively.
    cf_params:
        Additional params that should be passed to the obj, grad, hess functions of
        cf. For example, the high-dimensional matrix V.
    start_it : int, optional (default: 0)
        Iteration to start counting from. This helps keeping a single count between
        multiple solver runs. This parameter is set to 0 by default.
    n_iter : int, optional (default: 100)
        Maximum number of gradient descent iterations.
    n_iter_check : int, optional (default: inf)
        Number of iterations before evaluating the global error. If the error
        is sufficiently low, we abort the optimization.
    n_iter_without_progress : int, optional (default: 300)
        Maximum number of iterations without progress before we abort the
        optimization.
    threshold_cf : float, optional (default: 0.)
        A positive number, if the cost function goes below this, the solver stops.
    threshold_its : int, optional (default: -1)
        A positive number, if the solver performs this number of iterations, it stops.
    threshold_check_size : float, optinoal (default: -1)
        A positive number, providing the size to which to scale the current embedding when checking its error.
    momentum : float, within (0.0, 1.0), optional (default: 0.8)
        The momentum generates a weight for previous gradients that decays
        exponentially.
    learning_rate : float, optional (default: 200.0)
        The learning rate for t-SNE is usually in the range [10.0, 1000.0]. If
        the learning rate is too high, the data may look like a 'ball' with any
        point approximately equidistant from its nearest neighbours. If the
        learning rate is too low, most points may look compressed in a dense
        cloud with few outliers.
    min_gain : float, optional (default: 0.01)
        Minimum individual gain for each parameter.
    vanilla: bool, optional (default: True)
        If True, then vanilla gradient descent with a constant learning rate is used.
        Vanilla gradient descent doesn't use fancy tricks like momentum to accelerate convergence.
    min_grad_norm : float, optional (default: 1e-7)
        If the gradient norm is below this threshold, the optimization will
        be aborted.
    error_tol : float, optional (default: 1e-9)
        If the difference between the previous lowest error and the current one (defined by
        the n_iter_check param) is below this threshold, the optimization will
        be aborted.
    size_tol : float, optional (default: None)
        If the distance from a point to the center surpases the size_tol, then the optimization
        will be aborted.
    verbose : int, optional (default: 0)
        Verbosity level.
    rescale: float, optional (default: None)
        If a positive float "rescale" is given, the embedding is rescaled to fit a bounding box with a diagonal length
        of "rescale". No rescaling is performed in all other cases.
    n_iter_rescale: int, optional (default: np.inf)
        If a number n is given, then all n iterations (including the 0th one), the embedding is rescaled according to
        the scale given in the `rescale` parameter.
    gradient_mask : numpy vector, optional (default: np.ones)
        A vector of 0/1 entries. Apply the gradient update only where the mask is 1.
    logging_dict : dict, optional (default: None)
        A dictionary to store results obtained by the solver.
    logging_key : String, optional (default: None)
        A prefix to prepend to the results in the logging_dict.
    Returns
    -------
    p : array, shape (n_params,)
        Optimum parameters.
    error : float
        Optimum.
    i : int
        Last iteration.
    """
    print("Running Gradient Descent, Verbosity: " + str(verbose))

    n_samples, n_components = y0.shape
    y = y0.copy().ravel()
    if callable(gradient_mask):
        gradient_mask = np.ones(y0.shape).ravel()
    else:
        gradient_mask = np.column_stack((gradient_mask, gradient_mask)).ravel()
    gradient_mask_inverse = np.ones(gradient_mask.shape) - gradient_mask
    update = None
    gains = None
    if not vanilla:
        update = np.zeros_like(y)
        gains = np.ones_like(y)
    error = np.finfo(float).max
    best_error = np.finfo(float).max
    best_iter = 0
    total_its = start_it + n_iter
    # Check whether the given rescale value is a float and larger than 0.
    if not (isinstance(rescale, float) or isinstance(rescale, int)) or rescale <= 0.:
        rescale = None

    threshold_check_size_found = False
    threshold_cf_found = False
    threshold_its_found = False

    # Start: logging
    logging = False
    log_arrays = None
    log_arrays_ids = None
    tic_l = None
    if logging_dict is not None:
        if logging_key is None:
            logging_key = ""
        else:
            logging_key = "_" + logging_key
        logging_key = "solver_gradient_descent" + logging_key
        logging_dict[logging_key] = {"grads_norms": [], "cfs": [], "cfs_rescaled": [
        ], "its": [], "times": [], "y_bbox": [], "hyperbolic": []}
        log_arrays = logging_dict["log_arrays"]
        if log_arrays:
            log_arrays_ids = logging_dict["log_arrays_ids"]
            if log_arrays_ids is None:
                log_arrays_ids = list(range(start_it, total_its))
            else:
                if type(log_arrays_ids) is list:  # TODO: missing more robust checks
                    log_arrays_ids = [
                        i for i in log_arrays_ids if start_it <= i < total_its]
            logging_dict[logging_key]["log_arrays_ids"] = log_arrays_ids
            logging_dict[logging_key]["embeddings"] = []
            logging_dict[logging_key]["gradients"] = []
        else:
            log_arrays_ids = None
        print("[gradient_descent] Warning: because of logging, the cf will be computed at every iteration")
        logging = True  # when logging True, data from every iteration will be gathered, but not used to assess convergence criteria
        tic_l = time()
    # End: logging

    tic = time()
    i = start_it-1
    for i in (pbar := tqdm(range(i+1, total_its), "Gradient Descent")):
        check_convergence = (i + 1) % n_iter_check == 0
        check_threshold = threshold_cf > 0. or threshold_its > 0
        # only compute the error when needed
        compute_error = check_convergence or check_threshold or i == n_iter - 1

        if compute_error or logging:  # TODO: add different levels of logging to avoid bottlenecks
            error, grad = cf.obj_grad(y, **cf_params)

            if isinstance(cf, HyperbolicKL):
                # New Fix: Inverse metric tensor factor that multiplies the derivative of the cost function
                # leading to a correct gradient update step
                if grad_scale_fix:
                    grad = ((1. - np.linalg.norm(y.reshape(n_samples, 2), axis=1)
                            ** 2) ** 2)[:, np.newaxis] * grad.reshape(n_samples, 2) / 4
                    grad = grad.flatten()

                grad_norm = linalg.norm(grad)
            else:
                grad_norm = linalg.norm(grad)
        else:
            grad = cf.grad(y, **cf_params)
            grad_norm = linalg.norm(grad)

        # Perform the actual gradient descent step
        if isinstance(cf, HyperbolicKL):
            if vanilla:
                # y = Model.exp_map(y, -learning_rate * grad * gradient_mask, n_samples)

                res = np.empty((n_samples, 2), dtype=ctypes.c_double)
                tsne.exp_map(y.reshape(n_samples, 2).astype(ctypes.c_double),
                             (-learning_rate * grad * gradient_mask)
                             .reshape(n_samples, 2)
                             .astype(ctypes.c_double),
                             res,
                             cf.params["params"]["num_threads"])
                y = res.ravel()
            else:
                inc = update * grad < 0.0
                dec = np.invert(inc)
                gains[inc] += 0.2
                gains[dec] *= 0.8
                np.clip(gains, min_gain, np.inf, out=gains)
                grad *= gains
                update = momentum * update - learning_rate * grad
                # y = Model.exp_map(y, update * gradient_mask, n_samples)
                res_exp = np.empty((n_samples, 2), dtype=ctypes.c_double)
                tsne.exp_map(y.reshape(n_samples, 2).astype(ctypes.c_double),
                             (update * gradient_mask)
                             .reshape(n_samples, 2)
                             .astype(ctypes.c_double),
                             res_exp,
                             cf.params["params"]["num_threads"])

                res_log = np.empty((n_samples, 2), dtype=ctypes.c_double)
                tsne.log_map(res_exp,
                             y.reshape(n_samples, 2).astype(ctypes.c_double),
                             res_log,
                             cf.params["params"]["num_threads"])
                y = res_exp.ravel()

                update = res_log.ravel() * -1

            res_constrain = np.empty((n_samples, 2), dtype=ctypes.c_double)
            tsne.constrain(y.reshape(n_samples, 2).astype(ctypes.c_double),
                           res_constrain,
                           cf.params["params"]["num_threads"])
            y = res_constrain.ravel()

        elif vanilla:
            y = y - learning_rate * grad * gradient_mask
        else:
            inc = update * grad < 0.0
            dec = np.invert(inc)
            gains[inc] += 0.2
            gains[dec] *= 0.8
            np.clip(gains, min_gain, np.inf, out=gains)
            grad *= gains
            update = momentum * update - learning_rate * grad
            y += update * gradient_mask

        pbar.set_description(
            f"Gradient Descent error: {error:.5f} grad_norm: {grad_norm:.5e}")

        # If a rescale value has been specified, rescale the embedding now to have the bounding box fit the given value.
        if rescale is not None and i % n_iter_rescale == 0:
            y = (y * gradient_mask_inverse) + ((y * gradient_mask) / (np.sqrt((np.max(
                y[0::2]) - np.min(y[0::2])) ** 2 + (np.max(y[1::2]) - np.min(y[1::2])) ** 2) / rescale))

        # Start:logging
        if logging:
            toc_l = time()
            duration_l = toc_l - tic_l

            # if isinstance(cf, HyperbolicKL):
            #     logging_dict[logging_key]["times"].append(duration_l)
            #     logging_dict[logging_key]["hyperbolic"].append(cf.results[-1])
            
            # if not isinstance(cf, HyperbolicKL) or i % 50 == 0 or i == total_its - 1:
            log_iteration(logging_dict, logging_key, i, y, n_samples, n_components,
                          cf_val=error, grad=grad, grad_norm=grad_norm,
                          log_arrays=log_arrays, log_arrays_ids=log_arrays_ids)
            tic_l = toc_l
        # End:logging

        # See whether the solver should stop because the given error threshold has been reached
        if check_threshold:

            # If a size for evaluation was given ...
            if not threshold_check_size_found and threshold_check_size > 0.:
                # ... compute the current size and ...
                current_embedding_size = np.sqrt(
                    (np.max(y[0::2]) - np.min(y[0::2])) ** 2 + (np.max(y[1::2]) - np.min(y[1::2])) ** 2)
                # ... if the curret size is smaller than the evaluation size, ...
                if current_embedding_size < threshold_check_size:
                    # ... scale the embedding to the given size and compute/store the corresponding error.
                    y_check_size = y.copy()
                    y_check_size /= current_embedding_size
                    y_check_size *= threshold_check_size
                    error_rescaled = cf.obj(y_check_size, **cf_params)
                    logging_dict[logging_key]["cfs_rescaled"].append(
                        error_rescaled)
                    # If the error is smaller than the given threshold, ...
                    if error_rescaled < threshold_cf:
                        # ... store the current iteration and stop checking for a rescaled value.
                        if logging:
                            logging_dict[logging_key]["threshold_check_size_rescaled_its"] = i - start_it
                            logging_dict[logging_key]["threshold_check_size_rescaled_embedding"] = y_check_size.reshape(
                                n_samples, n_components)
                            logging_dict[logging_key]["threshold_check_size_rescaled_cf"] = error_rescaled
                        threshold_check_size_found = True

            # If a threshold on the number of iterations was given:
            if not threshold_its_found and threshold_its > 0:
                # If given iteration has been reached:
                if i == threshold_its:
                    # Store the current iteration number, embedding, and error
                    if logging:
                        logging_dict[logging_key]["threshold_its_its"] = i - start_it
                        logging_dict[logging_key]["threshold_its_embedding"] = y.copy().reshape(
                            n_samples, n_components)
                        logging_dict[logging_key]["threshold_its_cf"] = error
                    # If a size was given, also store the current embedding scaled to the respective size
                    if threshold_check_size > 0. and logging:
                        current_embedding_size = np.sqrt(
                            (np.max(y[0::2]) - np.min(y[0::2])) ** 2 + (np.max(y[1::2]) - np.min(y[1::2])) ** 2)
                        y_its = y.copy()
                        y_its /= current_embedding_size
                        y_its *= threshold_check_size
                        error_rescaled = cf.obj(y_its, **cf_params)
                        logging_dict[logging_key]["threshold_its_rescaled_its"] = i - start_it
                        logging_dict[logging_key]["threshold_its_rescaled_embedding"] = y_its.reshape(
                            n_samples, n_components)
                        logging_dict[logging_key]["threshold_its_rescaled_cf"] = error_rescaled
                    threshold_its_found = True

            # If the current error is smaller or equal to the given threshold, ...
            if not threshold_cf_found and error <= threshold_cf:
                # Store the current iteration number, embedding, and error
                if logging:
                    logging_dict[logging_key]["threshold_cf_its"] = i - start_it
                    logging_dict[logging_key]["threshold_cf_embedding"] = y.copy().reshape(
                        n_samples, n_components)
                    logging_dict[logging_key]["threshold_cf_cf"] = error
                # If a size was given, also store the current embedding scaled to the respective size
                if threshold_check_size > 0. and logging:
                    current_embedding_size = np.sqrt(
                        (np.max(y[0::2]) - np.min(y[0::2])) ** 2 + (np.max(y[1::2]) - np.min(y[1::2])) ** 2)
                    y_cf = y.copy()
                    y_cf /= current_embedding_size
                    y_cf *= threshold_check_size
                    error_rescaled = cf.obj(y_cf, **cf_params)
                    logging_dict[logging_key]["threshold_cf_rescaled_its"] = i - start_it
                    logging_dict[logging_key]["threshold_cf_rescaled_embedding"] = y_cf.reshape(
                        n_samples, n_components)
                    logging_dict[logging_key]["threshold_cf_rescaled_cf"] = error_rescaled
                threshold_cf_found = True

            # If for all thresholds: either they are not given or they are satisfied, stop the solver.
            if verbose >= 1 and i % 1 == 0:
                print(
                    "Running iteration " + str(i) + " with "
                    + "Treshold Size: " +
                    str(threshold_check_size) + " (Found: " +
                    str(threshold_check_size_found) + "), "
                    + "Treshold Its: " +
                    str(threshold_its) + " (Found: " +
                    str(threshold_its_found) + "), "
                    + "Threshold Cf: " +
                    str(threshold_cf) + " (Found: " +
                    str(threshold_cf_found) + ")."
                )
            if (threshold_check_size <= 0. or threshold_check_size_found) and (threshold_its <= 0 or threshold_its_found) and (threshold_cf <= 0. or threshold_cf_found):
                return y.reshape(n_samples, n_components), error, i - start_it

        if check_convergence:
            toc = time()
            duration = toc - tic
            tic = toc

            if verbose >= 2:
                print("[t-SNE] Iteration %d: error = %.7f,"
                      " gradient norm = %.7f"
                      " (%s iterations in %0.3fs)"
                      % (i + 1, error, grad_norm, n_iter_check, duration))

            if error < best_error:
                if best_error - error <= error_tol:
                    if verbose >= 2:
                        print("[t-SNE] Error at iteration %d: did not make significant progress "
                              "during the last %d episodes. Error function change was %d. Finished."
                              % (i + 1, i - best_iter, best_error - error))
                    print("1")
                    break
                best_error = error
                best_iter = i
            elif i - best_iter > n_iter_without_progress:
                if verbose >= 2:
                    print("[t-SNE] Iteration %d: did not make any progress "
                          "during the last %d episodes. Finished."
                          % (i + 1, n_iter_without_progress))
                print("2")
                break
            if grad_norm <= min_grad_norm:
                if verbose >= 2:
                    print("[t-SNE] Iteration %d: gradient norm %f. Finished."
                          % (i + 1, grad_norm))
                print("3")
                break

            emb_point_dists = np.linalg.norm(
                y.reshape((n_samples, -1)), axis=1).max()
            if size_tol is not None and emb_point_dists > size_tol:
                if verbose >= 2:
                    print("[t-SNE] Iteration %d: max size %f. Finished." %
                          (i + 1, emb_point_dists))
                print("4")
                break

    return y.reshape(n_samples, n_components), error, total_its - start_it
