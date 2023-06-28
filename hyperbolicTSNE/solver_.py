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

from hyperbolicTSNE.cost_functions_ import HyperbolicKL
from hyperbolicTSNE import tsne_barnes_hut_hyperbolic

MACHINE_EPSILON = np.finfo(np.double).eps


def log_iteration(logging_dict, logging_key, it, y, n_samples, n_components,
                  cf=None, cf_params=None, cf_val=None, grad=None, grad_norm=None, log_arrays=False,
                  log_arrays_ids=None):
    if cf_val is None and grad is None:
        if cf is not None and cf_params is not None:
            cf_val, grad = cf.obj_grad(y, **cf_params)
        else:
            print("No cost function class provided, cf and grad not computed")

    y = y.copy().reshape(n_samples, n_components)
    y_bbox = list(np.concatenate((y.min(axis=0), y.max(axis=0))))
    y_bbox = (
        y_bbox[0], y_bbox[2], y_bbox[1], y_bbox[3], np.sqrt((y_bbox[0] - y_bbox[2]) ** 2 + (y_bbox[1] - y_bbox[3]) ** 2)
    )

    logging_dict[logging_key]["its"].append(it)
    logging_dict[logging_key]["cfs"].append(cf_val)
    logging_dict[logging_key]["grads_norms"].append(grad_norm)
    logging_dict[logging_key]["y_bbox"].append(y_bbox)
    if log_arrays and log_arrays_ids is not None and it in log_arrays_ids:
        # Store embedding and gradient to be returned in memory
        logging_dict[logging_key]["embeddings"].append(y)
        logging_dict[logging_key]["gradients"].append(grad.copy().reshape(n_samples, n_components))

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
        momentum=0.8, learning_rate=200.0, min_gain=0.01, vanilla=False, min_grad_norm=1e-7, error_tol=1e-9, verbose=0,
        rescale=None, n_iter_rescale=np.inf, gradient_mask=np.ones,
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
        logging_dict[logging_key] = {"grads_norms": [], "cfs": [], "cfs_rescaled": [], "its": [], "times": [], "y_bbox": [], "hyperbolic": []}
        log_arrays = logging_dict["log_arrays"]
        if log_arrays:
            log_arrays_ids = logging_dict["log_arrays_ids"]
            if log_arrays_ids is None:
                log_arrays_ids = list(range(start_it, total_its))
            else:
                if type(log_arrays_ids) is list:  # TODO: missing more robust checks
                    log_arrays_ids = [i for i in log_arrays_ids if start_it <= i < total_its]
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
    for i in tqdm(range(i+1, total_its), "Gradient Descent"):
        check_convergence = (i + 1) % n_iter_check == 0
        check_threshold = threshold_cf > 0. or threshold_its > 0
        # only compute the error when needed
        compute_error = check_convergence or check_threshold or i == n_iter - 1

        if compute_error or logging:  # TODO: add different levels of logging to avoid bottlenecks
            error, grad = cf.obj_grad(y, **cf_params)

            if isinstance(cf, HyperbolicKL):
                grad_r = grad.reshape(n_samples, 2)
                Y_r = y.reshape(n_samples, 2)

                metric = (1 - np.linalg.norm(Y_r, axis=1) ** 2) ** 2 / 4
                grad_r = (grad_r * metric[:, np.newaxis]).flatten()

                grad_norm = linalg.norm(grad_r)
            else:
                grad_norm = linalg.norm(grad)
        else:
            grad = cf.grad(y, **cf_params)
            grad_norm = linalg.norm(grad)

        # Perform the actual gradient descent step
        if isinstance(cf, HyperbolicKL):
            Model = cf.params["params"]["hyperbolic_model"]

            if vanilla:
                # y = Model.exp_map(y, -learning_rate * grad * gradient_mask, n_samples)

                res = np.empty((n_samples, 2), dtype=ctypes.c_double)
                tsne_barnes_hut_hyperbolic.exp_map_py(y.reshape(n_samples, 2).astype(ctypes.c_double),
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
                res = np.empty((n_samples, 2), dtype=ctypes.c_double)
                tsne_barnes_hut_hyperbolic.exp_map_py(y.reshape(n_samples, 2).astype(ctypes.c_double),
                                                      (update * gradient_mask)
                                                      .reshape(n_samples, 2)
                                                      .astype(ctypes.c_double),
                                                      res,
                                                      cf.params["params"]["num_threads"])
                y = res.ravel()

            y = Model.constrain(y, n_samples)
            # if Model is LorentzModel:
            #     y = Model.init_proj(y.reshape(n_samples, 3)[:, 1:], n_samples).ravel()
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

        # If a rescale value has been specified, rescale the embedding now to have the bounding box fit the given value.
        if rescale is not None and i % n_iter_rescale == 0:
            y = (y * gradient_mask_inverse) + ((y * gradient_mask) / (np.sqrt((np.max(y[0::2]) - np.min(y[0::2])) ** 2 + (np.max(y[1::2]) - np.min(y[1::2])) ** 2) / rescale))

        # Start:logging
        if logging:
            toc_l = time()
            duration_l = toc_l - tic_l

            if isinstance(cf, HyperbolicKL):
                logging_dict[logging_key]["times"].append(duration_l)
                logging_dict[logging_key]["hyperbolic"].append(cf.results[-1])

            # TODO: For grid run only run every 50 iterations if hyperbolic
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
                current_embedding_size = np.sqrt((np.max(y[0::2]) - np.min(y[0::2])) ** 2 + (np.max(y[1::2]) - np.min(y[1::2])) ** 2)
                # ... if the curret size is smaller than the evaluation size, ...
                if current_embedding_size < threshold_check_size:
                    # ... scale the embedding to the given size and compute/store the corresponding error.
                    y_check_size = y.copy()
                    y_check_size /= current_embedding_size
                    y_check_size *= threshold_check_size
                    error_rescaled = cf.obj(y_check_size, **cf_params)
                    logging_dict[logging_key]["cfs_rescaled"].append(error_rescaled)
                    # If the error is smaller than the given threshold, ...
                    if error_rescaled < threshold_cf:
                        # ... store the current iteration and stop checking for a rescaled value.
                        if logging:
                            logging_dict[logging_key]["threshold_check_size_rescaled_its"] = i - start_it
                            logging_dict[logging_key]["threshold_check_size_rescaled_embedding"] = y_check_size.reshape(n_samples, n_components)
                            logging_dict[logging_key]["threshold_check_size_rescaled_cf"] = error_rescaled
                        threshold_check_size_found = True

            # If a threshold on the number of iterations was given:
            if not threshold_its_found and threshold_its > 0:
                # If given iteration has been reached:
                if i == threshold_its:
                    # Store the current iteration number, embedding, and error
                    if logging:
                        logging_dict[logging_key]["threshold_its_its"] = i - start_it
                        logging_dict[logging_key]["threshold_its_embedding"] = y.copy().reshape(n_samples, n_components)
                        logging_dict[logging_key]["threshold_its_cf"] = error
                    # If a size was given, also store the current embedding scaled to the respective size
                    if threshold_check_size > 0. and logging:
                        current_embedding_size = np.sqrt((np.max(y[0::2]) - np.min(y[0::2])) ** 2 + (np.max(y[1::2]) - np.min(y[1::2])) ** 2)
                        y_its = y.copy()
                        y_its /= current_embedding_size
                        y_its *= threshold_check_size
                        error_rescaled = cf.obj(y_its, **cf_params)
                        logging_dict[logging_key]["threshold_its_rescaled_its"] = i - start_it
                        logging_dict[logging_key]["threshold_its_rescaled_embedding"] = y_its.reshape(n_samples, n_components)
                        logging_dict[logging_key]["threshold_its_rescaled_cf"] = error_rescaled
                    threshold_its_found = True

            # If the current error is smaller or equal to the given threshold, ...
            if not threshold_cf_found and error <= threshold_cf:
                # Store the current iteration number, embedding, and error
                if logging:
                    logging_dict[logging_key]["threshold_cf_its"] = i - start_it
                    logging_dict[logging_key]["threshold_cf_embedding"] = y.copy().reshape(n_samples, n_components)
                    logging_dict[logging_key]["threshold_cf_cf"] = error
                # If a size was given, also store the current embedding scaled to the respective size
                if threshold_check_size > 0. and logging:
                    current_embedding_size = np.sqrt((np.max(y[0::2]) - np.min(y[0::2])) ** 2 + (np.max(y[1::2]) - np.min(y[1::2])) ** 2)
                    y_cf = y.copy()
                    y_cf /= current_embedding_size
                    y_cf *= threshold_check_size
                    error_rescaled = cf.obj(y_cf, **cf_params)
                    logging_dict[logging_key]["threshold_cf_rescaled_its"] = i - start_it
                    logging_dict[logging_key]["threshold_cf_rescaled_embedding"] = y_cf.reshape(n_samples, n_components)
                    logging_dict[logging_key]["threshold_cf_rescaled_cf"] = error_rescaled
                threshold_cf_found = True

            # If for all thresholds: either they are not given or they are satisfied, stop the solver.
            if verbose >= 1 and i % 1 == 0:
                print(
                    "Running iteration " + str(i) + " with "
                    + "Treshold Size: " + str(threshold_check_size) + " (Found: " + str(threshold_check_size_found) + "), "
                    + "Treshold Its: " + str(threshold_its) + " (Found: " + str(threshold_its_found) + "), "
                    + "Threshold Cf: " + str(threshold_cf) + " (Found: " + str(threshold_cf_found) + ")."
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
    # FIXME Is this logging necessary as log iteration is called above already for all iterations?!
    # if logging:
    #     log_iteration(logging_dict, logging_key, i, y, n_samples, n_components,
    #                   cf=cf, cf_params=cf_params,
    #                   log_arrays=log_arrays, log_arrays_ids=log_arrays_ids)

    return y.reshape(n_samples, n_components), error, total_its - start_it


####################################
# Solver: line search-based Newton #
####################################

def ls_backtracking(it, y, p, g, cf_val, cf, cf_args, cf_kwargs, alpha0=1.0, rho=0.8, c=1e-1, return_history=False,
                    tolerance=1e-6, max_num_its=30, time_limit=30):
    """
    Backtracking line search for finding a suitable step length
    in gradient based optimization.

    Parameters
    ----------
    it :
    y :
    p :
    g :
    cf_val :
    cf :
    cf_args :
    cf_kwargs :
    alpha0 : float, optional
        Initial step length.
    rho : float, optional
        Reduction rate pf step length.
    c: float, optional
        --
    tolerance:
    return_history : bool, optional
        If true, in addition to the alpha and the error, the function also returns
        two arrays `hist_alphas` and `hist_errors` with the values that it tried.
    time_limit:

    Returns
    -------
    alpha
        the step length
    e
        the cost function value at y + alpha*p

    """
    alpha = alpha0
    tmp = c * np.dot(p, g)
    e = cf.obj(y + alpha * p, *cf_args, **cf_kwargs)
    if return_history:
        hist_alphas = list()
        hist_alphas.append(alpha)
        hist_errors = list()
        hist_errors.append(e)
    num_its = 0
    start = time()
    while e > cf_val + alpha * tmp:
        alpha = rho * alpha
        e = cf.obj(y + alpha * p, *cf_args, **cf_kwargs)
        if return_history:
            hist_alphas.append(alpha)
            hist_errors.append(e)
        if alpha < tolerance or num_its > max_num_its:
            alpha = 0.0
            break
        num_its += 1
        # if time_limit is not None and time() - start > time_limit:
        #    raise Exception('Problem with line search, it is taking too long to converge (elapsed %.2f secs)' % time_limit)
    if return_history:
        return alpha, e, hist_alphas, hist_errors
    else:
        return alpha, e


def compute_alpha(it, ls_method, y, p, g, prev_cf_val, cf, cf_args, cf_kwargs=None, ls_kwargs=None, verbose=0):
    """
    Entry point to different methods for computing the step length
    via line search (ls).

    Parameters
    __________
    it : int
        Current iteration in the optimization process. This is
        useful if the ls_method is a schedule which determines
        the step length based on the iteration.
    ls_method : str or callable
        Method to use for determining the step length. It can be
        a string, in which case backtracking and bracketing are
        supported. Or a callable which should be a function with the
        signature ls_fn_name(it, y, p, g, cf_val, cf, cf_args, cf_kwargs,
        **kwargs).
    y : array of shape (n_samples * n_components,)
        Points in the embedding flattened in row-major form.
    p : array of shape (n_samples * n_components,)
        Direction vector flattened in row-major form.
    g : array of shape (n_samples * n_components,)
        Gradient vector flattened in row-major form.
    prev_cf_val : float
        Cost function value in the previous iteration (y_{k-1})
    cf : instance of class extending from BaseCostFunction
        Instance of class that extends from BaseCostFunction
        Should have methods for computing the cost function
        value (.obj), the gradient (.grad) and the hessian (.hess).
    cf_args : (optional, default=None)
        Positional arguments for cf's methods (e.g. .obj(*cf_args)).
    cf_kwargs : (optional, default=None)
        Key-word arguments for cf's methods (e.g. .obj(**cf_kwargs)).
    ls_kwargs : (optional, default=None)
        Key-word arguments for the ls (e.g. ls_function(**ls_kwargs)).

    Returns
    -------
    alpha :
        Scaling for current direction p.
    cf_val :
        Cost function value with selected alpha (cf.obj(y + alpha*p, *cf_args, **cf_kwargs))
    """
    alpha = None
    cf_val = None
    available_ls_methods = ["backtracking"]  # , "bracketing", "perfect"]

    if ls_kwargs is None:
        ls_kwargs = {}
        if verbose > 10:
            print("[solver_][compute_alpha] No ls_kwargs provided, using default values")

    if cf_args is None:
        cf_args = []
        if verbose > 10:
            print("[solver_][compute_alpha] No cf_args provided, using default values, using []")

    if cf_kwargs is None:
        raise Exception(
            "[solver_][compute_alpha] No cf_kwargs provided, cost functions usually need at least one (V mat).")

    if type(ls_method) == str:
        if ls_method in available_ls_methods:
            try:
                if ls_method == 'backtracking':
                    alpha, cf_val = ls_backtracking(it, y, p, g, prev_cf_val, cf, cf_args, cf_kwargs, **ls_kwargs)
                # elif ls_method == 'bracketing':
                #    alpha, cf = bracketing_ls(y, V, old_cf, p, g, alpha0, cf_args, cf_kwargs, **ls_kwargs)
                # elif ls_method == 'perfect':
                #    alpha, cf = perfect_ls(y, V, old_cf, p, g, alpha0, cf_args, cf_kwargs, **ls_kwargs)
                if alpha is None:
                    raise Exception('[solver_][compute_alpha] Line search received a zero value')
            except Exception as e:
                raise Exception(e)
        else:
            raise Exception("[solver_][compute_alpha] Specified ls_method is not a valid one. "
                            "Supported methods are: {}".format(", ".join(available_ls_methods)))
    else:
        if callable(ls_method):
            try:
                alpha = ls_method(it, y, p, g, prev_cf_val, cf, cf_args, cf_kwargs, **ls_kwargs)
                cf_val = None
            except Exception as e:
                raise Exception(e)
        else:
            raise Exception("[solver_][compute_alpha] ls_method should either by string or a callable")

    return alpha, cf_val


def compute_direction(R, grad, n, d):
    """
    Method for computing the update.
    TODO: in the future it could accomodate even the delta-bar-delta scheme
    TODO: also other solvers like conjugate gradients

    Parameters
    ----------
    R : sksparse.cholmod.Factor
        Cholesky factor of Hessian.
    grad : array of shape (n_samples * n_components,)
        Gradient vector flattened in row-major form.
    n : int
        Number of points in the embedding.
    d : int
        Number of dimensions in the embedding.

    Returns
    -------
    pk
      array of the same shape as grad that solves B pk = grad
    """
    pk = -R(grad.astype(np.float64))

    return pk


def ls_newton_solver(
        y0, cf, cf_params, *,  # main parameters
        alpha0=1, ls_method="backtracking", ls_kwargs=None, use_last_alpha=False,  # step length method parameters
        start_it=0, n_iter=100, n_iter_check=50, n_iter_without_progress=300,  # optimization loop parameters
        min_grad_norm=1e-7, error_tol=1e-9,
        verbose=0, logging_dict=None, logging_key=None,
):
    """
    Batch Newton method (y' = y - alpha * H \dot grad)

    Parameters
    ----------
    y0 : array-like, shape (n_params, n_components)
        Initial parameter vector.
    cf : function or callable
        Class that implements the methods obj, grad, hess. These methods
        should return a float, vector and matrix, respectively.
    cf_params:
        Additional params that should be passed to the obj, grad, hess functions of
        cf. For example, the high-dimensional matrix V.
    alpha0 : float, optional (default: 1.0)
        Upper bound to start the line search from. This is the same as learning rate.
        Therefore if the hessian matrix is the identity, a large alpha_0 can be used.
        For quasi-Newton methods (hessian different from the identity), it is
        recommended to set this to 1 to avoid diverging from the solution.
    ls_method : str, optional (default: backtracking)
        Line search method to use. By default uses backtracking, which is conceptually
        simple and often yields good results. More complex methods based on interpolation
        will be added in the future, but this are more computationally expensive.
    ls_kwargs: dict, optional (default: None)
        Key-word arguments for ls's functions.
    use_last_alpha: bool, optional (default: false)
        Starts the line search of the current iteration from the value reached in the
        previous one.
    start_it : int
        Iteration to start counting from. This helps keeping a single count between
        multiple solver runs. This parameter is set to 0 by default.
    n_iter : int
        Maximum number of gradient descent iterations.
    n_iter_check : int
        Number of iterations before evaluating the global error. If the error
        is sufficiently low, we abort the optimization.
    n_iter_without_progress : int, optional (default: 300)
        Maximum number of iterations without progress before we abort the
        optimization.
    min_grad_norm : float, optional (default: 1e-7)
        If the gradient norm is below this threshold, the optimization will
        be aborted.
    error_tol : float, optional (default: 1e-9)
        If the difference between the previous lowest error and the current one (defined by
        the n_iter_check param) is below this threshold, the optimization will
        be aborted.
    verbose : int, optional (default: 0)
        Verbosity level.
    logging_dict :
    logging_key :
    Returns
    -------
    p : array, shape (n_params,)
        Optimum parameters.
    error : float
        Optimum.
    i : int
        Last iteration.
    """
    n_samples, n_components = y0.shape
    y = y0.copy().ravel()
    error = np.finfo(np.float).max
    best_error = np.finfo(np.float).max
    best_iter = i = 0
    total_its = start_it + n_iter

    if not ls_kwargs:
        ls_kwargs = {}
        ls_kwargs.update({"alpha0": alpha0})

    B = cf.hess(y, **cf_params)
    try:
        R = None  # cholesky(B)
    except Exception as e:
        raise Exception(e)

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
        logging_key = "solver_ls_newton_solver" + logging_key
        logging_dict[logging_key] = {"grads_norms": [], "cfs": [], "its": [], "times": [], "y_bbox": []}
        log_arrays = logging_dict["log_arrays"]
        if log_arrays:
            log_arrays_ids = logging_dict["log_arrays_ids"]
            if log_arrays_ids is None:
                log_arrays_ids = list(range(start_it, total_its))
            else:
                if type(log_arrays_ids) is list:  # TODO: missing more robust checks
                    log_arrays_ids = [i for i in log_arrays_ids if start_it <= i < total_its]
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
    for i in range(start_it, total_its):
        print("Iteration: {}".format(i))
        check_convergence = (i + 1) % n_iter_check == 0
        # only compute the error when needed
        compute_error = check_convergence or i == n_iter - 1

        if compute_error or logging:  # TODO: add different levels of logging to avoid bottlenecks
            error, grad = cf.obj_grad(y, **cf_params)
            grad_norm = linalg.norm(grad)
            # Start:logging
            if logging:
                log_iteration(logging_dict, logging_key, i, y, n_samples, n_components,
                              cf_val=error, grad=grad, grad_norm=grad_norm,
                              log_arrays=log_arrays, log_arrays_ids=log_arrays_ids)
            # End:logging
        else:
            grad = cf.grad(y, **cf_params)
            grad_norm = linalg.norm(grad)

        update = compute_direction(R, grad, n_samples, n_components)  # get product of gradient times hessian
        alpha, _ = compute_alpha(i, ls_method, y, update, grad, error, cf, None, cf_params,
                                 ls_kwargs)  # compute based on line search method
        if use_last_alpha:
            ls_kwargs["alpha0"] = alpha
        print(alpha)
        # FIXME: what happens if error is none?
        old_y = y.copy()
        y += alpha * update

        # Start:logging
        if logging:
            toc_l = time()
            duration_l = toc_l - tic_l
            logging_dict[logging_key]["times"].append(duration_l)
            tic_l = toc_l
        # End:logging

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
    if logging:
        log_iteration(logging_dict, logging_key, i, y, n_samples, n_components,
                      cf=cf, cf_params=cf_params,
                      log_arrays=log_arrays, log_arrays_ids=log_arrays_ids)
    return y.reshape(n_samples, n_components), error, i - start_it
