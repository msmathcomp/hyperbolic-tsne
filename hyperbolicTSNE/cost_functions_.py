""" Implementation of Hyperbolic Kullback-Leibler Divergence

The cost function class has two main methods:
 - obj:     Objective function giving the cost function's value.
 - grad:    Gradient of the cost function.
"""

import ctypes

import numpy as np

from sklearn.utils._openmp_helpers import _openmp_effective_n_threads

from hyperbolicTSNE.hyperbolic_barnes_hut.tsne import gradient

MACHINE_EPSILON = np.finfo(np.double).eps


def check_params(params):
    """Checks params dict includes supported key-values.
    Raises an exception if this is not the case.

    Parameters
    ----------
    params : _type_
        Cost function params in key-value format.
    """
    if "method" not in params or "params" not in params:
        raise ValueError("`other_params` should include a method string `method`, its params `params`.")
    if not isinstance(params["method"], str):
        raise TypeError("`method` of cost function should be a string.")
    if params["params"] is not None and not isinstance(params["params"], dict):
        raise TypeError("`params` should be either None or a dict with the appropriate setup parameters")

    general_params = ["num_threads", "verbose", "degrees_of_freedom", "calc_both", "area_split", "grad_fix"]
    if params["method"] == "exact":
        all_params = general_params + ["skip_num_points"]
    elif params["method"] == "barnes-hut":
        all_params = general_params + ["angle"]
    else:
        raise ValueError("HyperbolicKL method is not a valid one (available methods are `exact` and `barnes-hut`)")

    for p in params["params"]:
        if p not in all_params:
            raise ValueError(f"{p} is not in the param set of the `{params['method']}` version of HyperbolicKL.")
    for p in all_params:
        if p not in params["params"]:
            raise ValueError(
                f"{p} params is necessary for the `{params['method']}` version of HyperbolicKL. Please set a value or "
                f"use a preset."
            )


class HyperbolicKL:
    """
    Hyperbolic Kullback-Leibler Divergence cost function.

    Parameters
    ----------
    n_components : int, optional (default: 2)
        Dimension of the embedded space.
    other_params : dict
        Cost function params in key-value format.
    """
    def __init__(self, *, n_components, other_params=None):
        if other_params is None:
            raise ValueError(
                "No `other_params` specified for HyperbolicKL, please add your params or select one of the presets."
            )
        self.n_components = n_components
        self.params = other_params

        self.results = []

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, other_params):
        try:
            default_params = self._params
        except AttributeError:
            default_params = {}
        if not isinstance(other_params, dict):
            raise Exception("`other_params` should be a dict ... initializing with default params")
        else:
            for k, v in other_params.items():
                default_params[k] = v
            check_params(default_params)
            self._params = default_params

    #######################
    # Parameter templates #
    #######################

    @classmethod
    def exact_tsne(cls):
        """Parameter preset for the exact Hyperbolic tSNE cost function.

        Returns
        -------
        dict
            Cost function params in key-value format.
        """
        return {
            "method": "exact",
            "params": {
                "degrees_of_freedom": 1,
                "skip_num_points": 0,
                "num_threads": _openmp_effective_n_threads(),
                "verbose": False,
                "grad_fix": False,       # tells the cost function calculation to use the correct gradient
            }
        }

    @classmethod
    def bh_tsne(cls, angle=0.5):
        """Parameter preset for the accelerated Hyperbolic tSNE cost function.

        Parameters
        ----------
        angle : float, optional
            Degree of the approximation, by default 0.5

        Returns
        -------
        dict
            Cost function params in key-value format.
        """
        return {
            "method": "barnes-hut",
            "params": {
                "angle": angle, 
                "degrees_of_freedom": 1, 
                "num_threads": _openmp_effective_n_threads(),
                "verbose": False,
                "grad_fix": False,       # tells the cost function calculation to use the correct gradient
            }
        }

    #########################
    # User-facing functions #
    #########################

    def obj(self, Y, *, V):
        """Calculates the Hyperbolic KL Divergence of a given embedding.

        Parameters
        ----------
        Y : ndarray
            Flattened low dimensional embedding of length: n_samples x n_components.
        V : ndarray
            High-dimensional affinity matrix (P matrix in tSNE).

        Returns
        -------
        float
            KL Divergence value.
        """
        n_samples = V.shape[0]
        if self.params["method"] == "exact":
            raise NotImplementedError("Exact obj not implemented. Use obj_grad to get exact cost function value.")
        elif self.params["method"] == "barnes-hut":
            obj, _ = self._obj_bh(Y, V, n_samples)
            return obj

    def grad(self, Y, *, V):
        """Calculates the gradient of the Hyperbolic KL Divergence of 
        a given embedding.

        Parameters
        ----------
        Y : ndarray
            Flattened low dimensional embedding of length: n_samples x n_components.
        V : ndarray
            High-dimensional affinity matrix (P matrix in tSNE).

        Returns
        -------
        ndarray
            Array (n_samples x n_components) with KL Divergence gradient values.
        """
        n_samples = V.shape[0]
        if self.params["method"] == "exact":
            return self._grad_exact(Y, V, n_samples)
        elif self.params["method"] == "barnes-hut":
            _, grad = self._grad_bh(Y, V, n_samples)
            return grad

    def obj_grad(self, Y, *, V):
        """Calculates the Hyperbolic KL Divergence and its gradient 
        of a given embedding.

        Parameters
        ----------
        Y : ndarray
            Flattened low dimensional embedding of length: n_samples x n_components.
        V : ndarray
            High-dimensional affinity matrix (P matrix in tSNE).

        Returns
        -------
        float
            KL Divergence value.
        ndarray
            Array (n_samples x n_components) with KL Divergence gradient values.
        """
        n_samples = V.shape[0]
        if self.params["method"] == "exact":
            obj, grad = self._grad_exact(Y, V, n_samples)
            return obj, grad
        elif self.params["method"] == "barnes-hut":
            obj, grad = self._obj_bh(Y, V, n_samples)
            return obj, grad

    ##########################
    # Main private functions #
    ##########################

    def _obj_exact(self, Y, V, n_samples):
        """Exact computation of the KL Divergence.

        Parameters
        ----------
        Y : ndarray
            Flattened low dimensional embedding of length: n_samples x n_components.
        V : ndarray
            High-dimensional affinity matrix (P matrix in tSNE).
        n_samples : _type_
            Number of samples in the embedding.
        """
        pass

    def _grad_exact(self, Y, V, n_samples, save_timings=True):
        """Exact computation of the KL Divergence gradient.

        Parameters
        ----------
        Y : ndarray
            Flattened low dimensional embedding of length: n_samples x n_components.
        V : ndarray
            High-dimensional affinity matrix (P matrix in tSNE).
        n_samples : _type_
            Number of samples in the embedding.
        save_timings : bool, optional
            If True, saves per iteration times, by default True.

        Returns
        -------
        ndarray
            Array (n_samples x n_components) with KL Divergence gradient values.
        """
        Y = Y.astype(ctypes.c_double, copy=False)
        Y = Y.reshape(n_samples, self.n_components)

        val_V = V.data
        neighbors = V.indices.astype(np.int64, copy=False)
        indptr = V.indptr.astype(np.int64, copy=False)

        grad = np.zeros(Y.shape, dtype=ctypes.c_double)
        timings = np.zeros(4, dtype=ctypes.c_float)
        error = gradient(
            timings,
            val_V, Y, neighbors, indptr, grad,
            0.5,
            self.n_components,
            self.params["params"]["verbose"],
            dof=self.params["params"]["degrees_of_freedom"],
            compute_error=True,
            num_threads=self.params["params"]["num_threads"],
            exact=True,
            grad_fix=self.params["params"]["grad_fix"]
        )

        grad = grad.ravel()
        grad *= 4

        if save_timings:
            self.results.append(timings)

        return error, grad

    def _obj_bh(self, Y, V, n_samples):
        """Approximate computation of the KL Divergence.

        Parameters
        ----------
        Y : ndarray
            Flattened low dimensional embedding of length: n_samples x n_components.
        V : ndarray
            High-dimensional affinity matrix (P matrix in tSNE).
        n_samples : _type_
            Number of samples in the embedding.

        Returns
        -------
        float
            KL Divergence value.
        ndarray
            Array (n_samples x n_components) with KL Divergence gradient values.
        """
        return self._grad_bh(Y, V, n_samples)

    def _grad_bh(self, Y, V, n_samples, save_timings=True):
        """Approximate computation of the KL Divergence gradient.

        Parameters
        ----------
        Y : ndarray
            Flattened low dimensional embedding of length: n_samples x n_components.
        V : ndarray
            High-dimensional affinity matrix (P matrix in tSNE).
        n_samples : _type_
            Number of samples in the embedding.
        save_timings : bool, optional
            If True, saves per iteration times, by default True.

        Returns
        -------
        ndarray
            Array (n_samples x n_components) with KL Divergence gradient values.
        """
        Y = Y.astype(ctypes.c_double, copy=False)
        Y = Y.reshape(n_samples, self.n_components)

        val_V = V.data
        neighbors = V.indices.astype(np.int64, copy=False)
        indptr = V.indptr.astype(np.int64, copy=False)

        grad = np.zeros(Y.shape, dtype=ctypes.c_double)
        timings = np.zeros(4, dtype=ctypes.c_float)
        error = gradient(
            timings,
            val_V, Y, neighbors, indptr, grad,
            self.params["params"]["angle"],
            self.n_components,
            self.params["params"]["verbose"],
            dof=self.params["params"]["degrees_of_freedom"],
            compute_error=True,
            num_threads=self.params["params"]["num_threads"],
            exact=False,
            area_split=self.params["params"]["area_split"],
            grad_fix=self.params["params"]["grad_fix"]
        )

        grad = grad.ravel()
        grad *= 4

        if save_timings:
            self.results.append(timings)

        return error, grad
