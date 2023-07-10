""" Collection of Cost Functions for high dimensional data embedding optimization

This file collects several classes, each representing a cost function to be used in the context of high dimensional
data embedding optimization. Each class is derived from the BaseCostFunction class and therefore has to implement the
following methods:
 - obj:     Objective function giving the cost function's value.
 - grad:    Gradient of the cost function.
 - hess:    Hessian of the cost function.

Currently available cost functions include:
 - Kullback-Leibler Divergence
"""

import ctypes
from abc import ABC, abstractmethod

import numpy as np
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

from hyperbolicTSNE import tsne_barnes_hut, tsne_barnes_hut_hyperbolic
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads

MACHINE_EPSILON = np.finfo(np.double).eps


class BaseCostFunction(ABC):
    """ Base class for cost functions.

    Cost functions should have three methods, obj, grad, hess, which return a scalar, a (1 x nd) vector,
    and an (nd x nd) matrix respectively, where n is the number of points to be embedded and d is the low-dimensional
    embedding dimension.
    """

    def __init__(self, *, n_components, other_params=None):
        if other_params is None:
            other_params = {}
        self.n_components = n_components
        self.params = other_params

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, other_params):
        try:
            default_params = self._params
        except AttributeError:
            default_params = {}
        if type(other_params) is not dict:
            raise Exception("`other_params` should be a dict ... initializing with default params")
        else:
            for k, v in other_params.items():
                default_params[k] = v
            BaseCostFunction.check_params(default_params)
            self._params = default_params

    @classmethod
    def check_params(cls, params):
        if "method" not in params or "params" not in params:
            raise Exception(
                "`other_params` should include a method string `method`, its params `params`.")
        if type(params["method"]) is not str: # Each concrete implementation checks for specific methods.
            raise Exception("`method` of cost function should be a string.")
        if params["params"] is not None and type(params["params"]) is not dict: # Each concrete implementation checks for specific setup params.
            raise Exception("`params` should be either None or a dict with the appropriate setup parameters")

    @abstractmethod
    def obj(self, Y, *, V):
        pass

    @abstractmethod
    def grad(self, Y, *, V):
        pass

    @abstractmethod
    def obj_grad(self, Y, *, V):
        pass


class KLDivergence(BaseCostFunction):
    # TODO Class and method documentation

    def __init__(self, *, n_components, other_params=None):
        if other_params is None:
            raise Exception("No `other_params` specified for KLDivergence, please add your params or select one of the presets.")
        super(KLDivergence, self).__init__(n_components=n_components, other_params=other_params)

    @property
    def params(self):
        return super(KLDivergence, self).params

    @params.setter
    def params(self,  other_params):
        # Nice trick for using super setter: https://gist.github.com/Susensio/979259559e2bebcd0273f1a95d7c1e79
        super(KLDivergence, type(self)).params.fset(self, other_params)
        KLDivergence.check_params(self.params)

    @classmethod
    def check_params(cls, params):
        general_params = ["num_threads", "verbose", "degrees_of_freedom"]
        if params["method"] == "exact":
            all_params = general_params + ["skip_num_points"]
        elif params["method"] == "barnes-hut":
            all_params = general_params + ["angle"]
        else:
            raise Exception("KLDivergence method is not a valid one (available methods are `exact` and `barnes-hut`)")

        for p in params["params"]:
            if p not in all_params:
                raise Exception("%s is not in the param set of the `%s` version of KLDivergence." % (p, params["method"]))
        for p in all_params:
            if p not in params["params"]:
                raise Exception(
                    "%s params is necessary for the `%s` version of KLDivergence. Please set a value or use a preset." % (p, params["method"]))
        # TODO: add checks for specific values of parameters

    #######################
    # Parameter templates #
    #######################

    @classmethod
    def exact_tsne(cls):
        return {
            "method": "exact",
            "params": {"degrees_of_freedom": 1, "skip_num_points": 0, "num_threads": 1, "verbose": False}
        }

    @classmethod
    def bh_tsne(cls, angle=0.5):
        return {
            "method": "barnes-hut",
            "params": {"angle": angle, "degrees_of_freedom": 1, "num_threads": _openmp_effective_n_threads(), "verbose": False}
        }

    #########################
    # User-facing functions #
    #########################

    def obj(self, Y, *, V, forces="total", per_point=False):
        n_samples = V.shape[0]
        if self.params["method"] == "exact":
            return self._obj_exact(Y, V, n_samples, forces, per_point)
        elif self.params["method"] == "barnes-hut":
            if forces != "total" or per_point == True:
                raise Exception("If computing KLDivergence using Barnes-Hut, individualized features are not available.")
            obj, _ = self._obj_bh(Y, V, n_samples)
            return obj

    def grad(self, Y, *, V, forces="total", per_point=False):
        n_samples = V.shape[0]
        if self.params["method"] == "exact":
            return self._grad_exact(Y, V, n_samples, forces, per_point)
        elif self.params["method"] == "barnes-hut":
            if forces != "total" or per_point == True:
                raise Exception("If computing KLDivergence using Barnes-Hut, individualized features are not available.")
            _, grad = self._grad_bh(Y, V, n_samples)
            return grad

    def obj_grad(self, Y, *, V, forces="total", per_point=False):
        n_samples = V.shape[0]
        if self.params["method"] == "exact":
            print("Warning: the exact version of KL divergence computes obj and grad separately.")
            obj = self._obj_exact(Y, V, n_samples, forces, per_point)
            grad = self._grad_exact(Y, V, n_samples, forces, per_point)
            return obj, grad
        elif self.params["method"] == "barnes-hut":
            if forces != "total" or per_point == True:
                raise Exception("If computing KLDivergence using Barnes-Hut, individualized features are not available.")
            obj, grad = self._obj_bh(Y, V, n_samples)
            return obj, grad

    ##########################
    # Main private functions #
    ##########################

    def _grad_exact(self, Y, V, n_samples, forces="total", per_point=False):
        # TODO: numbify

        Y = Y.reshape(n_samples, self.n_components)
        V = squareform(np.array(V.todense()))  # FIXME: exact method assume dense V which is inefficient

        # W is a heavy-tailed distribution: Student's t-distribution
        dist = pdist(Y, "sqeuclidean")
        dist /= self.params["params"]["degrees_of_freedom"]
        dist += 1.
        dist **= (self.params["params"]["degrees_of_freedom"] + 1.0) / -2.0
        W = np.maximum(dist / (2.0 * np.sum(dist)), MACHINE_EPSILON)

        # Gradient: dC/dY
        # pdist always returns double precision distances. Thus we need to take
        grad = np.ndarray((n_samples, self.n_components), dtype=Y.dtype)
        VWd = squareform((V - W) * dist)
        for i in range(self.params["params"]["skip_num_points"], n_samples):
            grad[i] = np.dot(np.ravel(VWd[i], order='K'),
                             Y[i] - Y)
        grad = grad.ravel()
        c = 2.0 * (self.params["params"]["degrees_of_freedom"] + 1.0) / self.params["params"]["degrees_of_freedom"]
        grad *= c

        return grad

    def _obj_bh(self, Y, V, n_samples):

        Y = Y.astype(np.float32, copy=False)
        Y = Y.reshape(n_samples, self.n_components)

        val_V = V.data.astype(np.float32, copy=False)
        neighbors = V.indices.astype(np.int64, copy=False)
        indptr = V.indptr.astype(np.int64, copy=False)

        grad = np.zeros(Y.shape, dtype=np.float32)
        error = tsne_barnes_hut.gradient(
            val_V, Y, neighbors, indptr, grad,
            self.params["params"]["angle"],
            self.n_components,
            self.params["params"]["verbose"],
            dof=self.params["params"]["degrees_of_freedom"],
            compute_error=True,
            num_threads=self.params["params"]["num_threads"]
        )
        c = 2.0 * (self.params["params"]["degrees_of_freedom"] + 1.0) / self.params["params"]["degrees_of_freedom"]
        grad = grad.ravel()
        grad *= c

        return error, grad

    def _grad_bh(self, Y, V, n_samples):

        Y = Y.astype(np.float32, copy=False)
        Y = Y.reshape(n_samples, self.n_components)

        val_V = V.data.astype(np.float32, copy=False)
        neighbors = V.indices.astype(np.int64, copy=False)
        indptr = V.indptr.astype(np.int64, copy=False)

        grad = np.zeros(Y.shape, dtype=np.float32)
        error = tsne_barnes_hut.gradient(
            val_V, Y, neighbors, indptr, grad,
            self.params["params"]["angle"],
            self.n_components,
            self.params["params"]["verbose"],
            dof=self.params["params"]["degrees_of_freedom"],
            compute_error=True,
            num_threads=self.params["params"]["num_threads"]
        )
        c = 2.0 * (self.params["params"]["degrees_of_freedom"] + 1.0) / self.params["params"]["degrees_of_freedom"]
        grad = grad.ravel()
        grad *= c

        return error, grad

    #####################
    # Utility functions #
    #####################


class HyperbolicKL(BaseCostFunction):
    def __init__(self, *, n_components, other_params=None):
        if other_params is None:
            raise Exception(
                "No `other_params` specified for HyperbolicKL, please add your params or select one of the presets.")
        super(HyperbolicKL, self).__init__(n_components=n_components, other_params=other_params)
        self.results = []

    @property
    def params(self):
        return super(HyperbolicKL, self).params

    @params.setter
    def params(self, other_params):
        # Nice trick for using super setter: https://gist.github.com/Susensio/979259559e2bebcd0273f1a95d7c1e79
        super(HyperbolicKL, type(self)).params.fset(self, other_params)
        # HyperbolicKL.check_params(self.params)

    @classmethod
    def check_params(cls, params):
        general_params = ["num_threads", "verbose", "degrees_of_freedom", "hyperbolic_model", "calc_both", "area_split"]
        if params["method"] == "exact":
            all_params = general_params + ["skip_num_points"]
        elif params["method"] == "barnes-hut":
            all_params = general_params + ["angle"]
        else:
            raise Exception("HyperbolicKL method is not a valid one (available methods are `exact` and `barnes-hut`)")

        for p in params["params"]:
            if p not in all_params:
                raise Exception(
                    "%s is not in the param set of the `%s` version of HyperbolicKL." % (p, params["method"]))
        for p in all_params:
            if p not in params["params"]:
                raise Exception(
                    "%s params is necessary for the `%s` version of HyperbolicKL. Please set a value or use a preset." %
                    (p, params["method"]))
        # TODO: add checks for specific values of parameters

    #######################
    # Parameter templates #
    #######################

    @classmethod
    def exact_tsne(cls):
        return {
            "method": "exact",
            "params": {"degrees_of_freedom": 1, "skip_num_points": 0, "num_threads": _openmp_effective_n_threads(), "verbose": False}
        }

    @classmethod
    def bh_tsne(cls, angle=0.5):
        return {
            "method": "barnes-hut",
            "params": {"angle": angle, "degrees_of_freedom": 1, "num_threads": _openmp_effective_n_threads(),
                       "verbose": False}
        }

    #########################
    # User-facing functions #
    #########################

    def obj(self, Y, *, V, forces="total", per_point=False):
        n_samples = V.shape[0]
        if self.params["method"] == "exact":
            obj, _ = self._obj_exact(Y, V, n_samples, forces, per_point)
            return obj
        elif self.params["method"] == "barnes-hut":
            obj, _ = self._obj_bh(Y, V, n_samples)
            return obj

    def grad(self, Y, *, V, forces="total", per_point=False):
        n_samples = V.shape[0]
        if self.params["method"] == "exact":
            return self._grad_exact(Y, V, n_samples, forces, per_point)
        elif self.params["method"] == "barnes-hut":
            _, grad = self._grad_bh(Y, V, n_samples)
            return grad

    def obj_grad(self, Y, *, V, forces="total", per_point=False):
        n_samples = V.shape[0]
        if self.params["method"] == "exact":
            obj, grad = self._grad_exact(Y, V, n_samples, forces, per_point)
            return obj, grad
        elif self.params["method"] == "barnes-hut":
            obj, grad = self._obj_bh(Y, V, n_samples)
            return obj, grad

    ##########################
    # Main private functions #
    ##########################

    def _obj_exact(self, Y, V, n_samples):
        # TODO
        pass

    def _grad_exact(self, Y, V, n_samples, save_timings=True):
        Y = Y.astype(ctypes.c_double, copy=False)
        Y = Y.reshape(n_samples, self.n_components)

        val_V = V.data
        neighbors = V.indices.astype(np.int64, copy=False)
        indptr = V.indptr.astype(np.int64, copy=False)

        grad = np.zeros(Y.shape, dtype=ctypes.c_double)
        timings = np.zeros(4, dtype=ctypes.c_float)
        error = tsne_barnes_hut_hyperbolic.gradient(
            timings,
            val_V, Y, neighbors, indptr, grad,
            0.5,
            self.n_components,
            self.params["params"]["verbose"],
            dof=self.params["params"]["degrees_of_freedom"],
            compute_error=True,
            num_threads=self.params["params"]["num_threads"],
            exact=True
        )

        grad = grad.ravel()
        grad *= 4

        if save_timings:
            self.results.append(timings)

        return error, grad

    def _obj_bh(self, Y, V, n_samples):
        return self._grad_bh(Y, V, n_samples)

    def _grad_bh(self, Y, V, n_samples):
        Y = Y.astype(ctypes.c_double, copy=False)
        Y = Y.reshape(n_samples, self.n_components)

        val_V = V.data
        neighbors = V.indices.astype(np.int64, copy=False)
        indptr = V.indptr.astype(np.int64, copy=False)

        grad = np.zeros(Y.shape, dtype=ctypes.c_double)
        timings = np.zeros(4, dtype=ctypes.c_float)
        error = tsne_barnes_hut_hyperbolic.gradient(
            timings,
            val_V, Y, neighbors, indptr, grad,
            self.params["params"]["angle"],
            self.n_components,
            self.params["params"]["verbose"],
            dof=self.params["params"]["degrees_of_freedom"],
            compute_error=True,
            num_threads=self.params["params"]["num_threads"],
            exact=False,
            area_split=self.params["params"]["area_split"]
        )

        grad = grad.ravel()
        grad *= 4

        return error, grad
