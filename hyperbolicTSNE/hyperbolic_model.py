from abc import ABC, abstractmethod

import numpy as np
from scipy import linalg
from scipy.spatial.distance import pdist, squareform

MACHINE_EPSILON = np.finfo(np.double).eps


class BaseHyperbolicModel(ABC):
    def __init__(self, Y):
        self.Y = Y

    @abstractmethod
    def dist(self):
        raise NotImplementedError

    @abstractmethod
    def dist_grad(self):
        raise NotImplementedError

    def metric_tensor(self):
        return 1

    @classmethod
    def exp_map(cls, y, grad, n_samples):
        return y + grad

    @staticmethod
    def constrain(y, n_samples):
        return y

    @staticmethod
    def init_proj(y, n_samples):
        return y

    @classmethod
    def proj(cls, y, grad):
        return grad


class PoincareDiskModel(BaseHyperbolicModel):
    def __init__(self, Y):
        super().__init__(Y)

        self.e_dist = pdist(Y, "sqeuclidean")
        self.e_dist = squareform(self.e_dist, checks=False)

        # TODO: uv_dot = np.einsum('ab,cb->ac', self.Y, self.Y) diagonal
        self.norm = np.sum(Y ** 2, axis=1)

        self.norm_inv = np.maximum(1 - self.norm, MACHINE_EPSILON)
        self.sq_norm_inv = self.norm_inv ** 2
        self.gamma = 1 + 2 * self.e_dist / np.outer(self.norm_inv, self.norm_inv)

    def dist(self):
        h_dist = np.arccosh(self.gamma)
        h_dist = squareform(h_dist, checks=False)
        return h_dist

    def dist_grad(self):
        alpha = self.norm_inv[:, np.newaxis]
        alpha_squared = self.sq_norm_inv[:, np.newaxis]
        beta = self.norm_inv
        v_norm = self.norm
        u = self.Y[:, np.newaxis]
        v = self.Y

        # np.dot(u, v)
        uv_dot = np.einsum('ab,cb->ac', self.Y, self.Y)

        u_scalar = (v_norm - 2 * uv_dot + 1) / alpha_squared
        v_scalar = 1. / alpha

        shared_scalar = 4 / np.maximum(beta * np.sqrt(self.gamma ** 2 - 1), MACHINE_EPSILON)

        scaled_u = u_scalar[:, :, np.newaxis] * u
        scaled_v = v_scalar[:, :, np.newaxis] * v

        return shared_scalar[:, :, np.newaxis] * (scaled_u - scaled_v)

    def metric_tensor(self):
        gx = self.sq_norm_inv / 4
        return gx[:, np.newaxis]

    @classmethod
    def exp_map(cls, y, grad, n_samples):
        y = y.reshape(n_samples, 2)
        grad = grad.reshape(n_samples, 2)
        res = np.empty((n_samples, 2))

        for i in range(n_samples):
            z = y[i]
            z_norm_sq = np.linalg.norm(z) ** 2

            v = grad[i]

            metric = 2 / (1 - z_norm_sq)
            v_norm = np.linalg.norm(v)

            x = np.tanh((metric * v_norm) / 2) * (v / v_norm)
            x_norm_sq = np.linalg.norm(x) ** 2

            r_term = 1 + 2 * np.dot(z, x)

            z_scalar = (r_term + x_norm_sq)
            x_scalar = (1 - z_norm_sq)

            numerator = z_scalar * z + x_scalar * x
            denominator = r_term + z_norm_sq * x_norm_sq

            res[i] = numerator / denominator

        return res.ravel()

    @staticmethod
    def constrain(y, n_samples):
        y = y.reshape(n_samples, 2)
        for i, a in enumerate(y):
            a_norm = linalg.norm(a)
            if a_norm >= 1:
                y[i] = (y[i] / a_norm) - 1e6
        return y.ravel()


class LorentzModel(BaseHyperbolicModel):
    def __init__(self, Y):
        super().__init__(Y)

        self.Y_scalar_product = self.scalar_product(self.Y, self.Y)
        max_mdp = -(1 + 1e-10)
        self.Y_scalar_product[self.Y_scalar_product > max_mdp] = max_mdp

    @staticmethod
    def scalar_product(x, y):
        x0 = x[:, 0]
        xn = x[:, 1:]

        y0 = y[:, 0]
        yn = y[:, 1:]

        z = np.outer(x0, y0)

        # np.dot(u, v)
        n = np.einsum('ab,cb->ac', xn, yn)

        return -z + n

    def dist(self):
        h_dist = np.arccosh(-self.Y_scalar_product)
        h_dist = squareform(h_dist, checks=False)
        return h_dist

    def dist_grad(self):
        scalar = -1 / np.sqrt(self.Y_scalar_product ** 2 - 1)

        np.fill_diagonal(scalar, 0)

        return scalar[:, :, np.newaxis] * self.Y

    # @staticmethod
    # def init_proj(y, n_samples):
    #     res = np.empty((n_samples, 3))
    #
    #     for i, p in enumerate(y):
    #         norm_squared = np.linalg.norm(p) ** 2
    #         proj_scalar = 1 / (1 - norm_squared + MACHINE_EPSILON)
    #         t = 1 + norm_squared
    #         res[i] = proj_scalar * np.concatenate(([t], 2 * p))
    #
    #     return res

    @staticmethod
    def init_proj(y, n_samples):
        res = np.empty((n_samples, 3))

        for i, p in enumerate(y):
            res[i] = np.concatenate(([np.sqrt(1 + np.linalg.norm(p) ** 2)], p))

        return res

    def metric_tensor(self):
        return np.array([-1, 1, 1])

    @classmethod
    def exp_map(cls, y, grad, n_samples):
        y = y.reshape(n_samples, 3)
        grad = grad.reshape(n_samples, 3)
        grad_scalar_product = cls.scalar_product(grad, grad)

        for i, g in enumerate(grad):
            g_norm = np.sqrt(np.maximum(grad_scalar_product[i, i], 0))

            if g_norm == 0:
                grad[i] = y[i]
                continue

            grad[i] = np.cosh(g_norm) * y[i] + np.sinh(g_norm) * (g / g_norm)

        return grad.ravel()

    @classmethod
    def proj(cls, y, grad):
        product = cls.scalar_product(y, grad)
        scal = np.diagonal(product)[:, np.newaxis]
        return grad + scal * y


