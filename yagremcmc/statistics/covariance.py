import numpy as np

from scipy.linalg import cholesky, solve_triangular
from yagremcmc.statistics.interface import CovarianceOperatorInterface


class DiagonalCovarianceMatrix(CovarianceOperatorInterface):
    """
    Covariance matrix of independent random variables.

    Instance Attributes
    -------------------
    _precision : numpy.ndarray
        One-dimensional array storing the reciprocals of the diagonal entries of 
        the (diagonal) covariance matrix
    """

    def __init__(self, marginalVariances):

        self._precision = np.reciprocal(marginalVariances)

    @property
    def dimension(self):
        return self._precision.size

    def apply_chol_factor(self, x):
        return np.sqrt(np.reciprocal(self._precision)) * x

    def apply_inverse(self, x):
        return self._precision * x


class IIDCovarianceMatrix(DiagonalCovarianceMatrix):
    """
    Covariance matrix of i.i.d. random variables.
    """

    def __init__(self, dimension, variance):

        margVar = np.full(dimension, variance)
        super().__init__(margVar)


class DenseCovarianceMatrix(CovarianceOperatorInterface):

    def __init__(self, denseCovMat):

        s = denseCovMat.shape
        assert s[0] == s[1]

        self.dim_ = s[0]

        self.cholFactor_ = cholesky(denseCovMat, lower=True)

    @property
    def dimension(self):
        return self.dim_

    def apply_chol_factor(self, x):

        return self.cholFactor_ @ x

    def apply_inverse(self, x):

        y = solve_triangular(self.cholFactor_, x, lower=True)

        return solve_triangular(self.cholFactor_.T, y, lower=False)

    def dense(self):

        return np.matmul(self.cholFactor_, self.cholFactor_.T)
