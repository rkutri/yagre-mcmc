import numpy as np

from scipy.linalg import cholesky, solve_triangular
from yagremcmc.statistics.interface import CovarianceOperatorInterface


class DiagonalCovarianceMatrix(CovarianceOperatorInterface):
    """
    Covariance matrix of independent random variables.

    Instance Attributes
    -------------------
    precision_ : numpy.ndarray
        One-dimensional array storing the reciprocals of the diagonal entries of 
        the (diagonal) covariance matrix
    """

    def __init__(self, marginalVariances):

        self.precision_ = np.reciprocal(marginalVariances)
        self.scaling_ = 1.

    @property
    def dimension(self):
        return self.precision_.size

    @property
    def scaling(self):
        return self.scaling_

    @scaling.setter
    def scaling(self, value):
        self.scaling_ = value

    def apply_chol_factor(self, x):
        return np.sqrt(self.scaling_ * np.reciprocal(self.precision_)) * x

    def apply_inverse(self, x):
        return self.precision_ * x / self.scaling_


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

        self.scaling_ = 1.

    @property
    def dimension(self):
        return self.dim_

    @property
    def scaling(self):
        return self.scaling_

    @scaling.setter
    def scaling(self, value):
        self.scaling_ = value

    def apply_chol_factor(self, x):

        return self.cholFactor_ @ x

    def apply_inverse(self, x):

        y = solve_triangular(self.cholFactor_, x, lower=True)

        return solve_triangular(self.cholFactor_.T, y, lower=False)
    
    def dense(self):

        return np.matmul(self.cholFactor_, self.cholFactor_.T)
        
