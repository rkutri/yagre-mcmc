from numpy import sqrt, reciprocal, full
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

        self.precision_ = reciprocal(marginalVariances)
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
        return sqrt(self.scaling_ * reciprocal(self.precision_)) * x

    def apply_inverse(self, x):
        return self.precision_ * x / self.scaling_


class IIDCovarianceMatrix(DiagonalCovarianceMatrix):
    """
    Covariance matrix of i.i.d. random variables.
    """

    def __init__(self, dimension, variance):

        margVar = full(dimension, variance)
        super().__init__(margVar)


class DenseCovarianceMatrix(CovarianceOperatorInterface):
    pass
