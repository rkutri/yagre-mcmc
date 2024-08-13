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

    def apply_sqrt(self, x):
        return sqrt(reciprocal(self.precision_)) * x

    def apply_inverse(self, x):
        return self.precision_ * x


class IIDCovarianceMatrix(DiagonalCovarianceMatrix):
    """
    Covariance matrix of i.i.d. random variables.
    """

    def __init__(self, dimension, variance):

        margVar = full(dimension, variance)
        super().__init__(margVar)


class AdaptiveCovariance(CovarianceOperatorInterface):
    pass
