from yagremcmc.statistics.interface import CovarianceOperatorInterface
from yagremcmc.statistics.covariance import DenseCovarianceMatrix


class AdaptiveCovarianceMatrix(CovarianceOperatorInterface):

    def __init__(self, dimension, regularisationParameter):
        pass

    @property
    def dimension(self):
        pass

    @property
    def scaling(self):
        pass

    @scaling.setter
    def scaling(self, value):
        pass

    def update(self, vector):
        pass

    def apply_chol_factor(self, x):
        pass

    def apply_inverse(self, x):
        pass
