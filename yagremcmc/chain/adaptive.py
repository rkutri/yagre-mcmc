import numpy as np

from yagremcmc.statistics.interface import CovarianceOperatorInterface
from yagremcmc.statistics.covariance import DenseCovarianceMatrix


class ASCovarianceMatrix(CovarianceOperatorInterface):
    pass


class AMCovarianceMatrix(CovarianceOperatorInterface):

    def __init__(self, initMean, initSampCov, eps, nData):

        self.mean_ = initMean
        self.eps_ = eps
        self.nData_ = nData

        self.dim_ = initMean.size
        self.scaling_ = 1.

        regCov = self._dimension_scaling() \
            * ((1. - eps) * initSampCov + eps * np.eye(self.dim_))
        self.cov_ = DenseCovarianceMatrix(regCov)

    @property
    def dimension(self):
        return self.dim_

    @property
    def scaling(self):
        return self.scaling_

    @scaling.setter
    def scaling(self, value):
        self.scaling_ = value

    @property
    def nData(self):
        return self.nData_

    def dense_covariance_matrix(self):
        return self.cov_.dense()

    def update(self, vector):

        n = self.nData_
        nPlus = self.nData_ + 1
        nMinus = self.nData_ - 1

        newMean = (self.nData_ * self.mean_ + vector) / nPlus

        updCov = (nMinus / n) * self.cov_.dense() \
            + self._dimension_scaling() / n \
            * (n * np.outer(self.mean_, self.mean_)
               - nPlus * np.outer(newMean, newMean)
               + np.outer(vector, vector)
               + self.eps_ * np.eye(self.dim_))

        self.nData_ = nPlus
        self.mean_ = newMean
        self.cov_ = DenseCovarianceMatrix(updCov)

        self.cov_.scaling = self.scaling_

    def apply_chol_factor(self, x):
        return self.cov_.apply_chol_factor(x)

    def apply_inverse(self, x):
        return self.cov_.apply_inverse(x)

    def _dimension_scaling(self):
        return 4. / self.dim_
