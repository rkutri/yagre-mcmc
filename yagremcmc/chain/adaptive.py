import numpy as np

from yagremcmc.statistics.interface import CovarianceOperatorInterface
from yagremcmc.statistics.covariance import DenseCovarianceMatrix


class AdaptiveCovarianceMatrix(CovarianceOperatorInterface):

    def __init__(self, dimension, regularisationParameter):

        self.dim_ = dimension
        self.eps_ = regularisationParameter
        self.scaling_ = 1.
        self.nData_ = 0

        self.mean_ = None
        self.cov_ = None

    @property
    def dimension(self):
        return self.dim_

    @property
    def scaling(self):
        return self.scaling_

    @scaling.setter
    def scaling(self, value):
        self.scaling_ = value

    def dense_covariance_matrix(self):
        return self.cov_.dense()

    def initialise(self, nData, chain):

        data = chain.trajectory[-nData:]

        self.mean_ = np.mean(data)
        sampCov = np.cov(np.vstack(data), rowvar=False, bias=False)

        adaptCov = self._dimension_scaling() * (1. - self.eps_) * sampCov
        for i in range(self.dim_):
            adaptCov[i, i] += self.eps_

        self.cov_ = DenseCovarianceMatrix(adaptCov)
        self.cov_.scaling = self.scaling_

        self.nData_ = nData

    def update(self, vector):

        if self.mean_ is None or self.cov_ is None:
            raise ValueError(
                "Calling update before covariance is initialised.")

        n = self.nData_
        nPlus = self.nData_ + 1
        nMinus = self.nData_ - 1

        newMean = (self.nData_ * self.mean_ + vector) / nPlus

        updCov = (nMinus / n) * self.cov_.dense() \
            + self._dimension_scaling() / n \
                * (n * np.outer(self.mean_, self.mean_)
                    - nPlus * np.outer(newMean, newMean)
                    + np.outer(vector, vector)) \
            + self.eps_ * self._dimension_scaling() / n * np.eye(self.dim_)

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
