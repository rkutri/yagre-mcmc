from numpy import dot, reciprocal
from yagremcmc.statistics.interface import NoiseModelInterface
from yagremcmc.statistics.covariance import CovarianceMatrix


class CentredGaussianNoise(NoiseModelInterface):

    def __init__(self, covariance: CovarianceMatrix):
        self._cov = covariance

    def induced_norm_squared(self, vector) -> float:
        return self._cov.induced_norm_squared(vector)


class AEMNoise(NoiseModelInterface):

    def __init__(self, measurementNoise):

        self._dataNoise = measurementNoise
        self._errorMVar = None

    def induced_norm_squared(self, vector):

        dataNSq = self._dataNoise.induced_norm_squared(vector)

        if self._errorMVar is None:
            return dataNSq

        errorNSq = reciprocal(self._errorMVar) * vector

        return dataNSq + errorNSq

    def set_error_marginal_variance(self, mVar):
        self._errorMVar = mVar
