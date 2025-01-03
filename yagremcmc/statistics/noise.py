import numpy as np

from yagremcmc.statistics.interface import NoiseModelInterface
from yagremcmc.statistics.covariance import (CovarianceMatrix,
                                             DiagonalCovarianceMatrix)


class CentredGaussianNoise(NoiseModelInterface):

    def __init__(self, covariance: CovarianceMatrix):

        if not isinstance(covariance, CovarianceMatrix):
            raise ValueError("CentredGaussianNoise need to be instantiated "
                             f"with CovarianceMatrix. Got: {type(covariance)}")
        self._cov = covariance

    @property
    def covariance(self):
        return self._cov

    def induced_norm_squared(self, vector) -> float:
        return self._cov.induced_norm_squared(vector)


class AEMNoise(CentredGaussianNoise):

    def __init__(self, measurementNoise, useHeuristic):

        if not isinstance(measurementNoise.covariance,
                          DiagonalCovarianceMatrix):
            raise NotImplementedError(
                "Currently, AEM is only implemented for independent "
                "measurement noise.")

        self._dataNoise = measurementNoise
        self._aemNoise = None
        self._useHeuristic = useHeuristic

    def scaling_heuristic(mVar, eps=1e-6, maxScaling=100):

        minVal = max(np.min(mVar), eps)
        scaling = 2. * np.max(mVar) / minVal

        return min(scaling, maxScaling)


    def set_error_marginal_variance(self, mVar):

        noiseScaling = AEMNoise.scaling_heuristic(mVar) \
            if self._useHeuristic else 1.

        aemCov = DiagonalCovarianceMatrix(
            noiseScaling * mVar + self._dataNoise.covariance.marginalVariance)
        self._aemNoise = CentredGaussianNoise(aemCov)

    def induced_norm_squared(self, vector):

        if self._aemNoise is None:
            return self._dataNoise.induced_norm_squared(vector)

        return self._aemNoise.induced_norm_squared(vector)
