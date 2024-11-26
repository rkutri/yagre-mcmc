import numpy as np

from yagremcmc.utility.memoisation import EvaluationCache
from yagremcmc.statistics.interface import DensityInterface


class AdditiveGaussianNoiseLikelihood(DensityInterface):
    def __init__(self, data, forwardModel, noiseModel, tempering=1.0):
        """
            Tempering is always applied, however with default parameter 1.0.
        """
        self.data_ = data
        self.fwdModel_ = forwardModel
        self.noiseModel_ = noiseModel

        if tempering < 0.0 or tempering > 1.0:
            raise ValueError(f"Invalid tempering parameter value: {tempering}")

        self._tempering = tempering

        # Cache for memoized evaluation of likelihood
        cacheSize = 6
        self.llCache_ = EvaluationCache(cacheSize)

    def evaluate_log(self, parameter):
        """
        Evaluate the tempered log-likelihood.

        Uses memoization for efficiency. If the log-likelihood for the given
        parameter is already cached, it is retrieved from the cache. Otherwise,
        it is computed as:
            logL = -0.5 * ||data - forward_model(parameter)||^2_noise
        scaled by the tempering factor.
        """

        if self.llCache_.contains(parameter):
            return self.llCache_(parameter)

        dataMisfit = self.fwdModel_.evaluate(parameter) - self.data_.array

        dmNormSquared = np.apply_along_axis(
            lambda x: self.noiseModel_.induced_norm_squared(x), 1, dataMisfit
        )

        logL = -0.5 * np.sum(dmNormSquared)

        # Apply tempering and cache the tempered result
        temperedLogL = self._tempering * logL
        self.llCache_.add(parameter, temperedLogL)

        return temperedLogL
