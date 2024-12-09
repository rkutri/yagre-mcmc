import numpy as np

from yagremcmc.utility.memoisation import EvaluationCache
from yagremcmc.statistics.interface import DensityInterface
from yagremcmc.statistics.estimation import WelfordAccumulator 


class AdditiveNoiseLikelihood(DensityInterface):
    def __init__(self, data, forwardModel, noiseModel):

        self.data_ = data
        self.fwdModel_ = forwardModel
        self.noiseModel_ = noiseModel

        # Cache for memoized evaluation of likelihood
        cacheSize = 6
        self.llCache_ = EvaluationCache(cacheSize)

    def evaluate_log(self, parameter):
        """
        Evaluate the log-likelihood.

        Uses memoization for efficiency. If the log-likelihood for the given
        parameter is already cached, it is retrieved from the cache. Otherwise,
        it is computed as:
            logL = -0.5 * ||data - forward_model(parameter)||^2_noise
        """

        if self.llCache_.contains(parameter):
            return self.llCache_(parameter)

        dataMisfit = self.fwdModel_.evaluate(parameter) - self.data_.array

        dmNormSquared = np.apply_along_axis(
            lambda x: self.noiseModel_.induced_norm_squared(x), 1, dataMisfit
        )

        logL = -0.5 * np.sum(dmNormSquared)
        self.llCache_.add(parameter, logL)

        return logL


class AdaptiveErrorCorrection(DensityInterface):

    def __init__(self, likelihood : AdditiveNoiseLikelihood):

        self._likelihood = likelihood
        self._errorEstimator = WelfordAccumulator()

    def evaluate_log(self, parameter):
        pass



