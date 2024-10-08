import numpy as np

from yagremcmc.utility.memoisation import EvaluationCache
from yagremcmc.statistics.interface import DensityInterface


class BayesianRegressionLikelihood(DensityInterface):

    def __init__(self, data, forwardModel, noiseModel):

        self.data_ = data
        self.fwdModel_ = forwardModel
        self.noiseModel_ = noiseModel

        # in theory, only state and proposal likelihoods are required
        cacheSize = 6
        self.llCache_ = EvaluationCache(cacheSize)

    def evaluate_log(self, parameter):
        """
        memoised evaluation of the log-likelihood
        """

        if self.llCache_.contains(parameter):
            return self.llCache_(parameter)

        dataMisfit = self.fwdModel_.evaluate(parameter) - self.data_.array

        dmNormSquared = np.apply_along_axis(
            lambda x: self.noiseModel_.induced_norm_squared(x), 1, dataMisfit)

        logL = -0.5 * np.sum(dmNormSquared)

        self.llCache_.add(parameter, logL)

        return logL
