import numpy as np

from abc import abstractmethod
from typing import Any
from yagremcmc.parameter.interface import ParameterInterface
from yagremcmc.utility.memoisation import EvaluationCache
from yagremcmc.statistics.interface import DensityInterface
from yagremcmc.statistics.noise import CentredGaussianNoise, AEMNoise
from yagremcmc.statistics.estimation import WelfordAccumulator


class AdditiveNoiseLikelihood(DensityInterface):

    def __init__(self, data, forwardModel, noiseModel):

        self._data = data
        self._fwdModel = forwardModel
        self._noiseModel = noiseModel

        # Cache for memoized evaluation of likelihood
        cacheSize = 8
        self._llCache = EvaluationCache(cacheSize)

    @abstractmethod
    def process_model_evaluation(self, parameter: ParameterInterface):
        pass

    @abstractmethod
    def compute_residual(self, modelEval: Any):
        pass

    @abstractmethod
    def compute_log_likelihood(self, parameter: ParameterInterface):
        pass

    def evaluate_log(self, parameter: ParameterInterface):

        if self._llCache.contains(parameter):
            return self._llCache(parameter)

        modelEval = self.process_model_evaluation(parameter)
        residual = self.compute_residual(modelEval)

        logL = self.compute_log_likelihood(residual)
        self._llCache.add(parameter, logL)

        return logL


class AdditiveGaussianNoiseLikelihood(AdditiveNoiseLikelihood):

    def __init__(self, data, forwardModel, noiseModel):

        if not isinstance(noiseModel, CentredGaussianNoise):
            raise ValueError("AdditiveGaussianNoiseLikelihood requires "
                             "centred Gaussian noise.")

        super().__init__(data, forwardModel, noiseModel)

    def process_model_evaluation(self, parameter):
        return self._fwdModel.evaluate(parameter)

    def compute_residual(self, modelEval):
        return modelEval - self._data.array

    def compute_log_likelihood(self, residual):

        residualNormSq = np.apply_along_axis(
            lambda x: self._noiseModel.induced_norm_squared(x), 1, residual)

        return -0.5 * np.sum(residualNormSq)


class AEMLikelihood(AdditiveGaussianNoiseLikelihood):

    def __init__(self, data, forwardModel, noiseModel, minDataSize):

        self._minDataSize = minDataSize
        self._errorEstimator = WelfordAccumulator()

        noiseModel = AEMNoise(noiseModel)
        super().__init__(data, forwardModel, noiseModel)

        fwdEvalCacheSize = 2
        self._fwdEvalCache = EvaluationCache(fwdEvalCacheSize)

    def update_error_estimate(self, errorRealisation):

        self._errorEstimator.update(errorRealisation)

        mVar = self._errorEstimator.marginal_variance()
        self._noiseModel.set_error_marginal_variance(mVar)

    def cached_model_evaluation(self, parameter):

        if self._fwdEvalCache.contains(parameter):
            return self._fwdEvalCache(parameter)
        else:
            raise RuntimeError("Cache miss for forward model evaluation at "
                               f"parameter coefficient: {parameter.coefficient}.\n"
                               "last cached values for: \n"
                               f"p0 = {self._fwdEvalCache.keys[0].coefficient} \n"
                               f"p1 = {self._fwdEvalCache.keys[1].coefficient}")

    def process_model_evaluation(self, parameter):

        if self._fwdEvalCache.contains(parameter):
            return self._fwdEvalCache(parameter)

        modelEval = self._fwdModel.evaluate(parameter)
        self._fwdEvalCache.add(parameter, modelEval)

        return modelEval

    def compute_residual(self, modelEval):

        if self._errorEstimator.nData < self._minDataSize:
            return super().compute_residual(modelEval)

        return modelEval + self._errorEstimator.mean() - self._data.array

    def compute_log_likelihood(self, residual):

        if self._errorEstimator.nData < self._minDataSize:
            return super().compute_log_likelihood(residual)

        residualNormSq = np.apply_along_axis(
            lambda x: self._sumNoiseModel.induced_norm_squared(x), 1, residual)

        return -0.5 * np.sum(residualNormSq)
