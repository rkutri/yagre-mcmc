import numpy as np

from abc import abstractmethod
from typing import Any
from yagremcmc.parameter.interface import ParameterInterface
from yagremcmc.utility.memoisation import EvaluationCache, AEMCache
from yagremcmc.model.evaluation import AEMEvaluation
from yagremcmc.statistics.interface import DensityInterface
from yagremcmc.statistics.noise import CentredGaussianNoise, AEMNoise
from yagremcmc.statistics.estimation import WelfordAccumulator


class AdditiveNoiseLikelihood(DensityInterface):

    def __init__(self, data, forwardModel, noiseModel):

        self._data = data
        self._fwdModel = forwardModel
        self._noiseModel = noiseModel

    @abstractmethod
    def query_model_evaluation(self, parameter: ParameterInterface):
        pass

    @abstractmethod
    def query_log_likelihood(self, parameter: ParameterInterface):
        pass

    @abstractmethod
    def compute_residual(self, modelEval: Any):
        pass

    def compute_residual_norm_squared(self, forwardModelEval):

        def norm_squared(x):
            return self._noiseModel.induced_norm_squared(x)

        residual = self.compute_residual(forwardModelEval)
        return np.apply_along_axis(norm_squared, 1, residual)

    @abstractmethod
    def compute_log_likelihood(self, parameter: ParameterInterface):
        pass

    def evaluate_log(self, parameter: ParameterInterface):
        return self.query_log_likelihood(parameter)


class AdditiveGaussianNoiseLikelihood(AdditiveNoiseLikelihood):

    CACHESIZE = 5

    def __init__(self, data, forwardModel, noiseModel):

        if not isinstance(noiseModel, CentredGaussianNoise):
            raise ValueError("AdditiveGaussianNoiseLikelihood requires "
                             "centred Gaussian noise.")

        super().__init__(data, forwardModel, noiseModel)

        self._llCache = EvaluationCache(
            AdditiveGaussianNoiseLikelihood.CACHESIZE)

    def query_model_evaluation(self, parameter):
        return self._fwdModel.evaluate(parameter)

    def query_log_likelihood(self, parameter):

        if self._llCache.contains(parameter):
            return self._llCache.retrieve(parameter)

        return self.compute_log_likelihood(parameter)

    def compute_residual(self, modelEval):
        return modelEval - self._data.array

    def compute_log_likelihood(self, parameter):

        fmEval = self.query_model_evaluation(parameter)
        logL = -0.5 * np.sum(self.compute_residual_norm_squared(fmEval))

        self.update_cache(parameter, fmEval, logL)

        return logL

    def update_cache(self, parameter, fmEval, logL):
        self._llCache.add(parameter, logL)


class AEMLikelihood(AdditiveGaussianNoiseLikelihood):

    def __init__(self,
                 data,
                 forwardModel,
                 noiseModel,
                 minDataSize,
                 useNoiseHeuristic=False):

        if minDataSize < 2:
            raise ValueError("Smallest senisible data size for AEM is 2.")

        self._minDataSize = minDataSize
        self._accumulator = WelfordAccumulator()

        noiseModel = AEMNoise(noiseModel, useNoiseHeuristic)
        super().__init__(data, forwardModel, noiseModel)

        self._cache = AEMCache()
        self._nModelEvaluations = 0

    @property
    def accumulator(self):
        return self._accumulator

    def number_of_model_evaluations(self):
        return self._nModelEvaluations

    def query_model_evaluation(self, parameter: ParameterInterface):

        if self._cache.contains(parameter):
            return self._cache.retrieve(parameter)[0]

        self._nModelEvaluations += 1
        return self._fwdModel.evaluate(parameter)

    def query_log_likelihood(self, parameter: ParameterInterface):

        if self._cache.contains(parameter):
            return self._cache.retrieve(parameter)[1]

        return self.compute_log_likelihood(parameter)

    def compute_residual(self, modelEval):

        if self._accumulator.nData < self._minDataSize:
            return super().compute_residual(modelEval)

        return super().compute_residual(modelEval) + self._accumulator.mean()

    def update_cache(self, parameter, fmEval, logL):

        if self._cache.contains(parameter):
            return

        aemEval = AEMEvaluation(fmEval, logL)
        self._cache.add(parameter, aemEval)

    def update_error_estimate(self, errorRealisation):

        self._accumulator.update(errorRealisation)

        if self._accumulator.nData > self._minDataSize:

            mVar = self._accumulator.marginal_variance()
            self._noiseModel.set_error_marginal_variance(mVar)
