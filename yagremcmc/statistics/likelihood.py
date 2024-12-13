import numpy as np

from yagremcmc.utility.memoisation import EvaluationCache
from yagremcmc.statistics.interface import DensityInterface
from yagremcmc.statistics.noise import CentredGaussianNoise
from yagremcmc.statistics.estimation import WelfordAccumulator


class AdditiveGaussianNoiseLikelihood(DensityInterface):
    def __init__(self, data, forwardModel, noiseModel):

        self._data = data
        self._fwdModel = forwardModel

        if not isinstance(noiseModel, CentredGaussianNoise):
            raise ValueError("AdditiveGaussianNoiseLikelihood requires "
                             "Gaussian noise.")

        self._noiseModel = noiseModel

        # Cache for memoized evaluation of likelihood
        cacheSize = 8
        self._llCache = EvaluationCache(cacheSize)

    def process_model_evaluation(self, parameter):
        return self._fwdModel.evaluate(parameter)

    def compute_residual(self, modelEval):
        return modelEval - self._data.array

    def compute_log_likelihood(self, residual):

        residualNormSq = np.apply_along_axis(
            lambda x: self._noiseModel.induced_norm_squared(x), 1, residual)

        return -0.5 * np.sum(residualNormSq)

    def evaluate_log(self, parameter):
        """
        Evaluate the log-likelihood.

        Uses memoization for efficiency. If the log-likelihood for the given
        parameter is already cached, it is retrieved from the cache. Otherwise,
        it is computed as:
            logL = -0.5 * ||data - forward_model(parameter)||^2_noise
        """

        if self._llCache.contains(parameter):
            return self._llCache(parameter)

        modelEval = self.process_model_evaluation(parameter)
        residual = self.compute_residual(modelEval)

        logL = self.compute_log_likelihood(residual)
        self._llCache.add(parameter, logL)

        return logL


class AdaptiveErrorCorrectionLikelihood(AdditiveGaussianNoiseLikelihood):

    def __init__(self, data, forwardModel, noiseModel, minDataSize=500):

        super().__init__(data, forwardModel, noiseModel)

        self._sumNoiseModel = IndependentGaussianSumNoise(self._noiseModel)

        self._errorEstimator = WelfordAccumulator()

        fwdEvalCS = 2
        self._fwdEvalCache = EvaluationCache(fwdEvalCS)

    def update_error_estimate(self, errorRealisation):

        self._errorEstimator.update(errorRealisation)

        mVar = self._errorEstimator.marginal_variance()
        self._sumNoiseModel.update_model_error_noise(mVar)

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

        if self._errorEstimator.nData < minDataSize:
            return super().compute_residual(modelEval)

        return modelEval + self._errorEstimator.mean() - self._data.array

    def compute_log_likelihood(self, residual):

        if self._errorEstimator.nData < minDataSize:
            return super().compute_log_likelihood(residual)

        residualNormSq = np.apply_along_axis(
            lambda x: self._sumNoiseModel.induced_norm_squared(x), 1, residual)

        return -0.5 * np.sum(residualNormSq)
