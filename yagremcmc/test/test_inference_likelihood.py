import numpy as np
import pytest

from yagremcmc.statistics.data import Data
from yagremcmc.statistics.likelihood import AdditiveGaussianNoiseLikelihood
from yagremcmc.statistics.noise import CentredGaussianNoise
from yagremcmc.statistics.covariance import CovarianceMatrix
from yagremcmc.parameter.vector import ParameterVector
from yagremcmc.utility.memoisation import EvaluationCache


class MockNoise(CentredGaussianNoise):

    class DummyCovariance(CovarianceMatrix):

        @property
        def dimension(self):
            pass

        def apply_chol_factor(self, x):
            pass
        
        def apply_inverse(self, x):
            pass

        def induced_norm_squared(self, x):
            pass

    def __init__(self):

        super().__init__(covariance=MockNoise.DummyCovariance())

    def induced_norm_squared(self, x):
        return np.sqrt(np.sum(np.square(x)))


class MockForwardModel:
    def evaluate(self, parameter):
        return np.sum(parameter.coefficient)


@pytest.fixture
def mock_noise():
    return MockNoise()


@pytest.fixture
def mock_data():
    return Data(np.zeros((2, 2)))


@pytest.fixture
def mock_forward_model():
    return MockForwardModel()


@pytest.fixture
def mock_likelihood(mock_data, mock_forward_model, mock_noise):
    return AdditiveGaussianNoiseLikelihood(
        mock_data, mock_forward_model, mock_noise)


def test_initialisation(mock_likelihood, mock_data,
                        mock_forward_model, mock_noise):

    assert mock_likelihood._data == mock_data
    assert mock_likelihood._fwdModel == mock_forward_model
    assert mock_likelihood._noiseModel == mock_noise


@pytest.mark.skip()
def test_memoisation(mock_likelihood):

    parameter = ParameterVector(np.array([0.5, 0.5]))
    logLFirst = mock_likelihood.evaluate_log(parameter)
    logLCached = mock_likelihood.evaluate_log(parameter)

    assert logLFirst == logLCached
    assert mock_likelihood._llCache.contains(parameter)

    assert mock_likelihood._llCache(parameter) == logLFirst


@pytest.mark.skip()
def test_cache_eviction():

    cache = EvaluationCache(2)

    param1 = ParameterVector(np.array([0.1, 0.1]))
    param2 = ParameterVector(np.array([0.2, 0.2]))
    param3 = ParameterVector(np.array([0.3, 0.3]))

    cache.add(param1, 1.0)
    cache.add(param2, 2.0)

    assert cache.contains(param1)
    assert cache.contains(param2)

    cache.add(param3, 3.0)

    assert not cache.contains(param1)
    assert cache.contains(param2)
    assert cache.contains(param3)
    assert cache(param2) == 2.0
    assert cache(param3) == 3.0


@pytest.mark.skip()
def test_stress_test_memoisation(mock_noise, mock_forward_model):

    np.random.seed(19)

    numTests = 10000
    cacheSize = 2
    paramSize = 1000

    mockData = Data(np.array([[0]]))
    likelihood = AdditiveGaussianNoiseLikelihood(
        mockData, mock_forward_model, mock_noise)

    likelihood._llCache = EvaluationCache(cacheSize)

    cacheHits = 0

    for i in range(numTests):

        if (i == 0):

            paramOld = ParameterVector(np.random.rand(paramSize))
            likelihood.evaluate_log(paramOld)

        paramNew = ParameterVector(np.random.rand(paramSize))

        likelihood.evaluate_log(paramNew)

        if likelihood._llCache.contains(paramOld):

            cacheHits += 1

            logLCached = likelihood._llCache(paramOld)
            logL = likelihood.evaluate_log(paramOld)

            assert logLCached == logL

        if likelihood._llCache.contains(paramNew):

            cacheHits += 1

            logLCached = likelihood._llCache(paramNew)
            logL = likelihood.evaluate_log(paramNew)

            assert logLCached == logL

        paramOld = paramNew

    cacheHitPerc = 50. * cacheHits / numTests
    print(
        f"number of cache hits: {cacheHits}. Corresponds to {cacheHitPerc} %")

    cacheTOL = 1e-3
    assert cacheHits <= 2. * (1. + cacheTOL) * numTests
