import pytest
import numpy as np

from numpy.random import seed
from yagremcmc.parameter.vector import ParameterVector
from yagremcmc.chain.method.deprecated.am import AMBuilder
from yagremcmc.statistics.covariance import IIDCovarianceMatrix
from yagremcmc.test.testSetup import GaussianTargetDensity2d


@pytest.fixture
def setup_am():

    seed(19)

    tgtMean = ParameterVector(np.array([1., 1.5]))
    tgtCov = np.array([[3.2, -0.4], [-0.4, 0.2]])
    tgtDensity = GaussianTargetDensity2d(tgtMean, tgtCov)

    proposalCov = IIDCovarianceMatrix(tgtMean.dimension, 0.25)

    chainBuilder = AMBuilder()
    chainBuilder.explicitTarget = tgtDensity
    chainBuilder.idleSteps = 100
    chainBuilder.collectionSteps = 500
    chainBuilder.regularisationParameter = 1e-4
    chainBuilder.initialCovariance = proposalCov

    mcmc = chainBuilder.build_method()

    return mcmc, tgtMean.coefficient, tgtCov


@pytest.mark.skip(reason="Use of adaptive proposals is deprecated for now.")
def test_mean_estimation(setup_am):

    seed(20)

    mcmc, trueMean, _ = setup_am

    nSteps = 50000
    initState = ParameterVector(np.array([0., -1.]))
    mcmc.run(nSteps, initState)

    burnIn = int(0.01 * nSteps)
    thinningStep = 8
    states = np.array(mcmc.chain.trajectory)[burnIn::thinningStep]

    meanEst = np.mean(states, axis=0)

    assert np.allclose(meanEst, trueMean, atol=0.03), f"Estimated mean {
        meanEst} differs from true mean {trueMean}"


@pytest.mark.skip(reason="Use of adaptive proposals is deprecated for now.")
def test_covariance_estimation(setup_am):

    mcmc, _, trueCov = setup_am

    seed(21)

    nSteps = 80000
    initState = ParameterVector(np.array([0., -1.]))
    mcmc.run(nSteps, initState)

    burnIn = int(0.01 * nSteps)
    thinningStep = 8
    states = np.array(mcmc.chain.trajectory)[burnIn::thinningStep]

    covEst = np.cov(states, rowvar=False)

    assert np.allclose(covEst, trueCov, atol=0.05), f"Estimated covariance\n{
        covEst}\ndiffers from true covariance\n{trueCov}"


@pytest.mark.skip(reason="Use of adaptive proposals is deprecated for now.")
def test_acceptance_rate(setup_am):

    seed(22)

    mcmc, trueMean, _ = setup_am

    nSteps = 10000
    initState = ParameterVector(trueMean)
    mcmc.run(nSteps, initState)

    acceptanceRate = mcmc.chain.diagnostics.global_acceptance_rate()

    assert 0.1 <= acceptanceRate <= 0.8, f"Acceptance rate {
        acceptanceRate} is out of expected range"
