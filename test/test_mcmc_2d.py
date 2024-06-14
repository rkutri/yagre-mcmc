import numpy as np

from test.testSetup import GaussianTargetDensity2d
from inference.metropolisedRandomWalk import MetropolisedRandomWalk
from parameter.vector import ParameterVector


tgtMean = ParameterVector.from_coefficient(np.array([1., 1.5]))
tgtCov = np.array(
    [[1.2, -0.2],
     [-0.2, 0.4]])
tgtDensity = GaussianTargetDensity2d(tgtMean, tgtCov)

proposalVariance = 0.25
mcmc = MetropolisedRandomWalk(tgtDensity, proposalVariance)

nSteps = 10000
initState = ParameterVector.from_coefficient(np.array([-2., -0.]))
mcmc.run(nSteps, initState, verbose=False)

states = np.array(mcmc.chain)

# postprocessing
burnin = 50
thinningStep = 4

mcmcSamples = states[burnin::thinningStep]


def test_moments():

    meanState = np.mean(states, axis=0)
    meanEst = np.mean(mcmcSamples, axis=0)

    stateCov = np.cov(states, rowvar=False)
    sampleCov = np.cov(mcmcSamples, rowvar=False)

    MTOL = 2e-1
    CTOL = 4e-1

    assert np.allclose(meanState, tgtMean.coefficient, atol=MTOL)
    assert np.allclose(meanEst, tgtMean.coefficient, atol=MTOL)

    assert np.allclose(sampleCov, tgtCov, atol=CTOL)
    assert np.allclose(stateCov, tgtCov, atol=2. * CTOL)
