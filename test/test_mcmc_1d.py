import numpy as np

from test.testSetup import GaussianTargetDensity1d
from inference.metropolisedRandomWalk import MetropolisedRandomWalk
from parameter.scalar import ScalarParameter


tgtMean = ScalarParameter(np.array([1.5]))
tgtVar = 1.
tgtDensity = GaussianTargetDensity1d(tgtMean, tgtVar)

mesh = np.linspace(-4., 4., 200, endpoint=True)

# evaluate target density and normalise
tgtDensityEval = tgtDensity.evaluate_on_mesh(mesh)
tgtDensityEval /= np.sqrt(2. * np.pi * tgtVar)


proposalVariance = 0.5
mcmc = MetropolisedRandomWalk(tgtDensity, proposalVariance)

nSteps = 10000
initState = ScalarParameter(np.array([-3.]))
mcmc.run(nSteps, initState, verbose=False)

states = np.array(mcmc.chain)

# postprocessing
burnin = int(0.02 * nSteps)
thinningStep = 4

mcmcSamples = states[burnin::thinningStep]

# estimate mean
meanSample = np.mean(mcmcSamples)
meanState = np.mean(states)

# estimate variance
sampleVar = np.var(mcmcSamples)
stateVar = np.var(states)


def test_moments():

    MTOL = 1e-1
    VTOL = 1e-1

    assert np.abs(tgtMean.coefficient - meanSample) < MTOL
    assert np.abs(tgtMean.coefficient - meanState) < MTOL

    assert np.abs(tgtVar - sampleVar) < VTOL
    assert np.abs(tgtVar - stateVar) < VTOL
