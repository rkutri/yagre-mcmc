import pytest

import numpy as np

from numpy.random import seed
from yagremcmc.test.testSetup import GaussianTargetDensity1d
from yagremcmc.statistics.covariance import IIDCovarianceMatrix
from yagremcmc.chain.method.mrw import MetropolisedRandomWalk
from yagremcmc.chain.diagnostics import DummyDiagnostics, AcceptanceRateDiagnostics
from yagremcmc.parameter.scalar import ScalarParameter


@pytest.mark.parametrize("Diagnostics", [DummyDiagnostics, AcceptanceRateDiagnostics])
def test_metropolishastings_initialisation(Diagnostics):

    tgtMean = ScalarParameter.from_coefficient(np.array([1.]))
    tgtVar = 1.
    tgtDensity = GaussianTargetDensity1d(tgtMean, tgtVar)

    proposalVariance = 0.5
    proposalCov = IIDCovarianceMatrix(1, proposalVariance)

    diagnostics = Diagnostics()

    mc = MetropolisedRandomWalk(tgtDensity, proposalCov, diagnostics)

    assert isinstance(mc.target, type(tgtDensity))
    assert mc.chain.trajectory == []


@pytest.mark.parametrize("Diagnostics", [DummyDiagnostics, AcceptanceRateDiagnostics])
def test_accept_reject(Diagnostics):

    tgtMean = ScalarParameter.from_coefficient(np.array([0.]))
    tgtVar = 1.
    tgtDensity = GaussianTargetDensity1d(tgtMean, tgtVar)

    proposalVariance = 0.5
    proposalCov = IIDCovarianceMatrix(1, proposalVariance)

    diagnostics = Diagnostics()

    mc = MetropolisedRandomWalk(tgtDensity, proposalCov, diagnostics)

    state = ScalarParameter.from_value(np.array([2.]))
    proposal = ScalarParameter.from_value(np.array([2.5]))

    transitionOutcome = mc._accept_reject(proposal, state)

    assert transitionOutcome.state in [proposal, state]

@pytest.mark.parametrize("Diagnostics", [DummyDiagnostics, AcceptanceRateDiagnostics])
def test_run_chain(Diagnostics):

    seed(18)

    tgtMean = ScalarParameter.from_coefficient(np.array([1.5]))
    tgtVar = 1.
    tgtDensity = GaussianTargetDensity1d(tgtMean, tgtVar)

    proposalVariance = 0.5
    proposalCov = IIDCovarianceMatrix(1, proposalVariance)

    diagnostics = Diagnostics()

    mc = MetropolisedRandomWalk(tgtDensity, proposalCov, diagnostics)

    tgtMean = ScalarParameter.from_coefficient(np.array([0.]))
    tgtVar = 1.
    tgtDensity = GaussianTargetDensity1d(tgtMean, tgtVar)

    proposalVariance = 0.5
    proposalCov = IIDCovarianceMatrix(1, proposalVariance)

    mc = MetropolisedRandomWalk(tgtDensity, proposalCov, diagnostics)

    nSteps = 15000
    initState = ScalarParameter.from_coefficient(np.array([-3.]))
    mc.run(nSteps, initState, verbose=False)

    assert len(mc.chain.trajectory) == nSteps

    states = np.array(mc.chain.trajectory)

    # postprocessing
    burnin = 200
    thinningStep = 6

    mcSamples = states[burnin::thinningStep]

    # estimate mean
    meanSample = np.mean(mcSamples)
    meanState = np.mean(states)

    # estimate variance
    sampleVar = np.var(mcSamples)
    stateVar = np.var(states)

    # test moments
    MTOL = 1e-1
    VTOL = 1e-1

    assert np.abs(tgtMean.coefficient - meanSample) < MTOL
    assert np.abs(tgtMean.coefficient - meanState) < 2. * MTOL

    assert np.abs(tgtVar - sampleVar) < VTOL
    assert np.abs(tgtVar - stateVar) < VTOL


if __name__ == "__main__":
    pytest.main()
