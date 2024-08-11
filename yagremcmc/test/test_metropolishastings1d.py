import pytest

import numpy as np

from numpy.random import seed
from yagremcmc.test.testSetup import GaussianTargetDensity1d
from yagremcmc.statistics.covariance import IIDCovarianceMatrix
from yagremcmc.chain.metropolisedRandomWalk import MetropolisedRandomWalk
from yagremcmc.parameter.scalar import ScalarParameter


def test_metropolishastings_initialisation():

    tgtMean = ScalarParameter.from_coefficient(np.array([1.]))
    tgtVar = 1.
    tgtDensity = GaussianTargetDensity1d(tgtMean, tgtVar)

    proposalVariance = 0.5
    proposalCov = IIDCovarianceMatrix(1, proposalVariance)

    mc = MetropolisedRandomWalk(tgtDensity, proposalCov)

    assert isinstance(mc.targetDensity_, type(tgtDensity))
    assert mc.chain == []


def test_generate_proposal():

    tgtMean = ScalarParameter.from_coefficient(np.array([-1.5]))
    tgtVar = 0.5
    tgtDensity = GaussianTargetDensity1d(tgtMean, tgtVar)

    proposalVariance = 0.5
    proposalCov = IIDCovarianceMatrix(1, proposalVariance)

    mc = MetropolisedRandomWalk(tgtDensity, proposalCov)

    state = ScalarParameter.from_coefficient(np.array([-3.]))
    proposal = mc.generate_proposal__(state)

    assert isinstance(proposal, type(state))


def test_accept_reject():

    tgtMean = ScalarParameter.from_coefficient(np.array([0.]))
    tgtVar = 1.
    tgtDensity = GaussianTargetDensity1d(tgtMean, tgtVar)

    proposalVariance = 0.5
    proposalCov = IIDCovarianceMatrix(1, proposalVariance)

    mc = MetropolisedRandomWalk(tgtDensity, proposalCov)

    state = ScalarParameter.from_value(np.array([2.]))
    proposal = ScalarParameter.from_value(np.array([2.5]))

    acceptedState = mc.accept_reject__(proposal, state)

    assert acceptedState in [proposal, state]


def test_run_chain():

    seed(18)

    tgtMean = ScalarParameter.from_coefficient(np.array([1.5]))
    tgtVar = 1.
    tgtDensity = GaussianTargetDensity1d(tgtMean, tgtVar)

    proposalVariance = 0.5
    proposalCov = IIDCovarianceMatrix(1, proposalVariance)

    mc = MetropolisedRandomWalk(tgtDensity, proposalCov)

    tgtMean = ScalarParameter.from_coefficient(np.array([0.]))
    tgtVar = 1.
    tgtDensity = GaussianTargetDensity1d(tgtMean, tgtVar)

    proposalVariance = 0.5
    proposalCov = IIDCovarianceMatrix(1, proposalVariance)

    mc = MetropolisedRandomWalk(tgtDensity, proposalCov)

    nSteps = 15000
    initState = ScalarParameter.from_coefficient(np.array([-3.]))
    mc.run(nSteps, initState, verbose=False)

    assert len(mc.chain) == nSteps

    states = np.array(mc.chain)

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
