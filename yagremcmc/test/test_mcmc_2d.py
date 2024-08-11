import pytest
import numpy as np

from numpy.random import seed
from yagremcmc.test.testSetup import GaussianTargetDensity2d
from yagremcmc.statistics.covariance import IIDCovarianceMatrix, DiagonalCovarianceMatrix
from yagremcmc.chain.metropolisedRandomWalk import MetropolisedRandomWalk
from yagremcmc.parameter.vector import ParameterVector


@pytest.mark.parametrize("mcmcProposal", ["iid", "indep"])
def test_moments(mcmcProposal):

    seed(116)

    # Target distribution setup
    tgtMean = ParameterVector.from_coefficient(np.array([1., 1.5]))
    tgtCov = np.array(
        [[1.2, -0.2],
         [-0.2, 0.4]]
    )
    tgtDensity = GaussianTargetDensity2d(tgtMean, tgtCov)

    # Proposal distribution setup based on mcmcProposal
    if mcmcProposal == 'iid':

        proposalMargVar = 0.25
        proposalCov = IIDCovarianceMatrix(tgtMean.dimension, proposalMargVar)

    elif mcmcProposal == 'indep':

        proposalMargVar = np.array([tgtCov[0, 0], tgtCov[1, 1]])
        proposalCov = DiagonalCovarianceMatrix(proposalMargVar)

    else:
        raise Exception(f"Proposal {mcmcProposal} not implemented")

    mcmc = MetropolisedRandomWalk(tgtDensity, proposalCov)

    # MCMC run setup
    nSteps = 40000
    initState = ParameterVector.from_coefficient(np.array([-2., 0.]))
    mcmc.run(nSteps, initState, verbose=False)

    states = np.array(mcmc.states)

    # Postprocessing
    burnin = 200
    thinningStep = 5

    mcmcSamples = states[burnin::thinningStep]

    # Moment tests
    meanState = np.mean(states, axis=0)
    meanEst = np.mean(mcmcSamples, axis=0)

    stateCov = np.cov(states, rowvar=False)
    sampleCov = np.cov(mcmcSamples, rowvar=False)

    MTOL = 5e-2
    CTOL = 1e-1

    assert np.allclose(meanState, tgtMean.coefficient, atol=MTOL), \
        f"Mean state does not match target mean with mcmcProposal='{mcmcProposal}'"
    assert np.allclose(meanEst, tgtMean.coefficient, atol=MTOL), \
        f"Mean estimate does not match target mean with mcmcProposal='{mcmcProposal}'"

    assert np.allclose(sampleCov, tgtCov, atol=CTOL), \
        f"Sample covariance does not match target covariance with mcmcProposal='{mcmcProposal}'"
    assert np.allclose(stateCov, tgtCov, atol=2. * CTOL), \
        f"State covariance does not match target covariance with mcmcProposal='{mcmcProposal}'"


if __name__ == "__main__":
    pytest.main()
