import pytest

import numpy as np

from yagremcmc.chain.method.mlda import MLDABuilder
from yagremcmc.statistics.covariance import IIDCovarianceMatrix
from yagremcmc.test.testSetup import GaussianTargetDensity2d
from yagremcmc.parameter.vector import ParameterVector
from yagremcmc.postprocessing import autocorrelation as ac


@pytest.fixture
def setup_mlda_test_data():
    """
    Fixture to provide shared setup data for MLDA tests.
    """
    tgtMean = np.array([1.0, 1.5])
    tgtCov = np.array([[2.4, -0.5], [-0.5, 0.7]])

    baseSurrMean = tgtMean + np.array([-0.05, 0.01])
    fineSurrMean = tgtMean + np.array([0.0, -0.01])
    baseSurrCov = 3.0 * np.array([[2.8, -0.1], [-0.1, 1.7]])
    fineSurrCov = 1.5 * np.array([[2.4, -0.3], [-0.3, 1.1]])

    tgtDensity = GaussianTargetDensity2d(ParameterVector(tgtMean), tgtCov)
    baseSurrDensity = GaussianTargetDensity2d(
        ParameterVector(baseSurrMean), baseSurrCov
    )
    fineSurrDensity = GaussianTargetDensity2d(
        ParameterVector(fineSurrMean), fineSurrCov
    )

    basePropCov = IIDCovarianceMatrix(len(tgtMean), 1.0)

    return {
        "tgtDensity": tgtDensity,
        "baseSurrDensity": baseSurrDensity,
        "fineSurrDensity": fineSurrDensity,
        "basePropCov": basePropCov,
        "tgtMean": tgtMean,
    }


@pytest.fixture
def mlda_chain_builder(setup_mlda_test_data):
    """
    Fixture to build the MLDA chain with the shared setup data.
    """
    data = setup_mlda_test_data
    chainBuilder = MLDABuilder()
    chainBuilder.explicitTarget = data["tgtDensity"]
    chainBuilder.surrogateTargets = [data["baseSurrDensity"], data["fineSurrDensity"]]
    chainBuilder.baseProposalCovariance = data["basePropCov"]
    chainBuilder.subChainLengths = [6, 6]
    return chainBuilder


def test_mlda_chain(setup_mlda_test_data, mlda_chain_builder):

    np.random.seed(42)

    # Build the MLDA method
    chainBuilder = mlda_chain_builder
    mcmc = chainBuilder.build_method()

    # Run the chain
    nChain = 20000
    initState = ParameterVector(np.array([-8.0, -7.0]))
    mcmc.run(nChain, initState, verbose=False)

    # Extract results
    states = np.array(mcmc.chain.trajectory)
    diagnostics = mcmc.chain.diagnostics

    # Assertions
    assert len(states) == nChain, "Chain length mismatch with expected steps."
    assert 0.1 < diagnostics.global_acceptance_rate() < 0.9, \
        "Pathological acceptance rate for given parameters."

    burnin = 500
    fixedThinning = 5

    # Test convergence of mean estimate
    mean_estimate = np.mean(states[burnin::fixedThinning], axis=0)
    tgtMean = setup_mlda_test_data["tgtMean"]
    assert np.allclose(mean_estimate, tgtMean, atol=0.1), \
        f"Mean estimate {mean_estimate} deviates from target mean {tgtMean}."


def test_mlda_perfect_surrogate(setup_mlda_test_data):
    """
    Test MLDA chain where the base and fine surrogates match the target measure.
    Expect an acceptance rate of 1.
    """
    # Set seed for reproducibility
    np.random.seed(123)

    # Extract shared setup data
    data = setup_mlda_test_data

    # Surrogates are set to the exact target
    tgtDensity = data["tgtDensity"]
    baseSurrDensity = tgtDensity
    fineSurrDensity = tgtDensity
    basePropCov = data["basePropCov"]

    # Build MLDA chain
    chainBuilder = MLDABuilder()
    chainBuilder.explicitTarget = tgtDensity
    chainBuilder.surrogateTargets = [baseSurrDensity, fineSurrDensity]
    chainBuilder.baseProposalCovariance = basePropCov
    chainBuilder.subChainLengths = [6, 50]

    mcmc = chainBuilder.build_method()

    # Run the chain
    nChain = 5000
    initState = ParameterVector(np.array([-8.0, -7.0]))
    mcmc.run(nChain, initState, verbose=False)

    # Extract diagnostics
    diagnostics = mcmc.chain.diagnostics

    # Assertions
    assert np.abs(diagnostics.global_acceptance_rate() - 1.) < 1e-3, \
        "Proposals are rejected although the surrogates match the target."


def test_mlda_two_level(setup_mlda_test_data):

    np.random.seed(456)

    # Extract shared setup data
    data = setup_mlda_test_data
    tgtDensity = data["tgtDensity"]

    # Create a single surrogate density
    surrogateMean = data["tgtMean"] + np.array([0.1, -0.2])
    surrogateCov = 1.5 * np.array([[2.5, -0.3], [-0.3, 0.9]])
    surrogateDensity = GaussianTargetDensity2d(ParameterVector(surrogateMean), surrogateCov)

    basePropCov = data["basePropCov"]

    # Build two-level MLDA chain
    chainBuilder = MLDABuilder()
    chainBuilder.explicitTarget = tgtDensity
    chainBuilder.surrogateTargets = [surrogateDensity]
    chainBuilder.baseProposalCovariance = basePropCov
    chainBuilder.subChainLengths = [10]

    mcmc = chainBuilder.build_method()

    # Run the chain
    nChain = 10000
    initState = ParameterVector(np.array([-5.0, 4.0]))
    mcmc.run(nChain, initState, verbose=False)

    # Extract diagnostics
    diagnostics = mcmc.chain.diagnostics
    acceptance_rate = diagnostics.global_acceptance_rate()

    # Postprocessing
    states = np.array(mcmc.chain.trajectory)
    burnin = 500
    thinningStep = 5
    mcmcSamples = states[burnin::thinningStep]

    meanEst = np.mean(mcmcSamples, axis=0)

    # Assertions
    assert len(mcmc.chain.trajectory) == nChain, "Chain length mismatch for two-level method."

    assert 0.1 < acceptance_rate < 0.9, \
        f"Acceptance rate {acceptance_rate} for two-level method is outside expected range."

    np.testing.assert_allclose(
        meanEst, data["tgtMean"], atol=0.1, 
        err_msg="Estimated mean from two-level method deviates significantly from target mean."
    )


def test_mlda_five_level_method(setup_mlda_test_data):
    """
    Test a five-level MLDA method with reasonable surrogate densities.
    """
    # Set seed for reproducibility
    np.random.seed(789)

    # Extract shared setup data
    data = setup_mlda_test_data
    tgtDensity = data["tgtDensity"]

    # Create surrogate densities for five levels
    surrogateMeans = [
        data["tgtMean"] + np.array([0.2, -0.3]),
        data["tgtMean"] + np.array([-0.1, 0.1]),
        data["tgtMean"] + np.array([0.05, -0.05]),
        data["tgtMean"] + np.array([0.0, 0.0]),
    ]
    surrogateCovs = [
        8. * np.array([[3.0, -0.2], [-0.2, 1.5]]),
        6. * np.array([[2.7, -0.25], [-0.25, 1.3]]),
        4. * np.array([[2.5, -0.3], [-0.3, 1.1]]),
        2. * np.array([[2.4, -0.35], [-0.35, 1.0]]),
    ]
    surrogateDensities = [
        GaussianTargetDensity2d(ParameterVector(mean), cov)
        for mean, cov in zip(surrogateMeans, surrogateCovs)
    ]

    basePropCov = data["basePropCov"]

    # Build five-level MLDA chain
    chainBuilder = MLDABuilder()
    chainBuilder.explicitTarget = tgtDensity
    chainBuilder.surrogateTargets = surrogateDensities
    chainBuilder.baseProposalCovariance = basePropCov
    chainBuilder.subChainLengths = [6, 4, 4, 3]

    mcmc = chainBuilder.build_method()

    # Run the chain
    nChain = 5000
    initState = ParameterVector(np.array([2.0, -3.0]))
    mcmc.run(nChain, initState, verbose=False)

    # Extract diagnostics
    diagnostics = mcmc.chain.diagnostics
    acceptance_rate = diagnostics.global_acceptance_rate()

    # Postprocessing
    states = np.array(mcmc.chain.trajectory)
    burnin = 200
    thinningStep = 3
    mcmcSamples = states[burnin::thinningStep]

    meanState = np.mean(states, axis=0)
    meanEst = np.mean(mcmcSamples, axis=0)

    # Assertions
    assert len(mcmc.chain.trajectory) == nChain, "Chain length mismatch for five-level method."
    assert 0.1 < acceptance_rate < 0.9, \
        f"Acceptance rate {acceptance_rate} for five-level method is outside expected range."
    np.testing.assert_allclose(
        meanEst, data["tgtMean"], atol=0.2, 
        err_msg="Estimated mean from five-level method deviates significantly from target mean."
    )

