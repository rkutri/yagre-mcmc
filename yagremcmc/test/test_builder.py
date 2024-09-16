import pytest
from unittest.mock import Mock, create_autospec
from yagremcmc.chain.metropolisedRandomWalk import MetropolisedRandomWalk, MRWBuilder
from yagremcmc.chain.preconditionedCrankNicolson import PreconditionedCrankNicolson, PCNBuilder


def test_MRWBuilder():

    # Mock the necessary components
    mockPosterior = Mock()
    mockProposalCov = Mock()
    mockChain = create_autospec(MetropolisedRandomWalk, instance=True)

    # MRWBuilder setup
    chainBuilder = MRWBuilder()
    chainBuilder.proposalCovariance = mockProposalCov
    chainBuilder.bayesModel = mockPosterior

    # Mock the MetropolisedRandomWalk construction
    with pytest.raises(ValueError, match="Proposal Covariance not set for MRW"):
        chainBuilder.proposalCovariance = None
        chainBuilder.build_method()

    chainBuilder.proposalCovariance = mockProposalCov
    chainBuilder.build_from_model = Mock(return_value=mockChain)

    # Build chain
    chain = chainBuilder.build_method()

    # Assertions
    assert chain == mockChain
    chainBuilder.build_from_model.assert_called_once()


def test_PCNBuilder():

    # Mock the necessary components
    mockLikelihood = Mock()
    mockPrior = Mock()
    mockChain = create_autospec(PreconditionedCrankNicolson, instance=True)

    # PCNBuilder setup
    chainBuilder = PCNBuilder()
    chainBuilder.stepSize = 0.01
    chainBuilder.bayesModel = Mock(likelihood=mockLikelihood, prior=mockPrior)

    # Mock the PreconditionedCrankNicolson construction
    chainBuilder.build_from_model = Mock(return_value=mockChain)

    # Build chain
    chain = chainBuilder.build_method()

    # Assertions
    assert chain == mockChain
    chainBuilder.build_from_model.assert_called_once()


def test_PCNBuilder_error():

    # Initialize PCNBuilder with a mock
    chainBuilder = PCNBuilder()
    chainBuilder.stepSize = 0.01

    # Set an explicit target (not allowed for PCNBuilder)
    chainBuilder.explicitTarget = Mock()

    # Ensure RuntimeError is raised
    with pytest.raises(RuntimeError):
        chainBuilder.build_method()
