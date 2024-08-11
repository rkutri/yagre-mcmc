import pytest
from unittest.mock import Mock, create_autospec
from yagremcmc.chain.metropolisedRandomWalk import MetropolisedRandomWalk, MRWFactory
from yagremcmc.chain.preconditionedCrankNicolson import PreconditionedCrankNicolson, PCNFactory


def test_MRWFactory():

    # Mock the necessary components
    mockPosterior = Mock()
    mockProposalCov = Mock()
    mockChain = create_autospec(MetropolisedRandomWalk, instance=True)

    # MRWFactory setup
    chainFactory = MRWFactory()
    chainFactory.set_proposal_covariance(mockProposalCov)
    chainFactory.set_bayes_model(mockPosterior)

    # Mock the MetropolisedRandomWalk construction
    with pytest.raises(ValueError, match="Proposal Covariance not set for MRW"):
        chainFactory.set_proposal_covariance(None)
        chainFactory.build_chain()

    chainFactory.set_proposal_covariance(mockProposalCov)
    chainFactory.build_from_model = Mock(return_value=mockChain)

    # Build chain
    chain = chainFactory.build_chain()

    # Assertions
    assert chain == mockChain
    chainFactory.build_from_model.assert_called_once()


def test_PCNFactory():

    # Mock the necessary components
    mockLikelihood = Mock()
    mockPrior = Mock()
    mockChain = create_autospec(PreconditionedCrankNicolson, instance=True)

    # PCNFactory setup
    chainFactory = PCNFactory()
    chainFactory.set_step_size(0.01)
    chainFactory.set_bayes_model(
        Mock(likelihood=mockLikelihood, prior=mockPrior))

    # Mock the PreconditionedCrankNicolson construction
    chainFactory.build_from_model = Mock(return_value=mockChain)

    # Build chain
    chain = chainFactory.build_chain()

    # Assertions
    assert chain == mockChain
    chainFactory.build_from_model.assert_called_once()


def test_PCNFactory_error():

    # Initialize PCNFactory with a mock
    chainFactory = PCNFactory()
    chainFactory.set_step_size(0.01)

    # Set an explicit target (not allowed for PCNFactory)
    chainFactory.set_explicit_target(Mock())

    # Ensure RuntimeError is raised
    with pytest.raises(RuntimeError):
        chainFactory.build_chain()
