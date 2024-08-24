import logging

from numpy import exp
from enum import Enum
from yagremcmc.chain.interface import ProposalMethodInterface
from yagremcmc.chain.metropolisHastings import MetropolisHastings, UnnormalisedPosterior
from yagremcmc.chain.adaptive import AdaptiveCovarianceMatrix
from yagremcmc.chain.factory import ChainFactory
from yagremcmc.statistics.parameterLaw import Gaussian

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class AdaptivityStatus(Enum):
    NONE = -1
    IDLE = 0
    COLLECTION = 1
    ADAPTIVE = 2

# TODO: implement State pattern


class AdaptiveMRWProposal(ProposalMethodInterface):
    """
    Adaptive Metropolis Random Walk Proposal Method.

    Parameters:
    - initCov: Initial covariance matrix to be used during the IDLE and COLLECTION phases.
    - idleSteps: Number of steps during which the covariance is not updated (IDLE phase).
    - collectionSteps: Number of steps where samples are collected but the covariance is not updated (COLLECTION phase).
    - regParam: Regularization parameter used for adaptive covariance calculation.
    """

    def __init__(self, initCov, idleSteps, collectionSteps, regParam):

        if initCov.dimension == 1:
            raise NotImplementedError("AM not implemented for scalar chains.")

        self.initCov_ = initCov
        self.adaptCov_ = AdaptiveCovarianceMatrix(initCov.dimension, regParam)
        self.idleSteps_ = idleSteps
        self.collectionSteps_ = collectionSteps
        self.status_ = AdaptivityStatus.IDLE
        self.lastChainLength_ = 0

        self.chain_ = None
        self.state_ = None
        self.proposalLaw_ = None

    @property
    def state(self):
        return self.state_

    @state.setter
    def state(self, newState):

        self.state_ = newState
        self._determine_status()
        self._update_proposal_law()

    @property
    def chain(self):
        return self.chain_

    @chain.setter
    def chain(self, newChain):
        self.chain_ = newChain

    def generate_proposal(self):

        if self.state_ is None or self.proposalLaw_ is None:
            raise ValueError(
                "Trying to generate proposal with undefined state.")

        if self.chain_ is None:
            raise ValueError(
                "Adaptive Proposal is not associated with a chain.")

        return self.proposalLaw_.generate_realisation()

    def _update_covariance(self):

        if self.status_ == AdaptivityStatus.ADAPTIVE:
            if self.lastChainLength_ == 0:

                logger.info(
                    f"Initialising adaptive covariance at {self.chain_.length} iterations.")
                self.adaptCov_.initialise(self.collectionSteps_, self.chain_)

            elif self.lastChainLength_ < self.chain_.length:

                assert self.lastChainLength_ == self.chain_.length - 1
                self.adaptCov_.update(self.chain_.trajectory[-1])

            self.lastChainLength_ = self.chain_.length

    def _update_proposal_law(self):

        if self.status_ in (AdaptivityStatus.IDLE, AdaptivityStatus.COLLECTION):
            self.proposalLaw_ = Gaussian(self.state_, self.initCov_)
        elif self.status_ == AdaptivityStatus.ADAPTIVE:
            self._update_covariance()
            self.proposalLaw_ = Gaussian(self.state_, self.adaptCov_)
        else:
            raise ValueError("Invalid status when setting state.")

    def _determine_status(self):

        chain_length = self.chain_.length
        if chain_length < self.idleSteps_:
            self.status_ = AdaptivityStatus.IDLE
        elif self.idleSteps_ <= chain_length < self.idleSteps_ + self.collectionSteps_:
            self.status_ = AdaptivityStatus.COLLECTION
            if self.lastChainLength_ == self.idleSteps_:
                logger.info(
                    f"Start collecting samples for adaptive covariance at {chain_length} iterations.")
        else:
            self.status_ = AdaptivityStatus.ADAPTIVE


class AdaptiveMetropolis(MetropolisHastings):
    """
    Metropolis-Hastings algorithm with adaptive proposal distribution.

    Parameters:
    - targetDensity: The target density to sample from.
    - initCov: Initial covariance matrix.
    - idleSteps: Number of steps during which the covariance is not updated.
    - collectionSteps: Number of steps where samples are collected without updating the covariance.
    - regParam: Regularization parameter used for adaptive covariance calculation.
    """

    def __init__(self, targetDensity, initCov, idleSteps, collectionSteps, regParam):

        proposalMethod = AdaptiveMRWProposal(
            initCov, idleSteps, collectionSteps, regParam)
        super().__init__(targetDensity, proposalMethod)
        self.proposalMethod_.chain = self.chain_

    def _acceptance_probability(self, proposal, state):
        densityRatio = exp(self.targetDensity_.evaluate_log(proposal)
                           - self.targetDensity_.evaluate_log(state))
        return min(densityRatio, 1.)


class AMFactory(ChainFactory):
    """
    Factory for creating Adaptive Metropolis-Hastings (AM) chains.

    Attributes:
    - idleSteps_: Number of steps during which the covariance is not updated.
    - collectionSteps_: Number of steps where samples are collected without updating the covariance.
    - regularisationParameter_: Regularization parameter for covariance adaptation.
    - initialCovariance_: Initial covariance matrix.
    """

    def __init__(self):
        super().__init__()
        self.idleSteps_ = None
        self.collectionSteps_ = None
        self.regularisationParameter_ = None
        self.initialCovariance_ = None

    @property
    def idleSteps(self):
        return self.idleSteps_

    @idleSteps.setter
    def idleSteps(self, iSteps):
        self.idleSteps_ = iSteps

    @property
    def collectionSteps(self):
        return self.collectionSteps_

    @collectionSteps.setter
    def collectionSteps(self, cSteps):
        self.collectionSteps_ = cSteps

    @property
    def regularisationParameter(self):
        return self.regularisationParameter_

    @regularisationParameter.setter
    def regularisationParameter(self, eps):
        if eps < 0:
            raise ValueError("Regularisation parameter must be non-negative.")
        self.regularisationParameter_ = eps

    @property
    def initialCovariance(self):
        return self.initialCovariance_

    @initialCovariance.setter
    def initialCovariance(self, cov):
        self.initialCovariance_ = cov

    def build_from_model(self) -> MetropolisHastings:

        self._validate_parameters()
        targetDensity = UnnormalisedPosterior(self.bayesModel)
        return AdaptiveMetropolis(targetDensity, self.initialCovariance_, self.idleSteps_, self.collectionSteps_, self.regularisationParameter_)

    def build_from_target(self) -> MetropolisHastings:

        self._validate_parameters()
        return AdaptiveMetropolis(self.explicitTarget, self.initialCovariance_, self.idleSteps_, self.collectionSteps_, self.regularisationParameter_)

    def _validate_parameters(self) -> None:

        if self.idleSteps_ is None:
            raise ValueError("Number of idle steps not set in AM.")
        if self.collectionSteps_ is None:
            raise ValueError("Number of collection steps not set in AM.")
        if self.regularisationParameter_ is None:
            raise ValueError("Regularisation parameter not set in AM.")
        if self.initialCovariance_ is None:
            raise ValueError("Initial covariance not set in AM.")
