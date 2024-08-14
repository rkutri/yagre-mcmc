from numpy import exp
from enum import Enum
from yagremcmc.chain.interface import ProposalMethodInterface
from yagremcmc.chain.metropolisHastings import MetropolisHastings
from yagremcmc.chain.adaptive import AdaptiveCovarianceMatrix
from yagremcmc.chain.factory import ChainFactory
from yagremcmc.statistics.parameterLaw import Gaussian


class AdaptivityStatus(Enum):

    NONE = -1
    BURNIN = 0
    COLLECTION = 1
    ADAPTIVE = 2


# TODO implement the state design pattern, depending on status
class AdaptiveMRWProposal(ProposalMethodInterface):
    """
    burnInSteps: are ignored
    collectionSteps: samples are collected but the covariance not yet 
                     updated
    """

    def __init__(self, initCov, burnInSteps, collectionSteps, regParam):

        self.initCov_ = initCov
        self.adaptCov_ = AdaptiveCovarianceMatrix(initCov.dimension, regParam)
        self.bSteps_ = burnInSteps
        self.cSteps_ = collectionSteps
        self.status_ = AdaptivityStatus.BURNIN
        self.lastChainLength_ = 0

        self.chain_ = None
        self.state_ = None
        self.proposalLaw = None

    @property
    def state(self):
        return self.state_

    @state.setter
    def state(self, newState):

        self._determine_status()

        if self.status_ == AdaptivityStatus.BURNIN \
                or self.status_ == AdaptivityStatus.COLLECTION:

            self.proposalLaw_ = Gaussian(self.state_, self.initCov_)

        elif self.status_ == AdaptivityStatus.ADAPTIVE:

            self.proposalLaw_ = Gaussian(self.state_, self.adaptCov_)

        else:
            raise ValueError("Invalid status when setting state.")

        self.state_ = newState

    @property
    def chain(self):
        return self.chain_

    @property
    def chain(self, newChain):
        self.chain_ = newChain

    def generate_proposal(self):

        if self.state_ is None or self.proposalLaw_ is None:
            raise ValueError(
                "Trying to generate proposal with undefined state.")

        if self.chain_ is None:
            raise ValueError("Adaptive Proposal is not associated to a chain.")

        if self.status_ == AdaptivityStatus.COLLECTION \
                or self.status_ == AdaptivityStatus.ADAPTIVE:
            self._update_covariance()

        return self.proposalLaw_.generate_realisation()

    def _update_covariance(self):

        if self.lastChainLength_ < self.chain_.length:
            self.adaptCov_.update(self.chain_[-1])

        # force a copy
        self.lastChainLength_ = int(self.chain_.length)

    def _determine_status(self):

        if self.chain_.length_ < self.bSteps_:
            self.status_ = AdaptivityStatus.BURNIN
        elif self.chain_.length < self.cSteps_:
            self.status_ = AdaptivityStatus.COLLECTION
        else:
            self.status_ = AdaptivityStatus.ADAPTIVE


class AdaptiveMetropolis(MetropolisHastings):

    def __init__(self, targetDensity, initCov, burninSteps, collectionSteps, regParam):

        proposalMethod = AdaptiveMRWProposal(
            initCov, burninSteps, collectionSteps, regParam)

        super().__init__(targetDensity, proposalMethod)

        self.proposalMethod_.chain = self.chain_

    def _acceptance_probability(self, proposal, state):

        densityRatio = exp(self.targetDensity_.evaluate_log(proposal)
                           - self.targetDensity_.evaluate_log(state))

        return densityRatio if densityRatio < 1. else 1.


# TODO: make the hyperparameters properties instead of using explicit setters
class AMFactory(ChainFactory):
    pass
