import numpy as np

from yagremcmc.chain.interface import ProposalMethodInterface
from yagremcmc.chain.metropolisHastings import MetropolisHastings, UnnormalisedPosterior
from yagremcmc.chain.metropolisedRandomWalk import MRWProposal
from yagremcmc.chain.adaptive import AdaptiveCovarianceMatrix
from yagremcmc.chain.factory import ChainFactory
from yagremcmc.statistics.parameterLaw import Gaussian


class AMAdaptive(ProposalMethodInterface):
    """
    Adaptive Proposals
    """

    def __init__(self, chain, eps, cSteps):

        self.chain_ = chain

        initData = self.chain_.trajectory[-cSteps:]

        nData = len(initData)
        self.offset_ = chain.length - cSteps

        mean = np.mean(initData, axis=0)
        sampCov = np.cov(np.vstack(initData), rowvar=False, bias=False)

        self.cov_ = AdaptiveCovarianceMatrix(mean, sampCov, eps, nData)

        self.state_ = None
        self.proposalLaw_ = None

    @property
    def state(self):
        return self.state_

    @state.setter
    def state(self, newState):

        self.state_ = newState
        self.proposalLaw_ = Gaussian(self.state_, self.cov_)

    def generate_proposal(self):

        self._update_covariance()

        return self.proposalLaw_.generate_realisation()

    def _update_covariance(self):

        if self.cov_.nData == self.chain_.length:
            return

        if self.chain_.length - self.offset_ - self.cov_.nData > 1:

            raise RuntimeError("adaptive covariance is lagging behind more than"
                               " two states.")

        self.cov_.update(self.chain_.trajectory[-1])


class AdaptiveMRWProposal(ProposalMethodInterface):
    """
    Adaptive Metropolis Random Walk Proposal Method.
    """

    def __init__(self, initCov, idleSteps, collectionSteps, regParam):
        """
        Parameters:
        - initCov: Initial covariance matrix to be used during the IDLE and COLLECTION phases.
        - idleSteps: Number of steps during which the covariance is not updated (IDLE phase).
        - collectionSteps: Number of steps where samples are collected but the covariance is not updated (COLLECTION phase).
        - regParam: Regularization parameter used for adaptive covariance calculation.
        """

        if initCov.dimension == 1:
            raise NotImplementedError("AM not implemented for scalar chains.")

        self.proposalMethod_ = MRWProposal(initCov)

        self.iSteps_ = idleSteps
        self.cSteps_ = collectionSteps
        self.eps_ = regParam

        self.chain_ = None

    @property
    def state(self):
        return self.proposalMethod_.state

    @state.setter
    def state(self, newState):
        self.proposalMethod_.state = newState

    @property
    def chain(self):
        return self.chain_

    @chain.setter
    def chain(self, newChain):
        self.chain_ = newChain

    def _determine_proposal_method(self):

        if self.chain_.length < self.iSteps_ + self.cSteps_:
            assert isinstance(self.proposalMethod_, MRWProposal)

        elif self.chain_.length == self.iSteps_ + self.cSteps_:

            currentState = self.proposalMethod_.state

            self.proposalMethod_ = AMAdaptive(
                self.chain_, self.eps_, self.cSteps_)
            self.proposalMethod_.state = currentState

        elif self.chain_.length > self.iSteps_ + self.cSteps_:
            assert isinstance(self.proposalMethod_, AMAdaptive)

        else:
            raise RuntimeError("Undefined adaptive Metropolis chain state.")

    def generate_proposal(self):

        if self.chain_ is None:
            raise ValueError(
                "Adaptive Proposal is not associated with a chain yet.")

        self._determine_proposal_method()

        return self.proposalMethod_.generate_proposal()


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
        densityRatio = np.exp(self.targetDensity_.evaluate_log(proposal)
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
