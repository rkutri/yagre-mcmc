import logging

import numpy as np

from yagremcmc.chain.proposal import ProposalMethod
from yagremcmc.chain.metropolisHastings import MetropolisHastings, UnnormalisedPosterior
from yagremcmc.chain.method.mrw import MRWProposal
from yagremcmc.chain.adaptive import AMCovarianceMatrix
from yagremcmc.chain.builder import ChainBuilder
from yagremcmc.statistics.parameterLaw import Gaussian


amLogger = logging.getLogger(__name__)
amLogger.setLevel(logging.INFO)

consoleHandler = logging.StreamHandler()

formatter = logging.Formatter('%(levelname)s - %(name)s: %(message)s')
consoleHandler.setFormatter(formatter)

amLogger.addHandler(consoleHandler)


class AMProposal(ProposalMethod):
    """
    Adaptive Proposals
    """

    def __init__(self, chain, eps, cSteps):

        super().__init__()

        self._chain = chain

        initData = self._chain.trajectory[-cSteps:]

        nData = len(initData)
        self._offset = chain.length - cSteps

        mean = np.mean(initData, axis=0)
        sampCov = np.cov(np.vstack(initData), rowvar=False, bias=False)

        self._cov = AMCovarianceMatrix(mean, sampCov, eps, nData)

        self._proposalLaw = None

    def set_state(self, newState):

        self._state = newState
        self._proposalLaw = Gaussian(self._state, self._cov)

    def generate_proposal(self):

        self._update_covariance()

        return self._proposalLaw.generate_realisation()

    def _update_covariance(self):

        if self._cov.nData == self._chain.length:
            return

        if self._chain.length - self._offset - self._cov.nData > 1:
            raise RuntimeError("adaptive covariance is lagging behind more "
                               " than two states.")

        self._cov.update(self._chain.trajectory[-1])


class AdaptiveMRWProposal(ProposalMethod):
    """
    Adaptive Metropolis Random Walk Proposal Method.
    """

    def __init__(self, initCov, idleSteps, collectionSteps, regParam):
        """
        Parameters:
        - initCov: Initial covariance matrix to be used during the IDLE and
                     COLLECTION phases.
        - idleSteps: Number of steps during which the covariance is not updated
                     (IDLE phase).
        - collectionSteps: Number of steps where samples are collected but the
                           covariance is not updated (COLLECTION phase).
        - regParam: Regularization parameter used for adaptive covariance
                    calculation.
        """

        if initCov.dimension == 1:
            raise NotImplementedError("AM not implemented for scalar chains.")

        super().__init__()

        self._proposalMethod = MRWProposal(initCov)

        self.iSteps_ = idleSteps
        self.cSteps_ = collectionSteps
        self.eps_ = regParam

        self._chain = None

    def get_state(self):
        return self._proposalMethod.get_state()

    def set_state(self, newState):
        self._proposalMethod.set_state(newState)

    @property
    def chain(self):
        return self._chain

    @chain.setter
    def chain(self, newChain):
        self._chain = newChain

    def _determine_proposal_method(self):

        if self._chain.length < self.iSteps_ + self.cSteps_:
            assert isinstance(self._proposalMethod, MRWProposal)

        elif self._chain.length == self.iSteps_ + self.cSteps_:

            currentState = self._proposalMethod.get_state()

            self._proposalMethod = AMProposal(
                self._chain, self.eps_, self.cSteps_)
            self._proposalMethod.set_state(currentState)

            amLogger.info("Start adaptive covariance")

        elif self._chain.length > self.iSteps_ + self.cSteps_:
            assert isinstance(self._proposalMethod, AMProposal)

        else:
            raise RuntimeError("Undefined adaptive Metropolis chain state.")

    def generate_proposal(self):

        if self._chain is None:
            raise ValueError(
                "Adaptive Proposal is not associated with a chain yet.")

        self._determine_proposal_method()

        return self._proposalMethod.generate_proposal()


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

    def __init__(self, targetDensity, initCov,
                 idleSteps, collectionSteps, regParam):

        proposalMethod = AdaptiveMRWProposal(
            initCov, idleSteps, collectionSteps, regParam)
        super().__init__(targetDensity, proposalMethod)
        self._proposalMethod.chain = self._chain

    def _acceptance_probability(self, proposal, state):
        densityRatio = np.exp(self._tgtDensity.evaluate_log(proposal)
                              - self._tgtDensity.evaluate_log(state))
        return min(densityRatio, 1.)


class AMBuilder(ChainBuilder):
    """
    Builder for creating Adaptive Metropolis-Hastings (AM) chains.

    Attributes:
    - _idleSteps: Number of steps during which the covariance is not updated.
    - _collectionSteps: Number of steps where samples are collected without updating the covariance.
    - _regularisationParameter: Regularization parameter for covariance adaptation.
    - _initialCovariance: Initial covariance matrix.
    """

    def __init__(self):
        super().__init__()
        self._idleSteps = None
        self._collectionSteps = None
        self._regularisationParameter = None
        self._initialCovariance = None

    @property
    def idleSteps(self):
        return self._idleSteps

    @idleSteps.setter
    def idleSteps(self, iSteps):
        self._idleSteps = iSteps

    @property
    def collectionSteps(self):
        return self._collectionSteps

    @collectionSteps.setter
    def collectionSteps(self, cSteps):
        self._collectionSteps = cSteps

    @property
    def regularisationParameter(self):
        return self._regularisationParameter

    @regularisationParameter.setter
    def regularisationParameter(self, eps):
        if eps < 0:
            raise ValueError("Regularisation parameter must be non-negative.")
        self._regularisationParameter = eps

    @property
    def initialCovariance(self):
        return self._initialCovariance

    @initialCovariance.setter
    def initialCovariance(self, cov):
        self._initialCovariance = cov

    def build_from_model(self) -> MetropolisHastings:

        self._validate_parameters()
        targetDensity = UnnormalisedPosterior(self._bayesModel)
        return AdaptiveMetropolis(targetDensity, self._initialCovariance,
                                  self._idleSteps, self._collectionSteps, self._regularisationParameter)

    def build_from_target(self) -> MetropolisHastings:

        self._validate_parameters()
        return AdaptiveMetropolis(self._explicitTarget, self._initialCovariance,
                                  self._idleSteps, self._collectionSteps, self._regularisationParameter)

    def _validate_parameters(self) -> None:

        if self._idleSteps is None:
            raise ValueError("Number of idle steps not set in AM.")
        if self._collectionSteps is None:
            raise ValueError("Number of collection steps not set in AM.")
        if self._regularisationParameter is None:
            raise ValueError("Regularisation parameter not set in AM.")
        if self._initialCovariance is None:
            raise ValueError("Initial covariance not set in AM.")
