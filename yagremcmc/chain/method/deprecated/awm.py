import logging

import numpy as np

from yagremcmc.chain.adaptive import AdaptiveMRWProposal, AdaptiveCovarianceMatrix
from yagremcmc.chain.metropolisHastings import MetropolisHastings
from yagremcmc.chain.builder import ChainBuilder
from yagremcmc.chain.target import UnnormalisedPosterior
from yagremcmc.statistics.covariance import DiagonalCovarianceMatrix


awCovLogger = logging.getLogger(__name__)
awCovLogger.setLevel(logging.INFO)

consoleHandler = logging.StreamHandler()

formatter = logging.Formatter('%(levelname)s - %(name)s: %(message)s')
consoleHandler.setFormatter(formatter)

awCovLogger.addHandler(consoleHandler)


class AdaptiveWeightingCovarianceMatrix(AdaptiveCovarianceMatrix):

    def __init__(self, initCov, idleSteps, collectionSteps):

        super().__init__(initCov)

        self._idleSteps = idleSteps
        self._collectionSteps = collectionSteps

        self._nData = 0

        # relevant quantities for Wellford's variance updates
        self._mean = np.zeros(initCov.dimension)
        self._aggSquaredDiff = np.zeros(initCov.dimension)

    def update(self):

        if self._chain is None:
            raise ValueError("Adaptive covariance not associated with a chain")

        nChain = self._chain.length

        if nChain <= self._idleSteps:
            return

        if nChain == self._idleSteps:
            awCovLogger.info(
                "Start collecting data for adaptive covariance.")

        states = self._chain.accepted_states(self._idleSteps)

        if not self._update_required(states):
            return

        # the order in lines 55 and 56 is important
        self._nData = len(states)
        self._wellford_update(states)

        if nChain > self._idleSteps + self._collectionSteps:

            if nChain == self._idleSteps + self._collectionSteps + 1:
                awCovLogger.info("Start using adaptive covariance.")

            if (self._nData < 2):
                raise ValueError(
                    "Marginal variance computation requires more than one sample")

            margVar = self._aggSquaredDiff / (self._nData - 1)

            self._cov == DiagonalCovarianceMatrix(margVar)

    def _wellford_update(self, states):
        """
        Use Wellford's algorithm for incremental variance updates
        """
        newState = states[-1]

        deviation = newState - self._mean

        self._mean += deviation / self._nData
        self._aggSquaredDiff += deviation * (newState - self._mean)

    def _update_required(self, acceptedStates):

        if self._nData == len(acceptedStates):
            return False

        else:

            if (len(acceptedStates) - self._nData == 1):
                return True
            else:
                raise RuntimeError("Inconsistent data size")


class AdaptiveWeightingMetropolis(MetropolisHastings):

    def __init__(self, targetDensity, initCov, idleSteps, collectionSteps):
        """
            Parameters:
            - targetDensity: The target density to sample from.
            - initCov: Initial covariance matrix.
            - idleSteps: Number of steps during which the covariance is not updated.
            - collectionSteps: Number of steps where samples are collected without updating the covariance.
        """

        adaptiveCovariance = AdaptiveWeightingCovarianceMatrix(
            initCov, idleSteps, collectionSteps)
        proposalMethod = AdaptiveMRWProposal(adaptiveCovariance)

        # instantiates self._chain
        super().__init__(targetDensity, proposalMethod)

        self._proposalMethod.covariance.set_chain(self._chain)

    def _acceptance_probability(self, proposal, state):

        densityRatio = np.exp(self._tgtDensity.evaluate_log(proposal)
                              - self._tgtDensity.evaluate_log(state))

        return np.min([densityRatio, 1.])


class AWMBuilder(ChainBuilder):
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
    def initialCovariance(self):
        return self._initialCovariance

    @initialCovariance.setter
    def initialCovariance(self, cov):
        self._initialCovariance = cov

    def build_from_model(self) -> MetropolisHastings:

        self._validate_parameters()
        targetDensity = UnnormalisedPosterior(self._bayesModel)
        return AdaptiveWeightingMetropolis(
            targetDensity, self._initialCovariance, self._idleSteps, self.
            _collectionSteps)

    def build_from_target(self) -> MetropolisHastings:

        self._validate_parameters()
        return AdaptiveWeightingMetropolis(
            self._explicitTarget, self._initialCovariance, self._idleSteps,
            self._collectionSteps)

    def _validate_parameters(self) -> None:

        if self._idleSteps is None:
            raise ValueError("Number of idle steps not set in AM.")
        if self._collectionSteps is None:
            raise ValueError("Number of collection steps not set in AM.")
        if self._initialCovariance is None:
            raise ValueError("Initial covariance not set in AM.")
