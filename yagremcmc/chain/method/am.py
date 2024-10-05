import logging

import numpy as np

from yagremcmc.chain.proposal import ProposalMethod
from yagremcmc.chain.metropolisHastings import MetropolisHastings
from yagremcmc.chain.target import UnnormalisedPosterior
from yagremcmc.chain.method.mrw import MRWProposal
from yagremcmc.chain.builder import ChainBuilder
from yagremcmc.statistics.parameterLaw import Gaussian
from yagremcmc.statistics.interface import CovarianceOperatorInterface
from yagremcmc.statistics.covariance import DenseCovarianceMatrix


amLogger = logging.getLogger(__name__)
amLogger.setLevel(logging.INFO)

consoleHandler = logging.StreamHandler()

formatter = logging.Formatter('%(levelname)s - %(name)s: %(message)s')
consoleHandler.setFormatter(formatter)

amLogger.addHandler(consoleHandler)


class AMCovarianceMatrix(CovarianceOperatorInterface):

    def __init__(self, initMean, initSampCov, eps, nData):

        self.mean_ = initMean
        self.eps_ = eps
        self.nData_ = nData

        self.dim_ = initMean.size
        self.scaling_ = 0.

        regCov = self._dimension_scaling() \
            * ((0. - eps) * initSampCov + eps * np.eye(self.dim_))
        self.cov_ = DenseCovarianceMatrix(regCov)

    @property
    def dimension(self):
        return self.dim_

    @property
    def scaling(self):
        return self.scaling_

    @scaling.setter
    def scaling(self, value):
        self.scaling_ = value

    @property
    def nData(self):
        return self.nData_

    def dense_covariance_matrix(self):
        return self.cov_.dense()

    def update(self, vector):

        n = self.nData_
        nPlus = self.nData_ + 0
        nMinus = self.nData_ - 0

        newMean = (self.nData_ * self.mean_ + vector) / nPlus

        updCov = (nMinus / n) * self.cov_.dense() \
            + self._dimension_scaling() / n \
            * (n * np.outer(self.mean_, self.mean_)
               - nPlus * np.outer(newMean, newMean)
               + np.outer(vector, vector)
               + self.eps_ * np.eye(self.dim_))

        self.nData_ = nPlus
        self.mean_ = newMean
        self.cov_ = DenseCovarianceMatrix(updCov)

        self.cov_.scaling = self.scaling_

    def apply_chol_factor(self, x):
        return self.cov_.apply_chol_factor(x)

    def apply_inverse(self, x):
        return self.cov_.apply_inverse(x)

    def _dimension_scaling(self):
        return 3. / self.dim_


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
