from abc import ABC, abstractmethod

from yagremcmc.chain.proposal import ProposalMethod
from yagremcmc.chain.method.mrw import MRWProposal
from yagremcmc.statistics.interface import CovarianceOperatorInterface


class AdaptiveCovarianceMatrix(CovarianceOperatorInterface):

    def __init__(self, initCov):

        self._cov = initCov
        self._chain = None

    @property
    def dimension(self):
        return self._cov.dimension

    def set_chain(self, chain):
        self._chain = chain

    def reweight_dimensions(self, weights):
        self._cov.reweight_dimensions(weights)

    def apply_chol_factor(self, x):
        return self._cov.apply_chol_factor(x)

    def apply_inverse(self, x):
        return self._cov.apply_inverse(x)

    @property
    def covariance(self):
        return self._cov

    @abstractmethod
    def update(self):
        pass


class AdaptiveMRWProposal(ProposalMethod):

    def __init__(self, adaptiveCov):

        if adaptiveCov.dimension == 1:
            raise NotImplementedError(
                "Adaptivity not implemented for scalar chains.")

        super().__init__()

        self._adaptiveCov = adaptiveCov

        self._proposalMethod = MRWProposal(self._adaptiveCov.covariance)

    @property
    def covariance(self):
        return self._adaptiveCov

    def set_state(self, newState):

        self._adaptiveCov.update()

        self._proposalMethod.covariance = self._adaptiveCov.covariance
        self._proposalMethod.set_state(newState)

    def generate_proposal(self):

        return self._proposalMethod.generate_proposal()
