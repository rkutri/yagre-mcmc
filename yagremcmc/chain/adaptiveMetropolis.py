from numpy import exp
from yagremcmc.chain.interface import ProposalMethodInterface
from yagremcmc.chain.metropolisHastings import MetropolisHastings
from yagremcmc.chain.factory import ChainFactory
from yagremcmc.statistics.covariance import AdaptiveCovariance
from yagremcmc.statistics.parameterLaw import Gaussian


class AdaptiveMRWProposal(ProposalMethodInterface):

    def __init__(self, initCov, idleSteps, minMargVar):

        self.cov_ = AdaptiveCovariance(initCov, minMargVar)
        self.idleSteps_ = idleSteps
        self.lastLength_ = self.chain_.length

        self.chain_ = None
        self.state_ = None
        self.proposalLaw = None

    @property
    def state(self):
        return self.state_

    @state.setter
    def state(self, newState):

        self.state_ = newState
        self.proposalLaw_ = Gaussian(self.state_, self.cov_)

    @property
    def chain(self):
        return self.chain_

    @property
    def chain(self, newChain):
        self.chain_ = newChain

    def generate_proposal(self):

        if self.state_ is None:
            raise ValueError(
                "Trying to generate proposal with undefined state.")

        if self.chain_ is None:
            raise ValueError("Adaptive Proposal is not associated to a chain.")

        self._update_covariance()
        self.lastLength_ = self.chain_.length

        return self.proposalLaw_.generate_realisation()

    def _update_covariance(self):

        if self.chain_.length <= self.idleSteps_ \
                or self.chain_.length <= self.lastLength_:
            return

        self.cov_.update(self.chain_)


class AdaptiveMetropolis(MetropolisHastings):

    def __init__(self, targetDensity, initCov, idleSteps, minMargVar):

        proposalMethod = AdaptiveMRWProposal(initCov, idleSteps, minMargVar)

        super().__init__(targetDensity, proposalMethod)

        self.proposalMethod_.chain = self.chain_

    def _acceptance_probability(self, proposal, state):

        densityRatio = exp(self.targetDensity_.evaluate_log(proposal)
                           - self.targetDensity_.evaluate_log(state))

        return densityRatio if densityRatio < 1. else 1.


# TODO: make the hyperparameters properties instead of using explicit setters
class AMFactory(ChainFactory):
    pass
