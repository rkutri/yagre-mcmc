from numpy import exp
from yagremcmc.chain.interface import ProposalMethodInterface
from yagremcmc.chain.metropolisHastings import MetropolisHastings
from yagremcmc.chain.factory import ChainFactory
from yagremcmc.statistics.parameterLaw import Gaussian
from yagremcmc.statistics.interface import DensityInterface


class UnnormalisedPosterior(DensityInterface):

    def __init__(self, model):

        self.model_ = model

    def evaluate_log(self, parameter):

        return self.model_.log_likelihood(parameter) \
            + self.model_.log_prior(parameter)


class MRWProposal(ProposalMethodInterface):

    def __init__(self, proposalCov):

        self.cov_ = proposalCov

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

        if self.state_ is None:
            raise ValueError(
                "Trying to generate proposal with undefined state")

        return self.proposalLaw_.generate_realisation()


class MetropolisedRandomWalk(MetropolisHastings):

    def __init__(self, targetDensity, proposalCov):

        proposalMethod = MRWProposal(proposalCov)

        super().__init__(targetDensity, proposalMethod)

    def _acceptance_probability(self, proposal, state):

        # proposal is symmetric
        densityRatio = exp(self.targetDensity_.evaluate_log(proposal)
                           - self.targetDensity_.evaluate_log(state))

        return densityRatio if densityRatio < 1. else 1.


class MRWFactory(ChainFactory):

    def __init__(self):

        super().__init__()
        self.proposalCov_ = None

    def set_proposal_covariance(self, covariance):

        # TODO: add checks to validate the covariance (e.g. s.p.d.)
        self.proposalCov_ = covariance

    def build_from_model(self) -> MetropolisHastings:

        if self.proposalCov_ is None:
            raise ValueError("Proposal Covariance not set for MRW")

        targetDensity = UnnormalisedPosterior(self.bayesModel_)

        return MetropolisedRandomWalk(targetDensity, self.proposalCov_)

    def build_from_target(self) -> MetropolisHastings:

        if self.proposalCov_ is None:
            raise ValueError("Proposal Covariance not set for MRW")

        return MetropolisedRandomWalk(self.explicitTarget_, self.proposalCov_)
