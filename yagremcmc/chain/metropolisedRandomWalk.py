from numpy import exp
from yagremcmc.chain.proposal import ProposalMethod
from yagremcmc.chain.metropolisHastings import MetropolisHastings, UnnormalisedPosterior
from yagremcmc.chain.factory import ChainFactory
from yagremcmc.statistics.parameterLaw import Gaussian


class MRWProposal(ProposalMethod):

    def __init__(self, proposalCov):

        super().__init__()

        self.cov_ = proposalCov

        self.proposalLaw_ = None

    def set_state(self, newState):

        self._state = newState
        self.proposalLaw_ = Gaussian(self._state, self.cov_)

    def generate_proposal(self):

        if self._state is None:
            raise ValueError(
                "Trying to generate proposal with undefined state")

        return self.proposalLaw_.generate_realisation()


class MetropolisedRandomWalk(MetropolisHastings):

    def __init__(self, targetDensity, proposalCov):

        proposalMethod = MRWProposal(proposalCov)

        super().__init__(targetDensity, proposalMethod)

    def _acceptance_probability(self, proposal, state):

        # proposal is symmetric
        densityRatio = exp(self._tgtDensity.evaluate_log(proposal)
                           - self._tgtDensity.evaluate_log(state))

        return densityRatio if densityRatio < 1. else 1.


class MRWFactory(ChainFactory):

    def __init__(self):

        super().__init__()
        self.proposalCov_ = None

    @property
    def proposalCovariance(self):
        return self.proposalCov_

    @proposalCovariance.setter
    def proposalCovariance(self, covariance):
        self.proposalCov_ = covariance

    def build_from_model(self) -> MetropolisHastings:

        self._validate_parameters()

        targetDensity = UnnormalisedPosterior(self.bayesModel_)

        return MetropolisedRandomWalk(targetDensity, self.proposalCov_)

    def build_from_target(self) -> MetropolisHastings:

        self._validate_parameters()

        return MetropolisedRandomWalk(self.explicitTarget_, self.proposalCov_)

    def _validate_parameters(self) -> None:

        if self.proposalCov_ is None:
            raise ValueError("Proposal Covariance not set for MRW")
