from numpy import zeros, sqrt, exp
from yagremcmc.chain.interface import ProposalMethodInterface
from yagremcmc.chain.metropolisHastings import MetropolisHastings
from yagremcmc.chain.factory import ChainFactory
from yagremcmc.statistics.parameterLaw import Gaussian


class PCNProposal(ProposalMethodInterface):

    def __init__(self, prior, stepSize):

        if not isinstance(prior, Gaussian):
            raise NotImplementedError("PCN only supports Gaussian priors")

        self.prior_ = prior
        self.stepSize_ = stepSize

        self.state_ = None
        self.proposalLaw_ = None

    @property
    def state(self):
        return self.state_

    @state.setter
    def state(self, newState):
        self.state_ = newState

    def generate_proposal(self):

        if self.state_ == None:
            raise ValueError(
                "Trying to generate proposal with undefined state")

        xi = self.prior_.generate_realisation()

        t = 2. * self.stepSize_
        ParamType = type(self.state_)

        return ParamType(
            sqrt(1. - t) * self.state_.coefficient + sqrt(t) * xi.coefficient)


class PreconditionedCrankNicolson(MetropolisHastings):

    def __init__(self, targetDensity, prior, stepSize):

        assert 0 < stepSize and stepSize <= 0.5

        if not prior.mean == type(prior.mean)(zeros(prior.mean.dimension)):
            raise ValueError("Preconditioned Crank Nicholson requires "
                             + "centred prior")

        proposalMethod = PCNProposal(prior, stepSize)

        super().__init__(targetDensity, proposalMethod)

    def _acceptance_probability(self, proposal, state):

        lRatio = exp(self.targetDensity_.evaluate_log(proposal)
                     - self.targetDensity_.evaluate_log(state))

        return lRatio if lRatio < 1. else 1.


class PCNFactory(ChainFactory):

    def __init__(self):

        super().__init__()
        self.stepSize_ = None

    def set_step_size(self, stepSize):
        self.stepSize_ = stepSize

    def build_from_model(self) -> MetropolisHastings:

        return PreconditionedCrankNicolson(self.bayesModel_.likelihood, self.bayesModel_.prior, self.stepSize_)

    def build_from_target(self) -> MetropolisHastings:

        raise RuntimeError(
            "PCN is only defined in relation to a Bayesian model")
