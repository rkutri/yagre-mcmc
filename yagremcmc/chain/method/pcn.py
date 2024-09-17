from numpy import zeros, sqrt, exp

from yagremcmc.chain.proposal import ProposalMethod
from yagremcmc.chain.metropolisHastings import MetropolisHastings
from yagremcmc.chain.builder import ChainBuilder
from yagremcmc.statistics.parameterLaw import Gaussian


class PCNProposal(ProposalMethod):

    def __init__(self, prior, stepSize):

        if not isinstance(prior, Gaussian):
            raise NotImplementedError("PCN only supports Gaussian priors")

        super().__init__()

        self.prior_ = prior
        self._stepSize = stepSize

        self.proposalLaw_ = None

    def generate_proposal(self):

        if self._state is None:
            raise ValueError(
                "Trying to generate proposal with undefined state")

        xi = self.prior_.generate_realisation()

        t = 2. * self._stepSize
        ParamType = type(self._state)

        return ParamType(
            sqrt(1. - t) * self._state.coefficient + sqrt(t) * xi.coefficient)


class PreconditionedCrankNicolson(MetropolisHastings):

    def __init__(self, targetDensity, prior, stepSize):

        assert 0 < stepSize and stepSize <= 0.5

        if not prior.mean == type(prior.mean)(zeros(prior.mean.dimension)):
            raise ValueError("Preconditioned Crank Nicholson requires "
                             + "centred prior")

        proposalMethod = PCNProposal(prior, stepSize)

        super().__init__(targetDensity, proposalMethod)

    def _acceptance_probability(self, proposal, state):

        lRatio = exp(self._tgtDensity.evaluate_log(proposal)
                     - self._tgtDensity.evaluate_log(state))

        return lRatio if lRatio < 1. else 1.


class PCNBuilder(ChainBuilder):

    def __init__(self):

        super().__init__()
        self._stepSize = None

    @property
    def stepSize(self):
        return self._stepSize

    @stepSize.setter
    def stepSize(self, h):
        self._stepSize = h

    def build_from_model(self) -> MetropolisHastings:

        self._validate_parameters()

        return PreconditionedCrankNicolson(
            self._bayesModel.likelihood, self._bayesModel.prior, self._stepSize)

    def build_from_target(self) -> MetropolisHastings:

        raise RuntimeError(
            "PCN is only defined in relation to a Bayesian model")

    def _validate_parameters(self) -> None:

        if self._stepSize is None:
            raise ValueError("Step size not set in PCN.")
