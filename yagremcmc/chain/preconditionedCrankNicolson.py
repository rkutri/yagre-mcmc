from numpy import zeros, sqrt, exp
from yagremcmc.chain.metropolisHastings import MetropolisHastings
from yagremcmc.chain.factory import ChainFactory


class PreconditionedCrankNicolson(MetropolisHastings):

    def __init__(self, targetDensity, prior, stepSize):

        super().__init__(targetDensity)

        assert 0 < stepSize and stepSize <= 0.5

        if not prior.mean == type(prior.mean)(zeros(prior.mean.dimension)):
            raise ValueError("Preconditioned Crank Nicholson requires "
                             + "centred prior")

        self.proposalLaw_ = prior
        self.stepSize_ = stepSize

    def generate_proposal__(self, state):

        xi = self.proposalLaw_.generate_realisation()

        ParamType = type(state)

        t = 2. * self.stepSize_
        return ParamType(
            sqrt(1. - t) * state.coefficient + sqrt(t) * xi.coefficient)

    def acceptance_probability__(self, proposal, state):

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

        raise RuntimeError("PCN is only defined in relation to a Bayesian model")
