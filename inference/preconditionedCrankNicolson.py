from numpy import zeros, sqrt, exp
from inference.interface import DensityInterface
from inference.markovChain import MetropolisHastings
from inference.likelihood import BayesianRegressionLikelihood


class PreconditionedCrankNicolson(MetropolisHastings):

    def __init__(self, targetDensity, prior, stepSize):

        super().__init__(targetDensity)

        assert 0 < stepSize and stepSize <= 0.5

        if not prior.mean == type(prior.mean)(zeros(prior.mean.dimension)):
            raise ValueError("Preconditioned Crank Nicholson requires "
                             + "centred prior")

        self.proposalLaw_ = prior
        self.stepSize_ = stepSize

    @classmethod
    def from_bayes_model(cls, model, stepSize):
        return cls(model.likelihood, model.prior, stepSize)

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
