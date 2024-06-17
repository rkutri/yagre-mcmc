from numpy import allclose, zeros_like, sqrt, exp
from inference.interface import TargetDensityInterface
from inference.markovChain import MetropolisHastings


class PCNTargetDensity(TargetDensityInterface):

    def __init__(self, likelihood):

        self.likelihood_ = likelihood

    def evaluate_ratio(self, sNum, sDenom):

        return exp(self.likelihood_.evaluate_log_likelihood(sNum)
                   - self.likelihood_.evaluate_log_likelihood(sDenom))


class PreconditionedCrankNicolson(MetropolisHastings):

    def __init__(self, targetDensity, prior, stepSize):

        super().__init__(targetDensity)

        assert 0 < stepSize and stepSize <= 0.5

        priorMeanCoeff = prior.mean.coefficient
        if (not allclose(priorMeanCoeff, zeros_like(priorMeanCoeff))):
            raise ValueError("Preconditioned Crank Nicholson requires "
                             + "centred prior")

        self.proposalLaw_ = prior
        self.stepSize_ = stepSize

    @classmethod
    def from_bayes_model(cls, model, stepSize):
        return cls(PCNTargetDensity(model.likelihood), model.prior, stepSize)

    def generate_proposal__(self, state):

        xi = self.proposalLaw_.generate_realisation()

        ParamType = type(state)

        t = 2. * self.stepSize_
        return ParamType(
            sqrt(
                1. -
                t) *
            state.coefficient +
            sqrt(t) *
            xi.coefficient)

    def acceptance_probability__(self, proposal, state):

        lRatio = self.targetDensity_.evaluate_ratio(proposal, state)

        return lRatio if lRatio < 1. else 1.
