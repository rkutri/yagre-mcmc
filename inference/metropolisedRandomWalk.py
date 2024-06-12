from numpy import exp
from inference.markovChain import MetropolisHastings
from inference.parameterLaw import IIDGaussian
from inference.interface import TargetDensityInterface


class UnnormalisedPosterior(TargetDensityInterface):

    def __init__(self, model):

        self.model_ = model

    def evaluate_ratio(self, sNum, sDenom):

        postRatio = self.model_.log_likelihood(sNum) \
            - self.model_.log_likelihood(sDenom) \
            + self.model_.log_prior(sNum) - \
            self.model_.log_prior(sDenom)

        return exp(postRatio)


class MetropolisedRandomWalk(MetropolisHastings):

    def __init__(self, targetDensity, proposalVariance):

        super().__init__(targetDensity)

        self.proposalVariance_ = proposalVariance

    @classmethod
    def from_bayes_model(cls, model, proposal):

        targetDensity = UnnormalisedPosterior(model)

        return cls(targetDensity, proposal)

    def generate_proposal__(self, state):

        proposalMeasure = IIDGaussian(state, self.proposalVariance_)

        realisation = proposalMeasure.generate_realisation()

        return realisation

    def acceptance_probability__(self, proposal, state):

        # proposal is symmetric
        densityRatio = self.targetDensity_.evaluate_ratio(proposal, state)

        return densityRatio if densityRatio < 1. else 1.
