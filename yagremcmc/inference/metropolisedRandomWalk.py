from numpy import exp
from yagremcmc.inference.markovChain import MetropolisHastings
from yagremcmc.inference.parameterLaw import Gaussian
from yagremcmc.inference.interface import DensityInterface


class UnnormalisedPosterior(DensityInterface):

    def __init__(self, model):

        self.model_ = model

    def evaluate_log(self, parameter):

        return self.model_.log_likelihood(parameter) \
            + self.model_.log_prior(parameter)


class MetropolisedRandomWalk(MetropolisHastings):

    def __init__(self, targetDensity, proposalCov):

        super().__init__(targetDensity)

        self.proposalCov_ = proposalCov

    @classmethod
    def from_bayes_model(cls, model, proposalCov):

        targetDensity = UnnormalisedPosterior(model)

        return cls(targetDensity, proposalCov)

    def generate_proposal__(self, state):

        proposalMeasure = Gaussian(state, self.proposalCov_)

        realisation = proposalMeasure.generate_realisation()

        return realisation

    def acceptance_probability__(self, proposal, state):

        # proposal is symmetric
        densityRatio = exp(self.targetDensity_.evaluate_log(proposal)
                           - self.targetDensity_.evaluate_log(state))

        return densityRatio if densityRatio < 1. else 1.
