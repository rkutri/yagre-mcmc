from numpy import exp
from yagremcmc.statistics.interface import BayesianModelInterface
from yagremcmc.statistics.likelihood import BayesianRegressionLikelihood


class BayesianRegressionModel(BayesianModelInterface):

    def __init__(self, data, prior, forwardModel, noiseModel):

        self.prior_ = prior
        self.likelihood_ = BayesianRegressionLikelihood(
            data, forwardModel, noiseModel)

    @property
    def prior(self):
        return self.prior_

    @property
    def likelihood(self):
        return self.likelihood_

    def log_prior(self, parameter):

        # the prior is itself a parameter law
        return self.prior_.density.evaluate_log(parameter)

    def log_likelihood(self, parameter):

        # the likelihood is just the Radon-Nikodym derivative
        return self.likelihood_.evaluate_log(parameter)
