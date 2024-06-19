from numpy import exp
from inference.interface import BayesianModelInterface
from inference.likelihood import BayesianRegressionLikelihood


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

        return self.prior_.evaluate_log_density(parameter)

    def log_likelihood(self, parameter):

        return self.likelihood_.evaluate_log_likelihood(parameter)
