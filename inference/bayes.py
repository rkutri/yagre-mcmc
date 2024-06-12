from numpy import exp
from inference.interface import BayesianModel
from inference.likelihood import BayesianRegressionLikelihood


class BayesianRegressionModel(BayesianModel):

    def __init__(self, data, prior, forwardMap, noiseModel):

        self.prior_ = prior
        self.likelihood_ = BayesianRegressionLikelihood(
            data, forwardMap, noiseModel)

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
