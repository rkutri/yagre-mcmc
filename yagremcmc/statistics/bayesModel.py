from yagremcmc.statistics.interface import BayesianModelInterface


class BayesianRegressionModel(BayesianModelInterface):

    def __init__(self, prior, likelihood):

        self._prior = prior
        self._likelihood = likelihood

    @property
    def prior(self):
        return self._prior

    @property
    def likelihood(self):
        return self._likelihood

    def log_prior(self, parameter):

        # the prior is itself a parameter law
        return self._prior.density.evaluate_log(parameter)

    def log_likelihood(self, parameter):

        # the likelihood is just the Radon-Nikodym derivative
        return self._likelihood.evaluate_log(parameter)
