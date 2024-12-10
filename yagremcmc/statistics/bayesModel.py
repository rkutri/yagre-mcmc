from yagremcmc.statistics.interface import BayesianModelInterface


class BayesianRegressionModel(BayesianModelInterface):

    def __init__(self, likelihood, prior):

        self._likelihood = likelihood
        self._prior = prior

    @property
    def prior(self):
        return self._prior

    @property
    def likelihood(self):
        return self._likelihood
