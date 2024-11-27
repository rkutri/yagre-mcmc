from yagremcmc.statistics.interface import DensityInterface


class UnnormalisedPosterior(DensityInterface):

    def __init__(self, model):

        self._model = model

    def evaluate_log(self, parameter):

        return self._model.log_likelihood(
            parameter) + self._model.log_prior(parameter)
