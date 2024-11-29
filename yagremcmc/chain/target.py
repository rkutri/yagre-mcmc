from yagremcmc.statistics.interface import DensityInterface


class UnnormalisedPosterior(DensityInterface):

    def __init__(self, model):

        self._model = model

    def evaluate_log(self, parameter):

        return self._model.log_likelihood(parameter) + \
                self._model.log_prior(parameter)


class TemperedUnnormalisedPosterior(UnnormalisedPosterior):

    def __init__(self, model, tempering):

        super().__init__(model)
        self._tempering = tempering

    @property
    def tempering(self):
        return self._tempering

    @tempering.setter
    def tempering(self, value):
        self._tempering = value

    def evaluate_log(self, parameter):

        return self._tempering * self._model.log_likelihood(parameter) + \
                self._model.log_prior(parameter)

