from yagremcmc.statistics.interface import DensityInterface


class UnnormalisedPosterior(DensityInterface):

    def __init__(self, likelihood, prior):

        self._likelihood = likelihood
        self._prior = prior

    def evaluate_log(self, parameter):

        return self._likelihood.evaluate_log(parameter) + \
            self._prior.density.evaluate_log(parameter)


class TemperedUnnormalisedPosterior(UnnormalisedPosterior):

    def __init__(self, likelihood, prior, tempering):

        super().__init__(likelihood, prior)
        self._tempering = tempering

    @property
    def tempering(self):
        return self._tempering

    @tempering.setter
    def tempering(self, value):
        self._tempering = value

    def evaluate_log(self, parameter):

        return self._tempering * self._likelihood.evaluate_log(parameter) + \
            self._prior.density.evaluate_log(parameter)


class BiasCorrection(DensityInterface):

    def __init__(self, density, correction):

        self._pristineDensity = density
        self._correction = correction

    @property
    def correction(self):
        return self._correction

    def evaluate_log(self, parameter):

        if self._correction is None:
            return self._pristineDensity.evaluate_log(parameter)

        else:

            correctedCoefficient = parameter.coefficient + self._correction
            correctedParameter = parameter.clone_with(correctedCoefficient)

            return self._pristineDensity.evaluate_log(correctedParameter)
