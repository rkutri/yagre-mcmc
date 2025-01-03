from yagremcmc.statistics.interface import DensityInterface


class UnnormalisedPosterior(DensityInterface):

    def __init__(self, likelihood, prior):

        self._likelihood = likelihood
        self._prior = prior

    @property
    def likelihood(self):
        return self._likelihood

    @property
    def prior(self):
        return self._prior

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

        paramVector = parameter.coefficient

        if self._correction is None:
            return self._pristineDensity.evaluate_log(paramVector)

        else:

            correctedVector = paramVector + self._correction
            return self._pristineDensity.evaluate_log(correctedVector)
