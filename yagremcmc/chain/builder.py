from abc import ABC, abstractmethod

from yagremcmc.chain.metropolisHastings import MetropolisHastings


class ChainBuilder(ABC):

    def __init__(self):

        self._bayesModel = None
        self._explicitTarget = None

    @property
    def bayesModel(self):
        return self._bayesModel

    @bayesModel.setter
    def bayesModel(self, model):
        self._bayesModel = model

    @property
    def explicitTarget(self):
        return self._explicitTarget

    @explicitTarget.setter
    def explicitTarget(self, density):
        self._explicitTarget = density

    def validate_target_measure(self):

        if self._bayesModel is None and self._explicitTarget is None:
            raise ValueError("Either bayesian model or explicit target density"
                             + " must be provided for chain setup")

        if self._bayesModel is not None and self._explicitTarget is not None:
            raise ValueError("Only one of bayes model or explicit target"
                             + " density should be provided.")

    def target_is_posterior(self):

        self.validate_target_measure()

        return self._bayesModel is not None

    def target_is_explicit(self):

        self.validate_target_measure()

        return self._explicitTarget is not None

    @abstractmethod
    def _validate_parameters(self) -> None:
        pass

    @abstractmethod
    def build_from_model(self) -> MetropolisHastings:
        pass

    @abstractmethod
    def build_from_target(self) -> MetropolisHastings:
        pass

    def build_method(self):

        self.validate_target_measure()

        if self.target_is_posterior():
            return self.build_from_model()

        if self.target_is_explicit():
            return self.build_from_target()

        raise ValueError("Invalid target distribution")
