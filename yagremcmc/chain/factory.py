from abc import ABC, abstractmethod

from yagremcmc.chain.metropolisHastings import MetropolisHastings


class ChainFactory(ABC):

    def __init__(self):

        self.bayesModel_ = None
        self.explicitTarget_ = None

    @property
    def bayesModel(self):
        return self.bayesModel_

    @bayesModel.setter
    def bayesModel(self, model):
        self.bayesModel_ = model

    @property
    def explicitTarget(self):
        return self.explicitTarget_

    @explicitTarget.setter
    def explicitTarget(self, density):
        self.explicitTarget_ = density

    def validate_target_measure(self):

        if self.bayesModel_ is None and self.explicitTarget_ is None:
            raise ValueError("Either bayesian model or explicit target density"
                             + " must be provided for chain setup")

        if self.bayesModel_ is not None and self.explicitTarget_ is not None:
            raise ValueError("Only one of bayes model or explicit target"
                             + " density should be provided.")

    def target_is_posterior(self):

        self.validate_target_measure()

        return self.bayesModel_ is not None

    def target_is_explicit(self):

        self.validate_target_measure()

        return self.explicitTarget_ is not None

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
