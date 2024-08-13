from abc import ABC, abstractmethod

from yagremcmc.chain.metropolisHastings import MetropolisHastings


class ChainFactory(ABC):

    def __init__(self):

        self.bayesModel_ = None
        self.explicitTarget_ = None

    def set_bayes_model(self, model):
        self.bayesModel_ = model

    def set_explicit_target(self, density):
        self.explicitTarget_ = density

    def validate(self):

        if self.bayesModel_ is None and self.explicitTarget_ is None:
            raise ValueError("Either bayesian model or explicit target density"
                             + " must be provided for chain setup")

        if self.bayesModel_ is not None and self.explicitTarget_ is not None:
            raise ValueError("Only one of bayes model or explicit target"
                             + " density should be provided.")

    def target_is_posterior(self):

        self.validate()

        return self.bayesModel_ is not None

    def target_is_explicit(self):

        self.validate()

        return self.explicitTarget_ is not None

    @abstractmethod
    def build_from_model(self) -> MetropolisHastings:
        pass

    @abstractmethod
    def build_from_target(self) -> MetropolisHastings:
        pass

    def build_method(self):

        self.validate()

        if self.target_is_posterior():
            return self.build_from_model()

        if self.target_is_explicit():
            return self.build_from_target()

        raise ValueError("Invalid target distribution")
