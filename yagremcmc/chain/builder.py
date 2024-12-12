from abc import ABC, abstractmethod

from yagremcmc.chain.metropolisHastings import MetropolisHastings
from yagremcmc.chain.diagnostics import AcceptanceRateDiagnostics


class ChainBuilder(ABC):

    def __init__(self):

        self._bayesModel = None
        self._explicitTarget = None

        # AcceptanceRateDiagnostics provides essentially no overhead and
        # is the default diagnostics for Markov Chains
        self._diagnostics = AcceptanceRateDiagnostics()

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

    @property
    def diagnostics(self):
        return self._diagnostics

    @diagnostics.setter
    def diagnostics(self, diagnostics):
        self._diagnostics = diagnostics

    def validate_target_measure(self):

        if self._bayesModel is None and self._explicitTarget is None:
            raise ValueError("Either bayesian model or explicit target density"
                             + " must be provided for chain setup")

        if self._bayesModel is not None and self._explicitTarget is not None:
            raise ValueError("Only one of bayes model or explicit target"
                             + " density should be provided.")

    def target_is_posterior(self):
        return self._bayesModel is not None

    def target_is_explicit(self):
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

    # TODO: use prototypes to be able to generate several methods from the
    #       same builder. Currently they share states if you try this.
    def build_method(self):

        self._validate_parameters()
        self.validate_target_measure()

        if self.target_is_posterior():
            return self.build_from_model()

        if self.target_is_explicit():
            return self.build_from_target()

        raise ValueError("Invalid target distribution")
