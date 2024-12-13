from yagremcmc.statistics.interface import BayesianModelInterface
from yagremcmc.utility.hierarchy import Hierarchy, SharedComponent


class BayesianRegressionModel(BayesianModelInterface):

    def __init__(self, likelihood, prior):

        self._likelihood = self._initialize_component(likelihood, "likelihood")
        self._prior = self._initialize_component(prior, "prior")

    def _initialize_component(self, component, name):

        if isinstance(component, Hierarchy):
            if isinstance(component, SharedComponent):
                return component.level(0)
            else:
                raise RuntimeError(f"Setting non-hierarchical {name} with "
                                   f"a {name} hierarchy")

        return component

    @property
    def prior(self):
        return self._prior

    @property
    def likelihood(self):
        return self._likelihood
