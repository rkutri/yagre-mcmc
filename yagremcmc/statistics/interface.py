from abc import ABC, abstractmethod
from numpy import ndarray
from yagremcmc.parameter.interface import ParameterInterface


class DensityInterface(ABC):

    @abstractmethod
    def evaluate_log(self, state: ParameterInterface) -> float:
        pass


class ParameterLawInterface(ABC):

    @abstractmethod
    def generate_realisation(self) -> ParameterInterface:
        pass


class CovarianceOperatorInterface(ABC):

    @abstractmethod
    def apply_inverse(self, x):
        pass


class BayesianModelInterface(ABC):

    @property
    @abstractmethod
    def likelihood(self) -> DensityInterface:
        pass

    @property
    @abstractmethod
    def prior(self) -> ParameterLawInterface:
        pass


class NoiseModelInterface(ABC):

    @abstractmethod
    def induced_norm_squared(self, vector) -> float:
        pass
