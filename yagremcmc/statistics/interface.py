from abc import ABC, abstractmethod
from yagremcmc.parameter.interface import ParameterInterface


class DensityInterface(ABC):

    @abstractmethod
    def evaluate_log(self, state: ParameterInterface) -> float:
        pass


class CovarianceOperatorInterface(ABC):

    @property
    @abstractmethod
    def dimension(self):
        pass

    @abstractmethod
    def apply_chol_factor(self, x):
        pass

    @abstractmethod
    def apply_inverse(self, x):
        pass


class BayesianModelInterface(ABC):

    @abstractmethod
    def log_prior(self, parameter: ParameterInterface) -> float:
        pass

    @abstractmethod
    def log_likelihood(self, parameter: ParameterInterface) -> float:
        pass


class NoiseModelInterface(ABC):

    @abstractmethod
    def induced_norm_squared(self, vector):
        pass
