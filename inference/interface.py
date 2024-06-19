from abc import ABC, abstractmethod
from parameter.interface import ParameterInterface


class TargetDensityInterface(ABC):

    @abstractmethod
    def evaluate_ratio(self, state1: ParameterInterface,
                       state2: ParameterInterface) -> float:
        pass


class ParameterLawInterface(ABC):

    @abstractmethod
    def evaluate_log_density(self, state: ParameterInterface) -> float:
        pass

    @abstractmethod
    def generate_realisation(self) -> ParameterInterface:
        pass


class BayesianModelInterface(ABC):

    @abstractmethod
    def log_prior(self, parameter: ParameterInterface) -> float:
        pass

    @abstractmethod
    def log_likelihood(self, parameter: ParameterInterface) -> float:
        pass


class LikelihoodInterface(ABC):

    @abstractmethod
    def evaluate_log_likelihood(self, parameter: ParameterInterface) -> float:
        pass


class NoiseModelInterface(ABC):

    @abstractmethod
    def induced_norm_squared(self, vector):
        pass
