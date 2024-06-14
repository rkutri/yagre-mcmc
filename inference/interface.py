from abc import ABC, abstractmethod, abstractproperty
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


class ForwardMapInterface(ABC):
    # TODO: make this adhere to a proxy pattern

    @abstractproperty
    def parameter(self):
        pass

    @parameter.setter
    def parameter(self, parameter: ParameterInterface) -> None:
        pass

    @abstractmethod
    def evaluate(self, x):
        pass


class BayesianModelInterface(ABC):

    @abstractmethod
    def log_prior(self, parameter: ParameterInterface) -> float:
        pass

    @abstractmethod
    def log_likelihood(self, parameter: ParameterInterface) -> float:
        pass


class DataInterface(ABC):

    @abstractproperty
    def size(self):
        pass

    @abstractproperty
    def input(self):
        pass

    @abstractproperty
    def output(self):
        pass


class LikelihoodInterface(ABC):

    @abstractmethod
    def evaluate_log_likelihood(self, parameter: ParameterInterface) -> float:
        pass


class NoiseModelInterface(ABC):

    @abstractmethod
    def induced_norm_squared(self, vector):
        pass
