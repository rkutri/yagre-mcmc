from abc import ABC, abstractmethod


class SolverConfig(ABC):
    pass


class Solver(ABC):

    @abstractmethod
    def to_config(self, parameter):
        pass

    @abstractmethod
    def invoke(self, config):
        pass
