from abc import ABC, abstractmethod


class SolverInterface(ABC):

    @property
    @abstractmethod
    def status(self):
        pass

    @property
    @abstractmethod
    def evaluation(self):
        pass

    @abstractmethod
    def interpolate(self, parameter):
        pass

    @abstractmethod
    def invoke(self):
        pass
