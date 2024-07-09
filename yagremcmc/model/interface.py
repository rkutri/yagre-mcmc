from abc import ABC, abstractproperty, abstractmethod


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
