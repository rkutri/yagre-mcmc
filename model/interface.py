from abc import ABC, abstractmethod


class Solver(ABC):

    @abstractmethod
    def configure(self, parameter):
        pass

    @abstractmethod
    def invoke(self):
        pass
