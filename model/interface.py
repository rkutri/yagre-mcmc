from abc import ABC, abstractproperty, abstractmethod


class ModelProblemInterface(ABC):

    @abstractproperty
    def solver(self):
        pass

    @abstractproperty
    def configuration(self):
        pass


class EvaluationRequestInterface(ABC):

    @property
    @abstractmethod
    def result(self):
        pass


    @abstractmethod
    def submit(self, solver):
        pass


class Solver(ABC):

    @abstractproperty
    def result(self):
        pass
