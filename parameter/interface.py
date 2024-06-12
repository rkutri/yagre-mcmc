from abc import ABC, abstractproperty, abstractmethod, abstractclassmethod


class ParameterInterface(ABC):

    @abstractclassmethod
    def from_interpolation(cls, value):
        pass

    @abstractproperty
    def dimension(self):
        pass

    @abstractproperty
    def vector_type(self):
        pass

    @abstractproperty
    def vector(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass
