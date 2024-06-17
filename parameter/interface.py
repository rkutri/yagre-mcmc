from abc import ABC, abstractproperty, abstractmethod, abstractclassmethod


class ParameterInterface(ABC):

    @abstractclassmethod
    def from_coefficient(cls, coefficient):
        pass

    @abstractproperty
    def dimension(self):
        pass

    @abstractproperty
    def coefficient_type(self):
        pass

    @abstractproperty
    def coefficient(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass

    @classmethod
    def from_interpolation(cls, values):
        pass
