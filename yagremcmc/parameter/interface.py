from abc import ABC, abstractmethod


class ParameterInterface(ABC):

    @classmethod
    @abstractmethod
    def from_coefficient(cls, coefficient):
        pass

    @property
    @abstractmethod
    def dimension(self):
        pass

    @property
    @abstractmethod
    def coefficient(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def __eq__(self, other):
        pass
