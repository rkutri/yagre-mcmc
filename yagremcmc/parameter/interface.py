from abc import ABC, abstractmethod


class ParameterInterface(ABC):
    """
    Coefficient vectors must be read-only. A ParameterInterface
    implementation represents a parameter with a given, fixed coefficient.
    """

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
        """ 
        Test for equality. Is used in memoisation of likelihood evaluations.
        """
        pass

    @abstractmethod
    def clone_with(self, newCoefficient):
        """
        Returns an instance of this class with the same configuration, but
        different coefficient vector.
        """
        pass
