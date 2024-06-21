from parameter.interface import ParameterInterface
from math import isclose


class ScalarParameter(ParameterInterface):

    def __init__(self, coefficient):

        if (isinstance(coefficient, float)):
            raise Exception("scalar parameters must be 1-dimensional array "
                            + "types")

        self.coefficient_ = coefficient
        self.coeffType_ = type(coefficient)

    @classmethod
    def from_coefficient(cls, coefficient):
        return cls(coefficient)

    @classmethod
    def from_value(cls, value):
        return cls(value)

    @property
    def dimension(self):
        return 1

    @property
    def coefficient_type(self):
        return self.coeffType_

    @property
    def coefficient(self):
        return self.coefficient_

    def evaluate(self):
        return self.coefficient_

    def __eq__(self, other):

        if isinstance(other, ScalarParameter):
            return isclose(self.coefficient_, other.coefficient)

        return NotImplemented
