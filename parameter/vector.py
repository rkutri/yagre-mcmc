from numpy import allclose
from parameter.interface import ParameterInterface


class ParameterVector(ParameterInterface):

    def __init__(self, coefficient):

        self.coefficient_ = coefficient

        self.dim_ = coefficient.size
        self.coefficientType_ = type(coefficient)

    @classmethod
    def from_coefficient(cls, coefficient):
        return cls(coefficient)

    @classmethod
    def from_value(cls, value):
        return cls(value)

    @property
    def dimension(self):
        return self.dim_

    @property
    def coefficient_type(self):
        return self.coeffType_

    @property
    def coefficient(self):
        return self.coefficient_

    def evaluate(self):
        return self.coefficient_

    def __eq__(self, other):

        aTol = 1e-12
        rTol = 1e-7

        if isinstance(other, ParameterVector):
            return allclose(self.coefficient_,
                            other.coefficient, rtol=rTol, atol=aTol)

        return NotImplemented
