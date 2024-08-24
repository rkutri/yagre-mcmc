from numpy import array_equal
from yagremcmc.parameter.interface import ParameterInterface


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

        # parameter coefficients are read-only, so a check for true equality
        # (no floating point comparison) is what we want here, as they
        # are not supposed to change from one evaluation to a subsequent one.
        if isinstance(other, self.__class__):
            return array_equal(self.coefficient_, other.coefficient)

        return NotImplemented


    def copy_with(self, newCoefficient):

        if not isinstance(newCoefficient, self.coefficientType_):
            raise ValueError("Trying to change coefficient type in cloning.")

        return ParameterVector(self.coefficient_)
    

