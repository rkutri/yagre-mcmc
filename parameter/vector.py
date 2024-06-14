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
    def from_interpolation(cls, value):
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
