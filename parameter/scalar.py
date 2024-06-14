from parameter.interface import ParameterInterface


class ScalarParameter(ParameterInterface):

    def __init__(self, coefficient):

        if (isinstance(coefficient, float)):
            raise Exception("scalar parameters must be 1-dimensional array " 
                             + "types")

        self.coefficient_ = coefficient
        self.coeffType_ = type(coefficient)


    @classmethod
    def from_interpolation(cls, value):
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
