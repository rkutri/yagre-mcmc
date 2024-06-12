from parameter.interface import ParameterInterface


class ScalarParameter(ParameterInterface):

    def __init__(self, vector):

        if (isinstance(vector, float)):
            raise Exception("scalar parameters must be 1-dimensional array " 
                             + "types")

        self.vector_ = vector
        self.vectorType_ = type(vector)


    @classmethod
    def from_interpolation(cls, value):
        return cls(value)

    @property
    def dimension(self):
        return 1

    @property
    def vector_type(self):
        return self.vectorType_

    @property
    def vector(self):
        return self.vector_

    def evaluate(self):
        return self.vector_
