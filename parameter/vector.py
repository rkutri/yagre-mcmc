from parameter.interface import ParameterInterface


class ParameterVector(ParameterInterface):

    def __init__(self, vectorArray):

        self.vector_ = vectorArray

        self.dim_ = vectorArray.size
        self.vectorType_ = type(vectorArray)

    @classmethod
    def from_interpolation(cls, value):
        return cls(value)

    @property
    def dimension(self):
        return self.dim_

    @property
    def vector_type(self):
        return self.vectorType_

    @property
    def vector(self):
        return self.vector_

    def evaluate(self):
        return self.vector_
