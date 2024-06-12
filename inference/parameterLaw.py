import numpy as np
from numpy import sqrt, exp
from numpy.random import standard_normal, uniform
from inference.interface import ParameterLawInterface
from parameter.interface import ParameterInterface

# TODO: if this becomes too crowded, it might be useful to implement
#       a suitable factory/builder pattern


class IIDGaussian(ParameterLawInterface):
    """
    Multivariate Gaussian, where the covariance matrix is diagonal, and 
    all the diagonal entries have the same value.
    """

    def __init__(self, mean: ParameterInterface, variance: float) -> None:

        self.mean_ = mean
        self.variance_ = variance

        self.paramType_ = type(mean)
        self.paramVectorType_ = mean.vector_type

    @property
    def mean(self):
        return self.mean_

    def evaluate_log_density(self, state):

        x = state.vector - self.mean_.vector

        return -0.5 * np.dot(x, x) / self.variance_

    def generate_realisation(self):

        xiVal = standard_normal(self.mean_.dimension)

        return self.paramType_(self.mean_.vector + sqrt(self.variance_) * xiVal)


class Uniform(ParameterLawInterface):

    def __init__(self, low: ParameterInterface, high: ParameterInterface) -> None:

        assert isinstance(high, type(low))

        self.paramType_ = type(low)
        self.paramValueType_ = low.vector_type

        self.low_ = low
        self.high_ = high

    def evaluate_density(self, state):

        x0 = state.vector[0]
        x1 = state.vector[1]

        lo0 = self.low_.vector[0]
        lo1 = self.low_.vector[1]

        hi0 = self.high_.vector[0]
        hi1 = self.high_.vector[1]

        d = state.dimension

        if (lo0 <= x0 and x0 <= hi0):

            if (lo1 <= x1 and x1 <= hi1):

                pdf = 1.

                for i in range(d):
                    pdf /= self.high_.vector[i] - self.low_.vector[i]

                return pdf

            else:
                return 0.
        else:
            return 0.

    def generate_realisation(self):

        low0 = self.low_.vector[0]
        low1 = self.low_.vector[1]

        high0 = self.high_.vector[0]
        high1 = self.high_.vector[1]

        rvVal = self.paramValueType()

        rvVal[0] = uniform(low0, high0, 1)
        rvVal[1] = uniform(low1, high1, 1)

        print(rvVal)

        return self.paramType_(rvVal)
