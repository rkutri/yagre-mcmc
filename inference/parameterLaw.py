from abc import abstractmethod
from numpy import sqrt, log, dot
from numpy.random import standard_normal, uniform
from inference.interface import DensityInterface
from parameter.interface import ParameterInterface


class ParameterLaw(DensityInterface):

    @abstractmethod
    def generate_realisation(self) -> ParameterInterface:
        pass


class IIDGaussian(ParameterLaw):
    """
    Multivariate Gaussian, where the covariance matrix is diagonal, and
    all the diagonal entries have the same value.
    """

    def __init__(self, mean: ParameterInterface, variance: float) -> None:

        self.mean_ = mean
        self.variance_ = variance

    @property
    def mean(self):
        return self.mean_

    def evaluate_log(self, state):

        x = state.coefficient - self.mean_.coefficient

        return -0.5 * dot(x, x) / self.variance_

    def generate_realisation(self):

        xiVal = standard_normal(self.mean_.dimension)

        ParamType = type(self.mean_)

        return ParamType(self.mean_.coefficient + sqrt(self.variance_) * xiVal)


class Uniform(ParameterLaw):

    def __init__(self, low: ParameterInterface,
                 high: ParameterInterface) -> None:

        assert isinstance(high, type(low))

        self.paramType_ = type(low)
        self.paramValueType_ = low.coefficient_type

        self.low_ = low
        self.high_ = high

    def evaluate(self, state):

        x0 = state.coefficient[0]
        x1 = state.coefficient[1]

        lo0 = self.low_.coefficient[0]
        lo1 = self.low_.coefficient[1]

        hi0 = self.high_.coefficient[0]
        hi1 = self.high_.coefficient[1]

        d = state.dimension

        if (lo0 <= x0 and x0 <= hi0):

            if (lo1 <= x1 and x1 <= hi1):

                pdf = 1.

                for i in range(d):
                    pdf /= self.high_.coefficient[i] - self.low_.coefficient[i]

                return pdf

            else:
                return 0.
        else:
            return 0.

    def evaluate_log(self, state):
        return log(self.evaluate(state))

    def generate_realisation(self):

        low0 = self.low_.coefficient[0]
        low1 = self.low_.coefficient[1]

        high0 = self.high_.coefficient[0]
        high1 = self.high_.coefficient[1]

        rvVal = self.paramValueType()

        rvVal[0] = uniform(low0, high0, 1)
        rvVal[1] = uniform(low1, high1, 1)

        print(rvVal)

        return self.paramType_(rvVal)
