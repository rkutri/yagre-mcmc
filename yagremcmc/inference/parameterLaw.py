from abc import abstractmethod
from numpy import sqrt, dot, reciprocal, full
from numpy.random import standard_normal, uniform
from yagremcmc.inference.interface import DensityInterface, CovarianceOperatorInterface
from yagremcmc.parameter.interface import ParameterInterface


class ParameterLaw(DensityInterface):

    @abstractmethod
    def generate_realisation(self) -> ParameterInterface:
        pass


class Gaussian(ParameterLaw):
    """
    Multivariate Gaussian distribution.

    Instance Attributes 
    -------------------
    mean_ : ParameterInterface
        Mean of the Gaussian.
    paramType_ : type
        Type of the parameter class.
    cov_: CovarianceOperatorInterface
        Covariance matrix of the Gaussian.
    """

    def __init__(self, mean: ParameterInterface, covariance: CovarianceOperatorInterface):

        self.mean_ = mean
        self.paramType_ = type(mean)
        self.cov_ = covariance

    @property
    def mean(self):
        return self.mean_

    def evaluate_log(self, state: ParameterInterface) -> float:

        x = state.coefficient - self.mean_.coefficient
        Px = self.cov_.apply_inverse(x)

        return -0.5 * dot(x, Px)

    def generate_realisation(self) -> ParameterInterface:

        xi = standard_normal(self.mean_.dimension)
        colouredXi = self.cov_.apply_sqrt(xi)

        return self.paramType_(self.mean_.coefficient + colouredXi)
