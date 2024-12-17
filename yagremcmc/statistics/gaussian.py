import numpy as np
from numpy.random import standard_normal, uniform
from yagremcmc.statistics.interface import CovarianceOperatorInterface, DensityInterface
from yagremcmc.statistics.parameterLaw import AbsolutelyContinuousParameterLaw
from yagremcmc.parameter.interface import ParameterInterface


class GaussianDensity(DensityInterface):

    def __init__(self, meanVector, covariance: CovarianceOperatorInterface):

        self._mean = meanVector
        self._cov = covariance

    @property
    def covariance(self):
        return self._cov

    def evaluate_log(self, parameter: ParameterInterface) -> float:

        vector = parameter.coefficient

        x = vector - self._mean
        return -0.5 * self._cov.induced_norm_squared(x)


class Gaussian(AbsolutelyContinuousParameterLaw):
    """
    Multivariate Gaussian distribution.

    Instance Attributes
    -------------------
    _mean : ParameterInterface
        Mean of the Gaussian.
    paramType_ : type
        Type of the parameter class.
    _cov: CovarianceOperatorInterface
        Covariance matrix of the Gaussian.
    """

    def __init__(self, mean: ParameterInterface,
                 covariance: CovarianceOperatorInterface):

        self._mean = mean
        self._cov = covariance

        self._density = GaussianDensity(self._mean.coefficient, self._cov)

    @property
    def mean(self):
        return self._mean

    @property
    def covariance(self):
        return self._cov

    @property
    def density(self):
        return self._density

    def generate_realisation(self) -> ParameterInterface:

        xi = standard_normal(self._mean.dimension)
        colouredXi = self._cov.apply_chol_factor(xi)

        return self._mean.clone_with(self._mean.coefficient + colouredXi)
