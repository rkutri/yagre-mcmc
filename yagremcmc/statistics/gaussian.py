from numpy import sqrt, dot, reciprocal, full
from numpy.random import standard_normal, uniform
from yagremcmc.statistics.interface import CovarianceOperatorInterface, DensityInterface
from yagremcmc.statistics.parameterLaw import AbsolutelyContinuousParameterLaw
from yagremcmc.parameter.interface import ParameterInterface


class GaussianDensity(DensityInterface):

    def __init__(self, mean, covariance):

        self._mean = mean
        self._cov = covariance

    def evaluate_log(self, state: ParameterInterface) -> float:

        x = state.coefficient - self._mean.coefficient
        Px = self._cov.apply_inverse(x)

        return -0.5 * dot(x, Px)


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

        self._density = GaussianDensity(self._mean, self._cov)

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
