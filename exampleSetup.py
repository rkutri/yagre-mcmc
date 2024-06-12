import inference.interface as ifc

from sys import exit
from numpy import exp, log, square, sqrt
from numpy.random import standard_normal
from scipy.stats import multivariate_normal
from scipy.integrate import solve_ivp
from inference.data import InputOutputData
from parameter.vector import ParameterVector


class GaussianTargetDensity1d(ifc.TargetDensityInterface):

    def __init__(self, mean, var):

        self.mean_ = mean
        self.var_ = var

    def evaluate_ratio(self, num, denom):

        return exp(0.5 / self.var_ * (square(self.mean_.vector - denom.vector)
                                      - square(self.mean_.vector - num.vector)))

    def evaluate_on_mesh(self, mesh):

        return exp(-0.5 * square((self.mean_.vector - mesh) / self.var_))


class GaussianTargetDensity2d(ifc.TargetDensityInterface):

    def __init__(self, mean, cov):

        self.dist_ = multivariate_normal(mean.vector, cov)

    def evaluate_ratio(self, pNum, pDenom):

        return exp(self.dist_.logpdf(pNum.vector)
                   - self.dist_.logpdf(pDenom.vector))

    def evaluate_on_mesh(self, mesh):

        return self.dist_.pdf(mesh)


class LotkaVolterraParameter(ParameterVector):

    @classmethod
    def from_interpolation(cls, value):
        return cls(log(value))

    def evaluate(self):
        return exp(self.vector_)


class LotkaVolterraForwardMap(ifc.ForwardMapInterface):
    """
    Two-dimensional Lotka-Volterra model with fixed parameters alpha and
    gamma and unknown interaction rates beta and delta (notation from Wiki).
    Measurement data is the vector of the solution at final time T
    """

    def __init__(self, initialParameter, T, alpha, gamma):

        self.parameter_ = initialParameter
        self.alpha_ = alpha
        self.gamma_ = gamma
        self.T_ = T

    @property
    def parameter(self):

        return self.parameter_

    @parameter.setter
    def parameter(self, parameter):

        self.parameter_ = parameter

    def flow__(self, t, x):

        beta = self.parameter_.evaluate()[0]
        delta = self.parameter_.evaluate()[1]

        return [self.alpha_ * x[0] - beta * x[0] * x[1],
                delta * x[0] * x[1] - self.gamma_ * x[1]]

    def evaluate(self, x):

        tBoundary = (0., self.T_)

        odeResult = solve_ivp(self.flow__, tBoundary, x, method='RK45')

        if (odeResult.status != 0):

            print("forward map evaluation failed. aborting program.")
            print(odeResult.message)
            exit()

        return odeResult.y[:, -1]

    def full_solution(self, x):

        tBoundary = (0., self.T_)

        odeResult = solve_ivp(self.flow__, tBoundary, x, method='RK45')

        if (odeResult.status != 0):

            print("forward map evaluation failed. aborting program.")
            print(odeResult.message)
            exit()

        return (odeResult.t, odeResult.y)


def generate_synthetic_data(fwdMap, inputData, noiseVar):

    paramDim = fwdMap.parameter.dimension
    sig = sqrt(noiseVar)

    outputData = [fwdMap.evaluate(x0) + sig * standard_normal(paramDim)
                  for x0 in inputData]

    return InputOutputData(inputData, outputData)
