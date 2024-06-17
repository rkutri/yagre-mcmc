from sys import exit

from numpy import exp, log, square, sqrt
from numpy.random import standard_normal
from scipy.stats import multivariate_normal
from scipy.integrate import solve_ivp

from parameter.vector import ParameterVector
from inference.interface import TargetDensityInterface
from inference.data import Data
from model.interface import SolverInterface
from model.forwardModel import ForwardModel


class GaussianTargetDensity1d(TargetDensityInterface):

    def __init__(self, mean, var):

        self.mean_ = mean
        self.var_ = var

    def evaluate_ratio(self, num, denom):

        return exp(0.5 / self.var_
                * (square(self.mean_.coefficient - denom.coefficient)
                   - square(self.mean_.coefficient - num.coefficient)))

    def evaluate_on_mesh(self, mesh):

        return exp(-0.5 * square((self.mean_.coefficient - mesh) / self.var_))


class GaussianTargetDensity2d(TargetDensityInterface):

    def __init__(self, mean, cov):

        self.dist_ = multivariate_normal(mean.coefficient, cov)

    def evaluate_ratio(self, pNum, pDenom):

        return exp(self.dist_.logpdf(pNum.coefficient)
                   - self.dist_.logpdf(pDenom.coefficient))

    def evaluate_on_mesh(self, mesh):

        return self.dist_.pdf(mesh)


class LotkaVolterraParameter(ParameterVector):

    @classmethod
    def from_coefficient(cls, coefficient):
        return cls(coefficient)

    @classmethod
    def from_interpolation(cls, value):
        return cls(log(value))

    def evaluate(self):
        return exp(self.coefficient_)


class LotkaVolterraSolver(SolverInterface):

    def __init__(self):

        self.param_ = [None, None]


    def flow__(self, t, x, alpha, beta, gamma, delta):

        return [alpha * x[0] - beta * x[0] * x[1],
                delta * x[0] * x[1] - gamma * x[1]]


    def configure(self, config):
        pass


    def interpolate(self, parameter):

        paramEval = parameter.evaluate()

        self.param_ = [paramEval[0], paramEval[1]]

        return 

    def invoke(self):

        alpha = 
        beta = 
        gamma = 
        delta = 

        odeResult = solve_ivp(self.flow_, tBoundary, x,
                             args=(alpha, beta, gamma, delta), method='LSODA')

        if (odeResult.status != 0):

            print("forward map evaluation failed. aborting program.")
            print(odeResult.message)
            exit()

        self.evaluation_ = odeResult.y[:, -1]




class LotkaVolterraModel(ForwardModel):

    def full_solution(self, parameter, x, config):

        tBoundary = (0., config['T'])

        alpha = config['alpha']
        gamma = config['beta']

        paramEval = parameter.evaluate()

        beta = paramEval[0]
        delta = paramEval[1]

        odeResult = solve_ivp(self.flow_, tBoundary, x,
                             args=(alpha, beta, gamma, delta), method='LSODA')

        if (odeResult.status != 0):

            print("forward map evaluation failed. aborting program.")
            print(odeResult.message)
            exit()

        return (odeResult.t, odeResult.y)


def generate_synthetic_data(parameter, fwdModel, design, noiseVar):

    paramDim = fwdModel.parameter.dimension
    sig = sqrt(noiseVar)

    measurement = [fwdModel.point_eval(x, parameter)
                   + sig * standard_normal(paramDim) for x in design]

    return Data(measurement)
