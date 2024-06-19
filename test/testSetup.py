from sys import exit

from numpy import exp, log, square, sqrt, zeros
from numpy.random import standard_normal
from scipy.stats import multivariate_normal
from scipy.integrate import solve_ivp

from parameter.vector import ParameterVector
from inference.interface import TargetDensityInterface
from inference.data import Data
from model.interface import SolverInterface
from model.forwardModel import EvaluationStatus


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

    def __init__(self, design, config):

        self.x_ = design
        self.tBoundary_ = (0., config['T'])
        self.fixedParam_ = [config['alpha'], config['gamma']]
        self.dataShape_ = (config['nData'], config['dataDim'])

        self.param_ = [None, None]
        self.evaluation_ = None
        self.status_ = EvaluationStatus.NONE

    @property
    def status(self):
        return self.status_

    @property
    def dataShape(self):
        return self.dataShape_

    @property
    def nData(self):
        return self.dataShape_[0]

    @property
    def dataDim(self):
        return self.dataShape_[1]

    @property
    def evaluation(self):
        return self.evaluation_

    def flow__(self, t, x, alpha, beta, gamma, delta):

        return [alpha * x[0] - beta * x[0] * x[1],
                delta * x[0] * x[1] - gamma * x[1]]

    def interpolate(self, parameter):

        paramEval = parameter.evaluate()

        self.param_ = [paramEval[0], paramEval[1]]

        return

    def invoke(self):

        self.status_ = EvaluationStatus.SUCCESS

        alpha = self.fixedParam_[0]
        beta = self.param_[0]
        gamma = self.fixedParam_[1]
        delta = self.param_[1]

        evaluation = zeros(self.dataShape_)

        odeFlow = lambda t, x: self.flow__(t, x, alpha, beta, gamma, delta)

        for n in range(self.dataShape_[0]):

            odeResult = solve_ivp(
                odeFlow, self.tBoundary_, self.x_[n, :], method='LSODA')

            if (odeResult.status != 0):

                print("forward map evaluation failed. Reason: \n"
                      + odeResult.message)

                self.status_ = EvaluationStatus.FAILURE

                self.evaluation_ = zeros(self.dataShape_)

                break

            evaluation[n, :] = odeResult.y[:, -1]

        self.evaluation_ = evaluation

    def full_solution(self, parameter, y0):

        paramEval = parameter.evaluate()

        beta = paramEval[0]
        delta = paramEval[1]

        alpha = self.fixedParam_[0]
        gamma = self.fixedParam_[1]

        odeFlow = lambda t, x: self.flow__(t, x, alpha, beta, gamma, delta)
        odeResult = solve_ivp(odeFlow, self.tBoundary_, y0, method='LSODA')

        if (odeResult.status != 0):

            print("forward map evaluation failed. aborting program.")
            print(odeResult.message)
            exit()

        return (odeResult.t, odeResult.y)


def generate_synthetic_data(parameter, solver, noiseVar):

    sig = sqrt(noiseVar)

    solver.interpolate(parameter)
    solver.invoke()

    measurement = solver.evaluation + sig * standard_normal(solver.dataShape)

    return Data(measurement)
