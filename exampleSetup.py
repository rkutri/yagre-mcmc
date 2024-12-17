import numpy as np

from yagremcmc.model.interface import SolverInterface
from yagremcmc.model.evaluation import EvaluationStatus
from yagremcmc.parameter.vector import ParameterVector


class ExampleLinearModelSolver(SolverInterface):
    """
    Solver class for the computation of $G(\\theta) = A \\theta + b$,
    for fixed $A$, $b$ and the parameter
    $\\theta = (\\theta_1, \\ldots, \\theta_d)^T$ of the model.
    """

    def __init__(self, A, b):

        self._paramCoordinates = None
        self._A = A
        self._b = b

        self._status = EvaluationStatus.NONE
        self._evaluation = None

    @property
    def status(self):
        return self._status

    @property
    def evaluation(self):

        if self._status == EvaluationStatus.FAILURE:
            raise RuntimeError("Trying to retrieve failed evaluation.")
        if self._status == EvaluationStatus.NONE:
            raise RuntimeError("Trying to retrieve evaluation before it was "
                               "performed.")

        return self._evaluation

    def interpolate(self, parameter):
        self._paramCoordinates = parameter.coefficient

    def invoke(self):

        try:

            self._evaluation = self._A @ self._paramCoordinates + self._b
            self._status = EvaluationStatus.SUCCESS

        except Exception as e:

            self._status = EvaluationStatus.FAILURE
            print("WARNING: Forward Model evaluation failed: " + str(e))


def evaluate_posterior(mesh, likelihood, prior):

    posterior = np.zeros(mesh.shape[:2])

    for i in range(mesh.shape[0]):
        for j in range(mesh.shape[1]):
            theta = mesh[i, j]
            thetaParam = ParameterVector(theta)

            logPost = likelihood.evaluate_log(thetaParam) \
                + prior.density.evaluate_log(thetaParam)
            posterior[i, j] = np.exp(logPost)

    # normalise
    integral = np.sum(posterior)

    return posterior / integral
