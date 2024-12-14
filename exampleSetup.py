from yagremcmc.model.interface import SolverInterface
from yagremcmc.model.evaluation import EvaluationStatus


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

    def explicit_posterior(self, theta):
        pass
