from yagremcmc.model.interface import SolverInterface
from yagremcmc.model.evaluation import EvaluationStatus


class ExampleLinearModelSolver(SolverInterface):
    """
    Solver class for the computation of $G(\theta) = A \theta + b$, for
    fixed $A$, $b$ and the parameter $\theta = (\theta_1, \ldots, \theta_d)^T$
    of the model.
    """

    def __init__(self, A, b):

        self_.parameter = None
        self._A = A
        self._b = b

        self._status = EvaluationStatus.NONE

    @property
    def status(self):
        return self._status

    @property
    def evaluation(self):
        return self._evaluation

    def interpolate(self, parameter):
        self._parameter = parameter.evaluate()

    def invoke(self):
        return self._A @ self._parameter + self._b

    def explicit_posterior(self, theta):
        pass
