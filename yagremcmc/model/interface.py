from abc import ABC, abstractmethod
from typing import Any
from yagremcmc.model.forwardModel import EvaluationStatus
from yagremcmc.parameter import ParameterInterface


class SolverInterface(ABC):
    """
    Interface for solvers that compute evaluations of a forward model.

    Each solver corresponds internally to a forward model configured
    with a specific parameter set. This interface defines methods
    for retrieving the solver's status, the result of evaluations,
    and performing the evaluation process.
    """

    @property
    @abstractmethod
    def status(self) -> EvaluationStatus:
        """
        Retrieve the status of the last evaluation performed by the solver.

        Returns:
            EvaluationStatus: The status of the last evaluation, which may
                              indicate success, failure, or another state
                              depending on the implementation.
        """
        pass

    @property
    @abstractmethod
    def evaluation(self) -> Any:
        """
        Retrieve the result of the last evaluation performed by the solver.

        Returns:
            Any: The result of the solver's evaluation. The type of the result
                 is model-dependent
        """
        pass

    @abstractmethod
    def interpolate(self, parameter: ParameterInterface) -> None:
        """
        Prepare the solver for evaluation with the provided parameter object.

        This method configures the solver's internal state to correspond
        to the forward model with the specified parameter set.

        Args:
            parameter (ParameterInterface): The parameter object defining the
                                            configuration for the next
                                            evaluation.
        """
        pass

    @abstractmethod
    def invoke(self) -> None:
        """
        Perform the evaluation of the forward model using the current
        configuration.

        This method executes the forward model's computation with the
        parameter and state set by the `interpolate` method. It updates
        the solver's status and stores the evaluation result internally.
        """
        pass
