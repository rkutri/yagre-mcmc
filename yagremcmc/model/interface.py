from abc import ABC, abstractmethod
from yagremcmc.model.evaluation import EvaluationStatus
from yagremcmc.parameter.interface import ParameterInterface


class SolverInterface(ABC):
    """
    Interface for solvers providing evaluations of the forward model.

    This interface represents a solver that is associated with a specific
    forward model. The associated parameter is set via 'interpolate'.
    """

    @property
    @abstractmethod
    def status(self) -> EvaluationStatus:
        """
        Retrieves status of the last solver evaluation

        Returns:
            EvaluationStatus: The status lf the last evaluation
        """
        pass

    @property
    @abstractmethod
    def evaluation(self) -> Any:
        """
        Retrieves the result of the last evaluation performed by the solver.

        The type of the evaluation result depends on the forward model used
        and the specific solver implementation.

        Returns:
            Any: The result of the solver's evaluation.	The type is
                 model-dependent.
        """
        pass

    @abstractmethod
    def interpolate(self, parameter: ParameterInterface) -> None:
        """
        Prepare the solver for evaluation with the provided parameter object.

        Internally, set the solver to a state which corresponds to the forward
        model with this fixed parameter.

        Args:
            parameter (ParameterInterface): The parameter object used to
                                            configure the solver for the next
                                            evaluation.

        Returns:
            None
        """
        pass

    @abstractmethod
    def invoke(self) -> None:
        """
        Perform the evaluation of the forward model with the current state.

        Returns:
            None
        """
        pass
