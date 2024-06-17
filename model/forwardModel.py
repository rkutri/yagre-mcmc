from enum import Enum


class EvaluationStatus(Enum):

    NONE = -1
    SUCCESS = 0
    FAILURE = 1


class ForwardModel:

    def __init__(self, solver):

        self.solver_ = solver

    def evaluate(self, parameter):

        self.solver_.interpolate(parameter)
        self.solver_.invoke()

        if (self.solver_.status == EvaluationStatus.SUCCESS):
            return self.solver_.evaluation

        else:
            raise Exception("Evaluation request failed.")
