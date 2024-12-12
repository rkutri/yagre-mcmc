from enum import Enum, unique
from yagremcmc.model.interface import SolverInterface
from yagremcmc.model.evaluation import EvaluationStatus


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
