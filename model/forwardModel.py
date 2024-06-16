from model.evaluation import EvaluationStatus
from model.forwardMap import ForwardMap


class ForwardModel:

    def __init__(self, solver):

        self.fwdMap_ = ForwardMap(solver)

    def evaluate(self, parameter):

        evalStatus = self.fwdMap_.request_evaluation(parameter)

        if (evalStatus == EvaluationStatus.SUCCESS):
            return self.fwdMap_.evaluation

        else:
            raise Exception("Evaluation request failed.")
