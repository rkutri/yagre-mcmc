from enum import Enum
from model.interface import EvaluationRequestInterface


class EvaluateAtDesign(EvaluationRequestInterface):

    def __init__(self, parameter, config, design):
        pass

    @property
    @abstractmethod
    def result(self):
        pass

class EvaluatePointwise(EvaluationRequestInterface):
    pass


