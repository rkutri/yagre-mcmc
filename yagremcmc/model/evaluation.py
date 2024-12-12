from enum import Enum, unique


@unique
class EvaluationStatus(Enum):

    NONE = -1
    SUCCESS = 0
    FAILURE = 1
