from enum import Enum, unique


@unique
class EvaluationStatus(Enum):

    NONE = -1
    SUCCESS = 0
    FAILURE = 1


class AEMEvaluation:

    def __init__(self, fwdModelEval, logLikeEval):

        self._fwdModelEval = fwdModelEval
        self._logLikeEval = logLikeEval

    @property
    def forwardModelEvaluation(self):
        return self._fwdModelEval

    @property
    def logLikelihoodEvaluation(self):
        return self._logLikeEval
