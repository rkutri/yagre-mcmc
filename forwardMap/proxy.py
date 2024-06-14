from forwardMap.interface import ForwardMapInterface


class ForwardMapProxy(ForwardMapInterface):

    @property
    def evaluationRequest(self):
        pass

    
    def evaluate(self, x):
        """
        Turns the call into an EvaluationRequest, which is passed to the 
        solver.
        """



