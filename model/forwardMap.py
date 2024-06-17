from enum import Enum

class EvaluationStatus(Enum):                                                   
                                                                                    
        NONE = -1                                                                   
        SUCCESS = 0                                                                 
        FAILURE = 1   

class ForwardMap:

    def __init__(self, solver):

        self.solver_ = solver

        self.evalRequest_ = None
        self.evaluation_ = None
        self.evalStatus_ = EvaluationStatus.NONE


    @property
    def evaluation(self):
        return self.evaluation_

    @property
    def status(self):
        self.evalStatus_

    @property
    def request(self):
        return self.evalRequest_

    @request.setter
    def request(self, request):
        self.evalRequest_ = request


    def evaluate(self):

        self.evalRequest_.submit(self.solver_)

        self.evaluation_ = self.solver_.result

