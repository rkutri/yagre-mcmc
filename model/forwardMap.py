class ForwardMap:

    def __init__(self, solver):

        self.solver_ = solver
        self.result_ = None

    @property
    def result(self):
        return self.result_

    def request_evaluation(self, parameter):

        solverConfig = self.solver_.to_config(parameter)
        self.result_ = self.solver_.invoke(solverConfig)
