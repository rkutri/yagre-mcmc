class ForwardMap:

    def __init__(self, solver):

        self.solver_ = solver
        self.result_ = None

    @property
    def result(self):
        return self.result_

    def request_evaluation(self, parameter):

        self.solver_.configure(parameter)
        self.result_ = self.solver_.invoke()
