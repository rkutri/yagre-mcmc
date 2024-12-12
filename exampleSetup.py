from yagremcmc.model.interface import SolverInterface


class ExampleLinearModelSolver(SolverInterface):
    """
    Solver class for the computation of $y = \Lambda_{\theta} A x$, for 
    a fixed matrix $A$ and a diagonal matrix
    $\Lambda_{\theta} = \mathrm{diag}(\theta_1, \ldots, \theta_d)$, where
    the $\{\theta_i\}$ are the parameters of the model.
    """

    @property
    def status(self):
        pass

    @property
    def evaluation(self):
        pass

    def interpolate(self, parameter):
        pass

    def invoke(self):
        pass


