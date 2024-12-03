import numpy as np

from yagremcmc.chain.interface import ChainDiagnostics
from yagremcmc.chain.transition import TransitionData


class DummyDiagnostics(ChainDiagnostics):

    def process(self, transitionData):
        return

    def print_diagnostics(self, logger):
        return

    def clear(self):
        return


class AcceptanceRateDiagnostics(ChainDiagnostics):

    def __init__(self) -> None:

        self._decisions = []
        self._lag = None

    @property
    def lag(self):
        return self._lag

    @lag.setter
    def lag(self, lag):
        self._lag = lag

    @property
    def decisions(self):
        return self._decisions

    def rolling_acceptance_rate(self):

        if self._lag is None:
            raise RuntimeError("lag of rolling acceptance rate not set")

        if len(self._decisions) < self._lag:
            raise RuntimeError("Insufficient data for rolling acceptance rate")

        return np.mean(np.array(self._decisions[-self._lag:]))

    def global_acceptance_rate(self):
        return np.mean(np.array(self._decisions))

    def process(self, transitionData):

        if transitionData.outcome == TransitionData.REJECTED:
            self._decisions.append(0)
        elif transitionData.outcome == TransitionData.ACCEPTED:
            self._decisions.append(1)
        else:
            raise RuntimeError("Invalid acceptance decision")

    def print_diagnostics(self, logger):

        rAccept = self.rolling_acceptance_rate()
        logger.info(f"  - rolling acceptance rate: {rAccept}")

    def clear(self):
        self._decisions = []


class MomentsDiagnostics(ChainDiagnostics):

    def __init__(self):

        self._mean = None
        self._margVar = None

    def update_mean(self, state):
        pass

    def update_marginal_variance(self, state):
        """
        Wellford's algorithm for stable accumulation of variances
        """
        pass

    @property
    def mean(self):
        return self._mean

    @property
    def marginalVariance(self):
        return self._margVar

    @property
    def conditionNumber(self):
        return np.max(self._margVar) / np.min(self._margVar)

    def print_diagnostics(self, logger):

        mvCond = self.conditionNumber

        logger.info(f"  - estimated mean: {self.mean}")
        logger.info(f"  - estimated condition number: {self.conditionNumber}")

    def process(self, transitionData):

        newState = transitionData.state.coefficient

        self.update_mean(newState)
        self.update_marginal_variance(newState)

    def clear(self):

        self._mean = None
        self._margVar = None


class FullDiagnostics(ChainDiagnostics):

    def __init__(self):

        self._dgns = [AcceptanceDiagnostics(), MomentsDiagnostics()]

    def print_diagnostics(self, logger):

        for dgn in self._dgns:
            dgn.print_diagnostics(logger)

    def process(self, transitionData):

        for dgn in self._dgns:
            dgn.process(transitionData)

    def clear(self):

        for dgn in self._dgns:
            dgn.clear()
